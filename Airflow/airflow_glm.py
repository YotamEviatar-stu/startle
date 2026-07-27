
import os
import numpy as np
import neurokit2 as nk
from scipy.signal import butter, filtfilt, lfilter, medfilt, resample

import extras.trial_epochs as trial_epochs
from Airflow import airflow_qc


# ── Layer 0 — Cache build (resample; not GLM-specific) ──────────────────────
# Moved here from airflow_amp_processor.py so the GLM path (primary) no longer
# depends on the BxB/legacy module for its own caching -- airflow_amp_processor.py
# now contains only its Layer 2 (apply_analysis_params + BxB scoring).

def _butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def _butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def _butter_lowpass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, data)

def process_session(mff_path, ratings_df, config):
    """ratings_df must be the FULL, unfiltered CSV (trial_epochs.load_all_trials_ratings)
    -- every row becomes a trial here, sound and no-sound alike.

    Event/epoch extraction always runs on the original full-rate, full-channel
    raw (read_raw_egi) before channel-pick/downsample -- DIN triggers must not
    be resampled or have other channels dropped before detection.

    Raises trial_epochs.TriggerAlignmentError on a D105/D{code} mismatch;
    callers must let this propagate rather than silently continue with an
    untrustworthy trial/CSV alignment.
    """
    import mne
    if config.VERBOSE:
        print(f"  Layer 1: Processing raw MFF ({os.path.basename(mff_path)}) ...")

    try:
        raw = mne.io.read_raw_egi(mff_path, preload=True, verbose=False)
    except Exception as e:
        print(f"  Error reading {mff_path}: {e}")
        return None

    native_sfreq = raw.info["sfreq"]
    trials = trial_epochs.build_trial_epochs(raw, ratings_df)  # raises on mismatch
    d101_sample = trial_epochs.first_session_marker_sample(raw)

    ch_names = raw.info["ch_names"]
    target_ch = next((c for c in ch_names if config.AIRFLOW_CHANNEL.lower() in c.lower()), None)
    if target_ch is None:
        if config.VERBOSE:
            print(f"  Channel '{config.AIRFLOW_CHANNEL}' not found in {ch_names}")
        return None

    raw.pick([target_ch])

    # ── Anti-alias lowpass only (native rate, before downsampling) ─────────────
    # pspm_resp_pp.m Stage 5 measures RA/RFR on the fully unfiltered `resp`
    # trace -- signal_raw is this project's stand-in for `resp`, so it must
    # not carry a highpass PsPM keeps out of that variable. The AIRFLOW_LOWPASS
    # anti-alias step stays: CACHE_SFREQ's Nyquist (12.5 Hz at 25 Hz) is below
    # AIRFLOW_LOWPASS (70 Hz), so this cutoff can't be realized after resample.
    raw.apply_function(
        lambda x: _butter_lowpass_filter(
            x.astype(np.float64), config.AIRFLOW_LOWPASS,
            native_sfreq, order=2),
        picks=0, channel_wise=True, verbose=False)

    raw.resample(sfreq=config.CACHE_SFREQ, npad="auto", verbose=False)

    cache_sfreq = raw.info["sfreq"]
    signal_raw  = raw.get_data(picks=0)[0]
    times_raw   = raw.times

    # Trigger sample indices are in native_sfreq space (from the un-resampled
    # raw); convert to seconds here so they're valid regardless of CACHE_SFREQ.
    trials_meta = []
    for t in trials:
        row = ratings_df.iloc[t["trial_index"]]
        trials_meta.append({
            "trial_index":        t["trial_index"],
            "code_value":         t["code_value"],
            "has_sound":          t["d110_sample"] is not None,
            "d105_time":          t["d105_sample"] / native_sfreq,
            "code_time":          t["code_sample"] / native_sfreq,
            "d110_time":          (t["d110_sample"] / native_sfreq
                                    if t["d110_sample"] is not None else None),
            "baseline_range_sec": tuple(s / native_sfreq for s in t["baseline_range"]),
            "response_range_sec": tuple(s / native_sfreq for s in t["response_range"]),
            "subjective_label":   row.get("subjective_label", np.nan),
            "image_type":         row.get("image_type", None),
            "image_detail":       row.get("file_name", None),
            "valence":            row.get("valenceRating", np.nan),
            "arousal":            row.get("arousalRating", np.nan),
        })

    payload = {
        "sfreq":       cache_sfreq,
        "channel":     target_ch,
        "signal_raw":  signal_raw,
        "times_raw":   times_raw,
        "trials_meta": trials_meta,
        "d101_time":   d101_sample / native_sfreq,
    }
    return payload


# ── STAGE 1 — Signal Conditioning (mixed: despike is a project addition; the
# filter cascade/downsample below matches pspm_resp_pp.m) ──────────────────

def despike_signal_raw(signal_raw, native_sfreq, k=50.0, max_run_sec=1.0):
    """Detect brief, extreme-amplitude samples in the native signal_raw and
    linearly interpolate over them, BEFORE any filtering.

    Why here and not left to the bandpass/z-score: phase1_filter_downsample's
    0.01 Hz highpass is a cascaded, zero-phase filtfilt, whose impulse
    response rings for hundreds of seconds. A single glitch sample that is
    invisible next to real breathing in signal_raw itself (e.g. RP06 eve:
    one native sample of -0.070 at t=503.44s, against a session MAD of
    0.00013 -- a ~540 session-robust-SD outlier lasting 0.12s) still turns
    into an ~86 SD excursion in raw_z spanning t=345-946s once it passes
    through that filter, dwarfing every real breath in the session and
    making both View 2 (session overview) and Phase 2 cycle detection
    unusable over that whole stretch.

    k=50 is a deliberately conservative gate: scanning every subject/session
    in this dataset, the largest genuine breath (sigh/cough included) never
    exceeds ~35 session-median-MAD units of signal_raw; k=50 sits above that
    ceiling with margin and far below actual glitches (RP06 eve: 540/520/88).
    max_run_sec caps how long a flagged run may be before it's left alone --
    a SUSTAINED large excursion (whole-session sensor contamination, e.g.
    DA01) is a different problem than a momentary glitch and belongs in
    SUBJECTS_EXCLUDE (airflow_config.py), not silent interpolation here.

    Returns (x_despiked, despiked_runs) where despiked_runs is a list of
    (start_idx, end_idx, start_sec, end_sec) for logging/inspection."""
    x = np.asarray(signal_raw, dtype=np.float64).copy()
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    if mad == 0:
        return x, []

    bad = np.abs((x - med) / mad) > k
    max_run = int(round(max_run_sec * native_sfreq))
    n = len(x)
    despiked_runs = []
    i = 0
    while i < n:
        if not bad[i]:
            i += 1
            continue
        j = i
        while j < n and bad[j]:
            j += 1
        run_len = j - i
        if run_len <= max_run:
            lo = x[i - 1] if i > 0 else (x[j] if j < n else med)
            hi = x[j] if j < n else (x[i - 1] if i > 0 else med)
            x[i:j] = np.linspace(lo, hi, run_len + 2)[1:-1]
            despiked_runs.append((i, j, i / native_sfreq, j / native_sfreq))
        i = j
    return x, despiked_runs


def phase1_filter_downsample(signal_raw, native_sfreq, target_sfreq=10.0,
                              bp_low=0.01, bp_high=0.6, zscore=True,
                              despike=True, despike_k=50.0, despike_max_run_sec=1.0):
    """Despike (see despike_signal_raw) + mean-center + TWO cascaded
    1st-order Butterworth filters (lowpass bp_high, highpass bp_low, each
    filtfilt) + downsample + (optional) robust z-score. Matches
    pspm_resp_pp.m Stage 1 exactly (references/pspm-preprocessing.md):
    filt.lpfreq=0.6/lporder=1, filt.hpfreq=0.01/hporder=1, direction='bi',
    filt.down=10 -- downsampling happens AFTER filtering, anti-aliased by
    the SAME 0.6 Hz lowpass that isolates the respiration band (0.6 Hz sits
    8.3x below the 5 Hz Nyquist at target_sfreq=10, ample margin) -- no
    separate anti-alias-only filter needed for a signal this narrowband.
    Runs on the cached signal_raw (25 Hz); no MFF reload needed.

    Z-scoring uses robust median/MAD (not mean/std) to handle outlier spikes
    without inflating the noise floor. Outlier spikes barely affect median/MAD,
    so genuine breathing keeps proper scale."""
    x = np.asarray(signal_raw, dtype=np.float64)
    if despike:
        x, _ = despike_signal_raw(x, native_sfreq, k=despike_k, max_run_sec=despike_max_run_sec)
    x = x - np.nanmean(x)

    nyq = native_sfreq / 2.0
    b_lp, a_lp = butter(1, bp_high / nyq, btype="low")
    xf = filtfilt(b_lp, a_lp, x)
    b_hp, a_hp = butter(1, bp_low / nyq, btype="high")
    xf = filtfilt(b_hp, a_hp, xf)

    n_target = int(round(len(xf) * target_sfreq / native_sfreq))
    x10 = resample(xf, n_target)

    if zscore:
        med = float(np.nanmedian(x10))
        mad = float(np.nanmedian(np.abs(x10 - med)))
        robust_sd = mad * 1.4826
        if robust_sd > 0:
            x10 = (x10 - med) / robust_sd

    return x10, target_sfreq


# ── STAGE 2 — Cycle Detection [PsPM-faithful] ───────────────────────────────

def detect_cycles(raw_z, sfreq, signal_raw, native_sfreq,
                   median_win_sec=1.0, refractory_sec=1.0):
    """Detect inspiration onsets on raw_z, then extract one cycle per breathing
    cycle -- amplitude (RA/RFR) is measured on signal_raw, not raw_z.

    This mirrors pspm_resp_pp.m's separation of two variables: `newresp`
    (mean-centred, filtered, downsampled -- used ONLY to find zero-crossing
    timestamps, lines 92-94/96-111) vs `resp` (the untouched signal, read back
    at native sample rate to measure amplitude, lines 140-141/150-151). PsPM
    never measures range() on the filtered detection copy. signal_raw (25 Hz,
    pre-Phase-1) is the closest available stand-in for PsPM's `resp` given this
    project only caches at 25 Hz (never full native MFF rate, see
    airflow_config.py CACHE_SFREQ) -- it is not filtered/z-scored the way raw_z
    is, so it does not clip fast inspiratory-flow peaks the way measuring on
    raw_z would (see references/pspm-preprocessing.md in the
    pspm-respiration-audit skill for why this matters more for a flow sensor
    than for PsPM's bellows/chest-strap signal).

    Returns list of dicts with onset_time, assign_time, RP, RA, RFR."""
    # Bandpass+downsample already applied in phase1_filter_downsample --
    # raw_z arrives pre-filtered, matching pspm_resp_pp.m Stage 1 exactly
    # (see that function's docstring). Only median-filtering + zero-crossing
    # detection remain here (still this function's own Stage 2 job).

    # Step 1: Smooth with median filter, then detect zero-crossings
    # Median filter removes wiggles so only true breath oscillations trigger crossings
    win = int(round(median_win_sec * sfreq))
    win = win + 1 if win % 2 == 0 else win  # Ensure odd kernel size
    win = max(win, 1)
    xm = medfilt(raw_z, kernel_size=win)

    # Detect negative zero-crossings (high to low) = inspiration onsets
    # (transition from expiration to inspiration)
    signs = np.sign(xm)
    signs[signs == 0] = 1  # Treat zero as positive
    cross_idx = np.flatnonzero((signs[:-1] > 0) & (signs[1:] < 0)) + 1

    # Step 2: Enforce refractory period (minimum gap between onsets)
    # pspm_resp_pp.m lines 113-116: ibi = diff(respstamp); indx = find(ibi < 1);
    # respstamp(indx + 1) = []. Single pass over the ORIGINAL candidate list --
    # a candidate is dropped if its gap to the PRECEDING candidate (not the
    # last kept one) is under refractory_sec. E.g. [0.0, 0.9, 1.7]: ibi =
    # [0.9, 0.8], both < 1s -> both 0.9 and 1.7 dropped -> result [0.0].
    refractory_samples = refractory_sec * sfreq
    cross_idx = np.asarray(cross_idx)
    if len(cross_idx) > 1:
        ibi = np.diff(cross_idx)
        drop = np.zeros(len(cross_idx), dtype=bool)
        drop[1:] = ibi < refractory_samples
        onsets = cross_idx[~drop].tolist()
    else:
        onsets = cross_idx.tolist()

    # Step 3: Extract features per cycle (each cycle = one breathing cycle).
    # RP comes from the raw_z-clock onset times (detection only). RA/RFR are
    # measured on signal_raw at native_sfreq -- pspm_resp_pp.m line 140:
    # `win = ceil(respstamp(k)*sr) : ceil(respstamp(k+1)*sr)`. A MATLAB colon
    # range is inclusive of both ends; Python slicing excludes the stop index,
    # so +1 is added below to match.
    n_native = len(signal_raw)
    cycles = []
    for i in range(len(onsets) - 1):
        start, end = onsets[i], onsets[i + 1]
        onset_time, next_onset_time = start / sfreq, end / sfreq
        rp = next_onset_time - onset_time

        i0 = int(np.ceil(onset_time * native_sfreq))
        i1 = min(int(np.ceil(next_onset_time * native_sfreq)) + 1, n_native)
        seg = signal_raw[i0:i1]
        if len(seg) == 0:
            continue
        # RP = Respiration Period (seconds)
        # RA = Respiration Amplitude (peak-to-trough, in native signal_raw units)
        # RFR = Respiration Flow Rate = RA/RP (native units per second)
        ra = float(np.max(seg) - np.min(seg))
        rfr = ra / rp if rp > 0 else np.nan
        cycles.append({
            "onset_time":  onset_time,
            "assign_time": next_onset_time,   # value assigned to the FOLLOWING onset
            "RP": rp, "RA": ra, "RFR": rfr,
        })
    return cycles


# ── STAGE 3 — Artifact Layer [PROJECT ADDITION] — Hampel amplitude gate ────

def _hampel_flag_cycles(cycles, k):
    """Flag breathing cycles whose RA is an UPPER outlier by the Hampel
    identifier, using the session's own robust spread:

        cutoff  = median(RA) + k * (MAD(RA) * 1.4826)
        flagged = RA > cutoff

    Upper bound only -- targets amplitude SPIKES (a cough / gross movement /
    sensor swing read as one giant "breath") and never flags a shallow real
    breath (that lower-side over-rejection was the failure mode of the earlier
    session-median *ratio* attempt; see GLM_ARTIFACT_METHOD in
    airflow_config.py). Because the bound is relative to this session's own
    median/MAD, it does NOT catch a uniformly-corrupted session (every cycle
    huge -> median huge too) -- that limitation is documented alongside the
    config setting.

    Returns a boolean array aligned to `cycles` (all False when there are fewer
    than 3 cycles, or the MAD is zero, i.e. nothing to compare against)."""
    n = len(cycles)
    if n < 3:
        return np.zeros(n, dtype=bool)
    ras = np.array([c["RA"] for c in cycles], dtype=float)
    med = float(np.median(ras))
    mad = float(np.median(np.abs(ras - med))) * 1.4826
    if not (mad > 0):
        return np.zeros(n, dtype=bool)
    return ras > (med + k * mad)


# ── STAGE 3 — Artifact Layer [PROJECT ADDITION] — NK2 cross-check ──────────
# detect_cycles: finds zero-crossings → produces RP/RA/RFR scores (but accepts noise)
# NK2: validates real breathing exists (checks amplitude/structure)
# Logic: If detect_cycles found cycles BUT NK2 found none → reject (artifact gate)


def _nk2_peak_trough_times(raw_z, sfreq, config):
    """NK2 (khodadad2018) peak/trough times, in seconds -- session-wide,
    computed once. Used ONLY as a no_cycles_found cross-check (see config);
    RP/RA/RFR scoring is completely untouched and still comes from detect_cycles."""
    rsp_signals, _ = nk.rsp_process(
        raw_z.astype(float), sampling_rate=int(round(sfreq)),
        method=config.RSP_CLEAN_METHOD, method_cleaning=config.RSP_PEAK_METHOD_CLEANING)
    peaks   = np.flatnonzero(rsp_signals["RSP_Peaks"].to_numpy())
    troughs = np.flatnonzero(rsp_signals["RSP_Troughs"].to_numpy())
    return peaks / sfreq, troughs / sfreq


# ── STAGE 4 — Continuous Series (mixed: interpolation/filtering is a project
# addition; the per-metric HP/LP cutoffs match PsPM's sensitivity filter) ──

def build_continuous_series(cycles, n_samples, sfreq, hp=0.001, lp=1.0, missing_spans=None):
    """Interpolate discrete per-cycle RP/RA/RFR knots (assign_time -> value) to a
    continuous 10 Hz series, then apply a causal per-metric sensitivity filter.

    WHY: GLM fitting needs continuous time series to match trial windows against
    canonical response functions -- not reconstructing waveform shape, but
    reconstructing how breathing *features* (RP/RA/RFR) evolve in time.
    """
    times_full = np.arange(n_samples) / sfreq
    series = {}

    # Unidirectional bandpass "sensitivity filter" (from the paper).
    # `hp` may be a scalar (same cutoff for all three metrics) OR a per-metric
    # dict {"RP":.., "RA":.., "RFR":..}: PsPM uses 0.01 Hz for RP and 0.001 Hz
    # for RA/RFR (see GLM_FINAL_HP in airflow_config.py). The low-pass is shared.
    # lfilter (not filtfilt) preserves causality: no future-looking.
    nyq = sfreq / 2.0

    for metric in ("RP", "RA", "RFR"):
        if len(cycles) < 2:
            series[metric] = np.full(n_samples, np.nan)
            continue

        hp_m = hp[metric] if isinstance(hp, dict) else hp
        b, a = butter(1, [hp_m / nyq, lp / nyq], btype="band")

        knot_t = np.array([c["assign_time"] for c in cycles])
        knot_v = np.array([c[metric] for c in cycles])
        order = np.argsort(knot_t)
        knot_t, knot_v = knot_t[order], knot_v[order]
        interp = np.interp(times_full, knot_t, knot_v, left=knot_v[0], right=knot_v[-1])
        filtered = lfilter(b, a, interp)

        # Blank out dropped-cycle spans AFTER filtering (PsPM model.missing):
        # the interpolated line was needed only to keep the causal IIR filter's
        # recursion well-behaved; NaN'ing pre-filter would poison every sample
        # after it via the filter's own feedback. fit_trial_glm/
        # fit_pooled_session_glm already mask NaN out of their regressions.
        if missing_spans:
            for start_t, end_t in missing_spans:
                mask = (times_full >= start_t) & (times_full < end_t)
                filtered[mask] = np.nan

        series[metric] = filtered

    return series, times_full


# ── STAGE 5 — GLM Fit [PsPM-faithful] ───────────────────────────────────────

def generate_canonical_rf(metric, rf_params, tw):
    """Canonical Response Function: Gaussian h(t) = exp(-(t-τ)² / (2σ²)).

    IMPORTANT: `tw` is the trial's FULL time window -- every sample from
    baseline start through response end (see fit_trial_glm below), not just
    a single instant. The template is a whole curve (rises toward τ, peaks at
    τ, falls away afterward) compared against the whole observed curve -- τ
    is just where this curve happens to peak, not the only point that matters.
    """
    tau, sigma = rf_params[metric]
    return np.exp(-((tw - tau) ** 2) / (2 * sigma ** 2))


def _spm_orth_columns(bs):
    """Serial Gram-Schmidt (spm_orth.m), no renormalization -- used for the
    post-convolution orthogonalization pspm_glm.m §14.4 applies to the convolved
    [CRF, dCRF] columns per condition. Column 0 kept as-is; each later column has
    its least-squares projection onto the earlier kept columns removed; a column
    that collapses to ~0 (collinear) is zeroed (spm_orth 'pad' default)."""
    n_cols = bs.shape[1]
    cols = [bs[:, 0].copy()]
    for i in range(1, n_cols):
        d = bs[:, i].copy()
        x = np.column_stack(cols)
        d = d - x @ (np.linalg.pinv(x) @ d)
        cols.append(d if np.sum(np.abs(d)) > 1e-12 else np.zeros_like(d))
    return np.column_stack(cols)


def orthogonalize_and_normalize_basis(bs):
    """spm_orth.m (see _spm_orth_columns) + the unit-range normalization line
    every pspm_bf_r*rf_e.m basis function applies before returning: `bs =
    spm_orth(bs); bs = bs ./ (max(bs) - min(bs))`. Applied to [CRF, dCRF/dt]
    ONLY -- the intercept column is added separately by the caller and is
    never orthogonalized against these, matching PsPM (spm_orth runs inside
    the basis-function file, before pspm_glm ever adds an intercept)."""
    n_cols = bs.shape[1]
    ortho_cols = [bs[:, 0].copy()]
    for i in range(1, n_cols):
        d = bs[:, i].copy()
        x = np.column_stack(ortho_cols)
        coeffs = np.linalg.pinv(x) @ d
        d = d - x @ coeffs
        if np.sum(np.abs(d)) > 1e-12:
            ortho_cols.append(d)
        else:
            ortho_cols.append(np.zeros_like(d))
    bs = np.column_stack(ortho_cols)

    ranges = bs.max(axis=0) - bs.min(axis=0)
    ranges[ranges == 0] = 1.0
    return bs / ranges


def fit_trial_glm(y_window, tw, metric, rf_params, use_derivative):
    """Solve y_actual = β₀ + β₁·CRF(t) + β₂·dCRF/dt over the trial's ENTIRE
    [baseline_start, response_end) window (see run_glm_scoring: i0 =
    baseline_range_sec[0], i1 = response_range_sec[1] -- the full epoch, not
    a peak-only slice). Every point in y_window contributes to the
    least-squares fit, and the CRF template spans that same whole window, so
    a high score (β₁) requires the OBSERVED curve's whole shape -- not just
    its highest point -- to track the template's rise-peak-fall pattern. A
    trial with a big peak at the wrong time will NOT automatically score high.

    β₀ (intercept) absorbs the pre-stimulus baseline level, so β₁ reflects
    template-matching strength after accounting for the trial's own baseline,
    not raw response magnitude.
    """
    reg = generate_canonical_rf(metric, rf_params, tw)
    bf_cols = [reg]
    if use_derivative.get(metric, False):
        bf_cols.append(np.gradient(reg, tw))
    bs = np.column_stack(bf_cols)
    # Orthogonalize + normalize the basis set (CRF [+ derivative]) BEFORE
    # adding the intercept -- matches pspm_bf_r*rf_e.m exactly (spm_orth then
    # unit-range normalization), see orthogonalize_and_normalize_basis above.
    bs = orthogonalize_and_normalize_basis(bs)
    X = np.column_stack([np.ones_like(tw)] + [bs[:, i] for i in range(bs.shape[1])])

    valid = np.isfinite(y_window) & np.all(np.isfinite(X), axis=1)
    if valid.sum() < X.shape[1] + 1:
        return np.nan, None

    beta = np.linalg.pinv(X[valid]) @ y_window[valid]
    fitted = X @ beta
    return float(beta[1]), fitted


# ── STAGE 5 — GLM Fit [PsPM-faithful] — pooled session-wide (pspm_glm.m) ───

def fit_pooled_session_glm(trials, series, sfreq, config, kernel_dur_sec=30.0):
    """PsPM's pooled, session-wide GLM (pspm_glm.m), as an alternative to the
    per-trial fit in fit_trial_glm. Selected by config.GLM_ESTIMATION ==
    "pooled_session".

    Instead of one regression per trial, this builds ONE design matrix for the
    entire session and solves it ONCE per metric:

      y(t)  =  β0·1  +  Σ_c [ β_c·(δ_c * CRF)(t)  +  γ_c·(δ_c * dCRF)(t) ]

    where c ranges over the session's CONDITIONS (valence label 1=Neg / 2=Neu,
    taken over the session's ACCEPTED trials only), δ_c is that condition's event
    train (a unit impulse at every accepted trial's onset_time), and CRF/dCRF are
    the same orthogonalized, unit-range canonical basis used per-trial. β_c (the
    CRF column's weight) is that condition's score — ONE number per condition per
    session, "similar to standard analysis of fMRI data" (pspm_glm.m,
    GLM_METHOD_FOUNDATIONS.md §5).

    Each accepted trial is written its own (session, condition) β_c as
    score_<metric>; rejected trials keep NaN and contribute no event to the
    design. Because the confirmatory cell averages score_<metric> over a
    session's accepted trials, that mean collapses to a trial-count-weighted mean
    of the Neg/Neu β_c — one value per session, Evening vs Morning.

    Faithfulness notes vs pspm_glm.m:
    - PsPM mean-centres the convolved design (model.centering defaults to 1); the
      convolved columns are mean-centred over the valid samples here before the
      intercept is prepended.
    - The basis is orthogonalized (spm_orth) + unit-range normalized BEFORE
      convolution — identical to the per-trial path (orthogonalize_and_normalize_basis).
    - Fixed latency (model.latency='fixed'): the basis is convolved at the literal
      event onset, no dictionary latency search.
    Runs off the SAME continuous `series` and SAME cache as the per-trial path —
    Layer 2, no FORCE_RELOAD.
    """
    accepted = [t for t in trials if not t.get("rejected")]
    if not accepted:
        return
    n_samples = len(series["RA"])

    # PsPM's basis functions (pspm_bf_r*rf_e.m) are defined over a fixed support
    # from -10 s to +30 s relative to event onset, orthogonalized + unit-range
    # normalized over THAT window before convolution. Sample the kernel over the
    # same support; `onset_off` is the sample index of t=0 within the kernel, used
    # to align the convolution so a trial's response peaks tau s AFTER its onset
    # (and the small pre-onset ramp lands before it), not kernel_dur late.
    kernel_start = -10.0
    tk = np.arange(kernel_start, kernel_dur_sec, 1.0 / sfreq)
    onset_off = int(round(-kernel_start * sfreq))

    for metric in ("RP", "RA", "RFR"):
        y = np.asarray(series[metric], dtype=np.float64)

        # Canonical basis kernel (orthogonalized + unit-range), PsPM -10..+30 support.
        crf = generate_canonical_rf(metric, config.GLM_RF_PARAMS, tk)
        cols = [crf]
        if config.GLM_USE_DERIVATIVE.get(metric, False):
            cols.append(np.gradient(crf, tk))
        basis = orthogonalize_and_normalize_basis(np.column_stack(cols))

        # Per-metric causal sensitivity high-pass for the DESIGN matrix
        # (pspm_glm.m §14.2, L577-579: convolved columns are high-pass filtered
        # with the modality filter, low-pass OFF; the data got HP+LP in
        # build_continuous_series). Matches the per-modality HP split.
        hp_m = config.GLM_FINAL_HP[metric] if isinstance(config.GLM_FINAL_HP, dict) else config.GLM_FINAL_HP
        b_hp, a_hp = butter(1, hp_m / (sfreq / 2.0), btype="high")

        # One event train per condition (valence label); one design column per
        # (condition, basis-column). col_is_crf[j] marks the score-bearing column.
        onsets_by_cond = {}
        for t in accepted:
            onsets_by_cond.setdefault(t["condition"], []).append(t["onset_time"])

        reg_cols, col_cond, col_is_crf = [], [], []
        for cond in sorted(onsets_by_cond):
            delta = np.zeros(n_samples)
            for ot in onsets_by_cond[cond]:
                idx = int(round(ot * sfreq))
                if 0 <= idx < n_samples:
                    delta[idx] = 1.0
            # Build this condition's convolved block, then follow PsPM's order:
            # convolve (§14.2) -> HP filter (§14.2) -> mean-centre (§14.3) ->
            # orthogonalize the CRF/dCRF columns against each other (§14.4).
            block_cols = []
            for bi in range(basis.shape[1]):
                # np.convolve places the kernel's first sample (t=kernel_start) at
                # each impulse; slice from onset_off so output[p] holds the kernel
                # at relative time (p - onset)/sfreq, i.e. onset aligned to t=0.
                conv = np.convolve(delta, basis[:, bi])[onset_off:onset_off + n_samples]
                conv = lfilter(b_hp, a_hp, conv)          # §14.2 design HP (causal)
                conv = conv - conv.mean()                 # §14.3 mean-centering
                block_cols.append(conv)
            block = np.column_stack(block_cols)
            if block.shape[1] > 1:
                block = _spm_orth_columns(block)          # §14.4 post-conv orth
            for bi in range(block.shape[1]):
                reg_cols.append(block[:, bi])
                col_cond.append(cond)
                col_is_crf.append(bi == 0)

        beta_by_cond = {c: np.nan for c in onsets_by_cond}
        full_fitted = None
        if reg_cols:
            Xr = np.column_stack(reg_cols)   # columns already HP-filtered + centred
            valid = np.isfinite(y) & np.all(np.isfinite(Xr), axis=1)
            if valid.sum() >= Xr.shape[1] + 2:
                X = np.column_stack([np.ones(n_samples), Xr])
                beta = np.linalg.pinv(X[valid]) @ y[valid]
                for j in range(len(col_cond)):
                    if col_is_crf[j]:
                        beta_by_cond[col_cond[j]] = float(beta[1 + j])
                full_fitted = X @ beta

        for t in accepted:
            t[f"score_{metric.lower()}"] = beta_by_cond.get(t["condition"], np.nan)
            if metric == "RA":
                i0, i1 = t["win_start_idx"], t["win_end_idx"]
                t["series_ra"] = y[i0:i1]
                t["fitted_ra"] = full_fitted[i0:i1] if full_fitted is not None else None


# ── STAGE 6 — Orchestration & Rejection Gates [PROJECT ADDITION] ───────────

def run_glm_scoring(sessions_cache, config):
    """INPUT:  sessions_cache (Layer 1 cache: 25Hz signal + trial metadata)
    OUTPUT: trials_data[subj][sess_key] = list of trial dicts with:
              • score_rp, score_ra, score_rfr (β₁ values)
              • rejected, rejection_reason
              • epoch_anal, times_anal (for plots)
    """
    trials_data = {}
    traces = {}
    qc_info = {}
    subjects_exclude = getattr(config, "SUBJECTS_EXCLUDE", {})
    artifact_method  = getattr(config, "GLM_ARTIFACT_METHOD", "hampel_reject_trials")
    artifact_k       = getattr(config, "GLM_ARTIFACT_K", 3.5)
    # SUBJECTS_EXCLUDE is ALWAYS honoured, regardless of GLM_ARTIFACT_METHOD:
    # it parks hand-flagged whole-session contamination (e.g. DA01's session-wide
    # saturation) that the Hampel bound structurally cannot catch, because that
    # bound is relative to each session's OWN median/MAD (see _hampel_flag_cycles
    # docstring and GLM_ARTIFACT_METHOD in airflow_config.py). GLM_ARTIFACT_METHOD
    # then controls only the per-trial/per-cycle handling applied to every
    # session that ISN'T parked here. Keys may be "SUBJ" (whole subject) or
    # "SUBJ/sess" (one session).
    for subj, sess_dict in sorted(sessions_cache.items()):
        if subj in subjects_exclude:
            continue
        trials_data[subj] = {}
        traces[subj] = {}
        qc_info[subj] = {}

        for sess_key, sess in sess_dict.items():
            if f"{subj}/{sess_key}" in subjects_exclude:
                continue
            signal_raw   = sess["signal_raw"]
            native_sfreq = float(sess["sfreq"])
            trials_meta  = sess["trials_meta"]

            qc_enabled = getattr(config, "QC_ENABLED", False)
            qc_spans = []
            if qc_enabled:
                qc = airflow_qc.qc_session(signal_raw, native_sfreq, config)
                signal_for_pipeline = qc["signal_clean"]
                qc_spans = list(qc["masked_spans"])
                qc_info[subj][sess_key] = {
                    "breath_size": qc["breath_size"],
                    "masked_spans": qc_spans,
                    "masked_sec": qc["masked_sec"],
                }
            else:
                signal_for_pipeline = signal_raw

            raw_z, sfreq = phase1_filter_downsample(
                signal_for_pipeline, native_sfreq,
                target_sfreq=config.GLM_TARGET_SFREQ,
                bp_low=config.GLM_CYCLE_BANDPASS[0],
                bp_high=config.GLM_CYCLE_BANDPASS[1],
                zscore=config.GLM_ZSCORE_RAW,
                despike=(not qc_enabled) and getattr(config, "GLM_DESPIKE_ENABLED", True),
                despike_k=getattr(config, "GLM_DESPIKE_K", 50.0),
                despike_max_run_sec=getattr(config, "GLM_DESPIKE_MAX_RUN_SEC", 1.0),
            )
            n_samples = len(raw_z)

            cycles = detect_cycles(
                raw_z, sfreq, signal_for_pipeline, native_sfreq,
                median_win_sec=config.GLM_MEDIAN_WIN_SEC,
                refractory_sec=config.GLM_REFRACTORY_SEC,
            )

            # ── Artifact handling (config.GLM_ARTIFACT_METHOD) ───────────────
            # hampel_drop_cycles : remove amplitude-spike cycles BEFORE the
            #   Phase-3 series is interpolated, so the spike never reaches any
            #   RA/RFR series or GLM fit (the cycle_gap gate then rejects any
            #   trial left with too long a dead stretch).
            # hampel_reject_trials : keep the cycles, but remember the flagged
            #   onset times so any trial whose window contains one is rejected
            #   below (reason "amplitude_artifact").
            # manual_exclude : no cycle-level action beyond the SUBJECTS_EXCLUDE
            #   skip above (which now applies to every method, not just this
            #   one) -- leaves surviving sessions' cycles/results unchanged.
            flagged_onset_times = []
            missing_spans = list(qc_spans)

            max_ratio = getattr(config, "QC_MAX_BREATH_RATIO", None)
            if max_ratio is not None:
                deep = airflow_qc.flag_deep_breaths(cycles, max_ratio)
                missing_spans += [(c["onset_time"], c["assign_time"])
                                  for c, bad in zip(cycles, deep) if bad]
                cycles = [c for c, bad in zip(cycles, deep) if not bad]
                if subj in qc_info and sess_key in qc_info[subj]:
                    qc_info[subj][sess_key]["deep_breaths"] = int(deep.sum())
            if artifact_method in ("hampel_drop_cycles", "hampel_reject_trials"):
                flags = _hampel_flag_cycles(cycles, artifact_k)
                if artifact_method == "hampel_drop_cycles":
                    # extend, not replace: qc_spans are already in missing_spans
                    # Dropped cycle's own [onset, assign) span is where its RA/RP/RFR
                    # were measured -- that's what's untrustworthy, not the
                    # surrounding real breaths used to bridge the interpolation.
                    missing_spans += [(c["onset_time"], c["assign_time"])
                                      for c, bad in zip(cycles, flags) if bad]
                    cycles = [c for c, bad in zip(cycles, flags) if not bad]
                else:  # hampel_reject_trials
                    flagged_onset_times = [c["onset_time"]
                                           for c, bad in zip(cycles, flags) if bad]

            # Session crop: GLM_PRE_FIXATION_SEC before the first trial's own
            # fixation onset (d105_time) through GLM_POST_CODE_SEC after the
            # last trial's picture-code onset (code_time) -- everything outside
            # this span is excluded from the pooled-session GLM's design/data.
            d105_times = [m["d105_time"] for m in trials_meta if m.get("d105_time") is not None]
            code_times = [m["code_time"] for m in trials_meta if m.get("code_time") is not None]
            win_start = win_end = None
            if d105_times and code_times:
                win_start = min(d105_times) - config.GLM_PRE_FIXATION_SEC
                win_end   = max(code_times) + config.GLM_POST_CODE_SEC
                n_samples = min(n_samples, int(np.ceil(max(win_end, 0.0) * sfreq)))
                if win_start > 0:
                    missing_spans = missing_spans + [(0.0, win_start)]

            series, times_full = build_continuous_series(
                cycles, n_samples, sfreq,
                hp=config.GLM_FINAL_HP, lp=config.GLM_FINAL_LP,
                missing_spans=missing_spans,
            )

            # NK2 cross-check for no_cycles_found -- same detector/config the
            # BxB path already applies identically to every subject (see
            # config for why this replaced session-median ratio thresholds).
            nk2_peak_times, nk2_trough_times = _nk2_peak_trough_times(raw_z, sfreq, config)

            d110_secs = np.array([m["d110_time"] for m in trials_meta
                                   if m["d110_time"] is not None])
            traces[subj][sess_key] = {
                "signal": raw_z, "sfreq": sfreq, "startle_samps_sec": d110_secs,
                "qc": qc_info[subj].get(sess_key),
                "signal_clean": signal_for_pipeline, "native_sfreq": native_sfreq,
                "scored_n_samples": n_samples,
                "scored_start_sec": win_start,
                "scored_end_sec": win_end,
            }

            trials = []
            for meta in trials_meta:
                # i0:i1 = the FULL trial window fed to fit_trial_glm (Phase 4) --
                # from baseline_range_sec[0] (baseline START) to
                # response_range_sec[1] (response END). The GLM score is a
                # regression over every sample in this whole span, not a
                # peak-only read -- see fit_trial_glm's docstring.
                b0, _ = meta["baseline_range_sec"]
                _, r1 = meta["response_range_sec"]
                onset_time = meta["code_time"]

                i0 = max(int(round(b0 * sfreq)), 0)
                i1 = min(int(round(r1 * sfreq)), n_samples)
                if i1 <= i0:
                    continue

                # rate_artifact must use the RAW per-cycle period, not the
                # Phase-3 causally-filtered RP series: that filter's 0.001 Hz
                # high-pass removes the DC level (typical breath period), so
                # the filtered series oscillates around ZERO -- 60/near-zero
                # blows up to bogus "rates" in the thousands of bpm. The
                # filtered series is correct as GLM input (Phase 4), but the
                # rejection gate needs actual physiological plausibility.
                cycles_in_window = [c for c in cycles if b0 <= c["onset_time"] < r1]
                rate_max_in_window = (
                    max(60.0 / c["RP"] for c in cycles_in_window if c["RP"] > 0)
                    if cycles_in_window else np.nan
                )

                # cycle_gap input: longest dead stretch BETWEEN detected onsets.
                # Window-edge segments (window-start -> first onset, last onset ->
                # window-end) are deliberately excluded: a window boundary landing
                # mid-breath is not an apnea, and including those edges fabricated
                # >12 s "gaps" in normally-breathing trials (rmax ~16 bpm) that the
                # manual flags mark as good (extras/trials_flagged). Trials with <2
                # detected onsets get NaN here and fall to no_cycles_found/flat_*.
                onset_times = [c["onset_time"] for c in cycles_in_window]
                max_gap_sec = (float(np.max(np.diff(onset_times)))
                               if len(onset_times) >= 2 else np.nan)

                # NK2 cross-check: real pre-onset peak (baseline) AND real
                # post-onset trough-then-peak (response) -- same has_cycles
                # structure the BxB path requires. Independent of the custom
                # detector's own cycle count above.
                nk2_has_cycles = (
                    np.any((nk2_peak_times >= b0) & (nk2_peak_times <= onset_time))
                    and np.any((nk2_trough_times > onset_time) & (nk2_trough_times < r1))
                )

                trials.append({
                    "trial_index":      meta["trial_index"],
                    "code_value":       meta["code_value"],
                    "has_sound":        meta["has_sound"],
                    "d110_rel":         (meta["d110_time"] - onset_time
                                          if meta["d110_time"] is not None else None),
                    "subjective_label": meta["subjective_label"],
                    "label":            meta["subjective_label"],
                    "valence":          meta["valence"],
                    "arousal":          meta["arousal"],
                    "image_detail":     meta["image_detail"],
                    "image_type":       meta.get("image_type", ""),
                    "win_start_idx":    i0,
                    "win_end_idx":      i1,
                    "onset_time":       onset_time,
                    "times_wide":       (times_full[i0:i1] - onset_time).astype(np.float32),
                    "n_cycles_in_window": len(cycles_in_window),
                    "max_gap_sec":      max_gap_sec,
                    "nk2_has_cycles":   bool(nk2_has_cycles),
                    # True only under GLM_ARTIFACT_METHOD="hampel_reject_trials"
                    # when a Hampel-flagged amplitude spike falls in this window
                    # (empty list -> always False for the other two methods).
                    "amp_artifact":     bool(any(b0 <= ot < r1 for ot in flagged_onset_times)),
                    "baseline_std":     None,
                    "baseline_mean":    None,
                    "response_std":     None,
                    "rate_max_in_window": rate_max_in_window,
                    "shape_msd":        None,
                    "score_rp":         None,
                    "score_ra":         None,
                    "score_rfr":        None,
                    "fitted_ra":        None,
                    "series_ra":        None,
                    "rejected":         None,
                    "rejection_reason": "",
                    "epoch_anal":       None,
                    "times_anal":       None,
                })

            _run_glm_analysis(trials, raw_z, series, sfreq, config)
            trials_data[subj][sess_key] = trials

    return trials_data, traces


def _run_glm_analysis(trials, raw_z, series, sfreq, config):
    """PASS 1: labels, per-trial baseline stats, session-level stats, shape
    centroid. PASS 2: rejection gates, then (per_trial estimation only) the
    per-trial GLM fit. Modifies trials[] in-place."""
    if not trials:
        return

    estimation = getattr(config, "GLM_ESTIMATION", "per_trial")

    # ══ PASS 1  Labels · per-trial std · session-level stats ══════════════════
    bl_stds = []
    for t in trials:
        if config.USE_SUBJECTIVE_TRIAL_TYPE:
            t["label"] = t["subjective_label"]
        else:
            img = str(t.get("image_type") or t.get("image_detail", "")).lower().strip()
            if "negative" in img or img == "1":
                t["label"] = 1
            elif "neutral" in img or img == "2":
                t["label"] = 2
            else:
                t["label"] = t["subjective_label"]

        if getattr(config, "GLM_CONDITION_FIELD", "label") == "has_sound":
            t["condition"] = 1 if t["has_sound"] else 2   # 1=Sound, 2=No-sound
        else:
            t["condition"] = t["label"]                     # 1=Negative, 2=Neutral

        i0, i1 = t["win_start_idx"], t["win_end_idx"]
        tw = t["times_wide"]
        window_raw = raw_z[i0:i1]
        pre_mask, post_mask = tw < 0, tw >= 0

        b_std = float(np.std(window_raw[pre_mask]))  if np.any(pre_mask)  else np.nan
        r_std = float(np.std(window_raw[post_mask])) if np.any(post_mask) else np.nan
        t["baseline_std"] = b_std
        t["response_std"] = r_std
        # Mean (not std) of the baseline window -- for baseline-corrected
        # timecourse plots only (t["epoch_anal"] - t["baseline_mean"]), same
        # convention as the BxB path. Not used by any rejection gate or score.
        t["baseline_mean"] = (float(np.mean(window_raw[pre_mask]))
                               if np.any(pre_mask) else np.nan)
        bl_stds.append(b_std)
        # t["rate_max_in_window"] was already computed in run_glm_scoring from
        # raw cycle periods (not the Phase-3 filtered series -- see there).

    sess_std_median = float(np.nanmedian(bl_stds)) if bl_stds else np.nan
    if getattr(config, "QC_ROBUST_GATES", False):
        _v = np.asarray(bl_stds, dtype=float)
        _v = _v[np.isfinite(_v)]
        sess_std_std = (airflow_qc.robust_sd(_v) if _v.size else np.nan)
    else:
        sess_std_std = float(np.nanstd(bl_stds)) if bl_stds else np.nan

    # ══ SHAPE-CENTROID OUTLIER DETECTION (reused verbatim on raw_z windows) ═══
    shape_enable    = getattr(config, "AIRFLOW_SHAPE_REJECTION_ENABLE", True)
    shape_threshold = getattr(config, "AIRFLOW_SHAPE_SD_THRESHOLD", 3.0)
    shape_win_min   = getattr(config, "AIRFLOW_SHAPE_WINDOW_MIN", -5.0)
    shape_win_max   = getattr(config, "AIRFLOW_SHAPE_WINDOW_MAX", 10.0)

    shape_msd    = [np.nan] * len(trials)
    shape_cutoff = np.nan

    if shape_enable and len(trials) > 1:
        expected_n_cols = int(round((shape_win_max - shape_win_min) * sfreq))
        shape_rows, shape_row_idxs = [], []
        for i, t in enumerate(trials):
            i0, i1 = t["win_start_idx"], t["win_end_idx"]
            mask = (t["times_wide"] >= shape_win_min) & (t["times_wide"] < shape_win_max)
            if expected_n_cols > 0 and int(np.sum(mask)) == expected_n_cols:
                shape_rows.append(raw_z[i0:i1][mask].astype(np.float64))
                shape_row_idxs.append(i)

        if len(shape_rows) > 1:
            M = np.vstack(shape_rows)
            row_means = M.mean(axis=1, keepdims=True)
            row_stds  = M.std(axis=1, keepdims=True)
            row_stds[row_stds == 0] = 1.0
            Mz = (M - row_means) / row_stds
            centroid = Mz.mean(axis=0)
            msd = np.mean((Mz - centroid) ** 2, axis=1)
            if getattr(config, "QC_ROBUST_GATES", False):
                shape_cutoff = airflow_qc.robust_cutoff(msd, shape_threshold)
            else:
                shape_cutoff = float(np.nanmean(msd)) + shape_threshold * float(np.nanstd(msd))
            for row_i, trial_i in enumerate(shape_row_idxs):
                shape_msd[trial_i] = float(msd[row_i])

    # ══ PASS 2  Rejection gates · GLM scoring ══════════════════════════════════
    for t, msd in zip(trials, shape_msd):
        i0, i1 = t["win_start_idx"], t["win_end_idx"]
        tw = t["times_wide"]
        bs, rs = t["baseline_std"], t["response_std"]

        # Reject if EITHER detector finds too few cycles here: the custom
        # zero-crossing detector found fewer than GLM_MIN_CYCLES_IN_WINDOW (a
        # 13-20 s window with <2 onsets can't yield even one respiration period
        # -- these are the ncyc<=1 trials the manual flags mark bad, previously
        # caught only incidentally by the cycle_gap edge artifact), OR NK2's
        # validated khodadad2018 detector disagrees that a real breath exists
        # (catches flat traces / step artifacts the custom detector's lack of
        # an amplitude floor would otherwise accept -- see config).
        min_cycles = getattr(config, "GLM_MIN_CYCLES_IN_WINDOW", 2)
        no_cycles_reject = t["n_cycles_in_window"] < min_cycles or not t["nk2_has_cycles"]

        z_threshold = getattr(config, "GLM_Z_SCORE_THRESHOLD", None)
        z_reject = (bool((bs - sess_std_median) / sess_std_std > z_threshold)
                    if z_threshold is not None and not np.isnan(bs) and sess_std_std > 0
                    else False)

        min_ratio = getattr(config, "GLM_MIN_STD_RATIO", None)
        flat_reject = (bool(bs < min_ratio * sess_std_median)
                       if min_ratio is not None and not np.isnan(bs)
                          and not np.isnan(sess_std_median)
                       else False)

        post_ratio = getattr(config, "GLM_POST_MIN_STD_RATIO", None)
        post_flat_reject = (bool(rs < post_ratio * sess_std_median)
                            if post_ratio is not None and not np.isnan(rs)
                               and not np.isnan(sess_std_median)
                            else False)

        rate_thresh = getattr(config, "GLM_RATE_ARTIFACT_THRESHOLD", None)
        rate_max = t["rate_max_in_window"]
        rate_reject = (bool(rate_max > rate_thresh)
                       if rate_thresh is not None and rate_max is not None
                          and np.isfinite(rate_max)
                       else False)

        shape_reject = (shape_enable and not np.isnan(msd) and not np.isnan(shape_cutoff)
                         and bool(msd > shape_cutoff))

        # cycle_gap: absolute minimum-rate floor (see config) -- a real cycle
        # at each edge of the window doesn't rule out a long dead stretch
        # between them, which neither no_cycles_found nor the std-based gates
        # catch (flat_signal/flat_response look at std over the WHOLE span).
        min_rate = getattr(config, "GLM_MIN_RATE_THRESHOLD", None)
        max_gap = t["max_gap_sec"]
        gap_reject = (bool(60.0 / max_gap < min_rate)
                      if min_rate is not None and max_gap is not None
                         and np.isfinite(max_gap) and max_gap > 0
                      else False)

        # amplitude_artifact: only ever True under the hampel_reject_trials
        # method (a Hampel-flagged amplitude spike in this trial's window); the
        # flag is precomputed in run_glm_scoring (see config.GLM_ARTIFACT_METHOD).
        amp_reject = bool(t.get("amp_artifact", False))

        min_knots = getattr(config, "QC_MIN_KNOTS", None)
        knot_reject = (bool(t["n_cycles_in_window"] < min_knots)
                       if min_knots is not None else False)

        min_valid = getattr(config, "QC_MIN_VALID_FRACTION", None)
        cov_reject = False
        if min_valid is not None:
            _y = series[config.GLM_PRIMARY_METRIC][i0:i1] \
                 if config.GLM_PRIMARY_METRIC in series else None
            if _y is not None and len(_y):
                cov_reject = bool(np.mean(np.isfinite(_y)) < min_valid)

        reasons = []
        if no_cycles_reject: reasons.append("no_cycles_found")
        if z_reject:         reasons.append("noisy_baseline")
        if flat_reject:      reasons.append("flat_signal")
        if rate_reject:      reasons.append("rate_artifact")
        if post_flat_reject: reasons.append("flat_response")
        if shape_reject:     reasons.append("atypical_shape")
        if gap_reject:       reasons.append("cycle_gap")
        if amp_reject:       reasons.append("amplitude_artifact")
        if knot_reject:      reasons.append("low_information")
        if cov_reject:       reasons.append("low_coverage")

        t["rejected"]         = bool(reasons)
        t["rejection_reason"] = ", ".join(reasons) if reasons else ""
        t["shape_msd"]        = msd

        t["epoch_anal"] = raw_z[i0:i1]
        t["times_anal"] = tw

        if t["rejected"]:
            t["score_rp"] = t["score_ra"] = t["score_rfr"] = np.nan
        elif estimation == "per_trial":
            for metric in ("RP", "RA", "RFR"):
                y_win = series[metric][i0:i1]
                beta, fitted = fit_trial_glm(y_win, tw, metric,
                                              config.GLM_RF_PARAMS,
                                              config.GLM_USE_DERIVATIVE)
                t[f"score_{metric.lower()}"] = beta
                if metric == "RA":
                    t["fitted_ra"] = fitted
                    t["series_ra"] = y_win   # for diagnostic plotting (own scale, not raw_z's)
        # estimation == "pooled_session": scores are filled AFTER this loop by a
        # single session-wide solve (fit_pooled_session_glm), since a pooled beta
        # depends on ALL accepted trials at once, not just this one.

    if estimation == "pooled_session":
        fit_pooled_session_glm(trials, series, sfreq, config)

    # ── score_ceiling gate (applies to BOTH estimation paths) ─────────────────
    # GLM_SCORE_MAX is None by default (disabled); when set, reject a trial whose
    # |score_ra| exceeds it, regardless of how the beta was estimated.
    score_max = getattr(config, "GLM_SCORE_MAX", None)
    if score_max is not None:
        for t in trials:
            if (not t["rejected"] and t["score_ra"] is not None
                    and np.isfinite(t["score_ra"]) and abs(t["score_ra"]) > score_max):
                t["rejected"]         = True
                t["score_rp"] = t["score_ra"] = t["score_rfr"] = np.nan
                t["rejection_reason"] = "score_ceiling"
