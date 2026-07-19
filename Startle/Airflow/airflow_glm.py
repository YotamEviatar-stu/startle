"""
airflow_glm.py — GLM/Deconvolution Scoring for Airflow Pipeline

Implements "A Linear Model for Event-Related Respiration Responses" (Bach 2016)

PIPELINE OVERVIEW (see functions for details):
  Phase 1: Mean-center + respiration-band filter (0.01-0.6 Hz, PsPM's own
           cutoffs) + downsample + z-score → raw_z, detection-only. The SAME
           filter that isolates the respiration band also serves as the
           anti-alias filter for the downsample (pspm_resp_pp.m Stage 1) --
           no separate anti-alias-only stage is needed for a narrowband signal.
  Phase 2: Detect cycles on raw_z; measure RP/RA/RFR on signal_raw → cycles[] list
           (mirrors pspm_resp_pp.m's newresp-for-detection / resp-for-amplitude split)
  Phase 3: Interpolate cycles → continuous 10Hz series
  Phase 4: Fit GLM per trial → score_rp, score_ra, score_rfr

Config: airflow_config.py (SCORING_METHOD = "glm_deconvolution")
Validation: NK2 cross-check (detects artifacts detect_cycles would accept)
"""

import numpy as np
import neurokit2 as nk
from scipy.signal import butter, filtfilt, lfilter, medfilt, resample


# ── Phase 1 ───────────────────────────────────────────────────────────────────

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


# ── Phase 2 ───────────────────────────────────────────────────────────────────

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

    A cycle = time interval from one inspiration onset to the next.
    E.g., if onsets are detected at t=[1s, 5s, 9s], cycles are [1→5s], [5→9s].
    Returns list of dicts with onset_time, assign_time, RP, RA, RFR."""
    # STAGE 1/2 (mean-center + 0.01-0.6 Hz cascaded filters) already applied
    # in phase1_filter_downsample -- raw_z arrives pre-filtered, matching
    # pspm_resp_pp.m Stage 1 exactly (see that function's docstring). Only
    # median-filtering + zero-crossing detection remain here.

    # STAGE 3: Smooth with median filter, then detect zero-crossings
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

    # STAGE 4: Enforce refractory period (minimum gap between onsets)
    """
    Default 1.0 sec = max 60 bpm. Keeps a candidate only if it's >= 1s
    after the last KEPT onset (not the last candidate).
    E.g. for onsets [0.0, 0.9, 1.7]: 0.9 is dropped (< 1s after 0.0), then
    1.7 is kept (1.7s after 0.0) -> result [0.0, 1.7].
    Deliberate deviation from pspm_resp_pp.m (lines 113-116), which checks
    gaps between raw consecutive crossings instead of the last kept one,
    and would have dropped 1.7 too just for sitting close to the
    already-noisy 0.9.
    """
    refractory_samples = refractory_sec * sfreq
    onsets = []
    last_onset = -np.inf
    for idx in cross_idx:
        if idx - last_onset >= refractory_samples:
            onsets.append(idx)
            last_onset = idx

    # STAGE 5: Extract features per cycle (each cycle = one breathing cycle).
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


# ── Amplitude-spike gate (Hampel identifier) ──────────────────────────────────

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


# ── Dual-Detector Logic ───────────────────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 → PHASE 3: Discrete Cycles to Continuous Series
# ═══════════════════════════════════════════════════════════════════════════════
#
# Phase 2 output: cycles[] list with discrete RP/RA/RFR values + timestamps
#   Example: cycles = [
#       {"RP": 4.0, "RA": 1.5, "RFR": 0.375, "assign_time": 5.0},
#       {"RP": 4.2, "RA": 1.6, "RFR": 0.38,  "assign_time": 9.2},
#       {"RP": 3.9, "RA": 1.4, "RFR": 0.36,  "assign_time": 13.1},
#   ]
#   (One point per breathing cycle, assigned to the FOLLOWING inspiration onset)
#
# Phase 3 converts this to continuous 10Hz series:
#   1. Take (assign_time, RP/RA/RFR) pairs
#   2. Interpolate linearly between them across the full session
#   3. Filter with unidirectional bandpass to smooth
#   Result: RP_series, RA_series, RFR_series — each 10 Hz, full session length
#
# Phase 4 uses these continuous series to fit per-trial GLM models.


# ── Phase 3 ───────────────────────────────────────────────────────────────────

def build_continuous_series(cycles, n_samples, sfreq, hp=0.001, lp=1.0):
    """PHASE 3: Discrete Cycles → Continuous Feature Time Series

    INPUT:  cycles[] list (from Phase 2, detect_cycles)
      Example:
        cycles = [
          {"RP": 4.0, "RA": 1.5, "RFR": 0.375, "assign_time": 5.0},
          {"RP": 4.2, "RA": 1.6, "RFR": 0.38,  "assign_time": 9.2},
          {"RP": 3.9, "RA": 1.4, "RFR": 0.36,  "assign_time": 13.1},
        ]

    OUTPUT: series (dict), times_full (array)
      series = {
          "RP": [4.0, 4.05, 4.1, 4.15, ...],  # 10Hz RP time series (full session)
          "RA": [1.5, 1.55, 1.6, 1.65, ...],  # 10Hz RA time series
          "RFR": [0.375, 0.378, 0.38, ...],   # 10Hz RFR time series
      }
      times_full = [0.0, 0.1, 0.2, 0.3, ...]  # Time labels (10Hz = 0.1s intervals)

    PROCESS:
      1. Extract (assign_time, RP/RA/RFR) pairs from cycles
      2. Linearly interpolate to 10Hz for full session length
         (fills gaps between discrete cycle measurements)
      3. Filter with unidirectional bandpass (per-metric high-pass, 1 Hz low-pass)
         RP high-pass 0.01 Hz, RA/RFR 0.001 Hz (PsPM parity, see hp arg below)
         Removes DC (mean breathing rate) + jitter; preserves causality
      4. Returns three smooth time series + time array

    WHY: Phase 4 GLM fitting needs continuous time series to match trial windows
         against canonical response functions. Not reconstructing waveform shape,
         but reconstructing how breathing *features* (RP/RA/RFR) evolve in time.
    """
    # Build time array for full session at 10 Hz
    times_full = np.arange(n_samples) / sfreq
    series = {}

    # Unidirectional bandpass "sensitivity filter" (from the paper).
    # High-pass removes DC (mean rate); low-pass (1 Hz) removes jitter.
    # `hp` may be a scalar (same cutoff for all three metrics) OR a per-metric
    # dict {"RP":.., "RA":.., "RFR":..}: PsPM uses 0.01 Hz for RP and 0.001 Hz
    # for RA/RFR (see GLM_FINAL_HP in airflow_config.py). The low-pass is shared.
    # lfilter (not filtfilt) preserves causality: no future-looking.
    nyq = sfreq / 2.0

    for metric in ("RP", "RA", "RFR"):
        # Handle edge case: fewer than 2 cycles
        if len(cycles) < 2:
            series[metric] = np.full(n_samples, np.nan)
            continue

        hp_m = hp[metric] if isinstance(hp, dict) else hp
        b, a = butter(1, [hp_m / nyq, lp / nyq], btype="band")

        # Extract (time, value) pairs for this metric
        # assign_time = where each cycle's value is assigned (following onset)
        knot_t = np.array([c["assign_time"] for c in cycles])
        knot_v = np.array([c[metric] for c in cycles])

        # Sort by time (ensure monotonic for interpolation)
        order = np.argsort(knot_t)
        knot_t, knot_v = knot_t[order], knot_v[order]

        # Linearly interpolate discrete cycle values to 10 Hz continuous series
        # (extend to edges with first/last value)
        interp = np.interp(times_full, knot_t, knot_v, left=knot_v[0], right=knot_v[-1])

        # Apply unidirectional sensitivity filter
        # (removes DC and high-frequency jitter while preserving onset timing)
        series[metric] = lfilter(b, a, interp)

    return series, times_full


# ── Phase 4 ───────────────────────────────────────────────────────────────────
# Per-Trial GLM Fitting: Solve Linear System for Each Trial
#
# SCOPE: Phase 4 runs ONCE PER TRIAL (not session-wide)
#
# For each trial:
#   INPUT:  y_window = trial's RP/RA/RFR time series (e.g., 150 samples)
#           tw = time array (relative to trial onset)
#           metric = "RP", "RA", or "RFR"
#
#   BUILD design matrix X:
#     X = [intercept | CRF_template | CRF_derivative]
#       column 0: all 1s (intercept β₀)
#       column 1: template curve (CRF)
#       column 2: template derivative (optional, for RA/RFR)
#     Shape: (150 × 3)
#
#   SOLVE linear system: β = pinv(X) @ y_window
#     Find β that minimizes ||y_window - X @ β||²
#     Result: β = [β₀, β₁, β₂]
#
#   EXTRACT SCORE: β₁ (the template coefficient)
#     β₁ = 2.0  → trial response 2× template strength
#     β₁ = 0.5  → trial response 0.5× template strength
#     β₁ ≈ 0    → no response (trial doesn't match template)
#     β₁ < 0    → opposite direction
#
#   OUTPUT: score_rp, score_ra, score_rfr (one β₁ per metric)


def generate_canonical_rf(metric, rf_params, tw):
    """Generate the expected response template for this metric.

    Canonical Response Function (CRF): Gaussian curve
      h(t) = exp(-(t-τ)² / (2σ²))

    Interpretation:
      τ = peak time (seconds after trigger when response is strongest)
      σ = spread (how wide the response window is)

    IMPORTANT: `tw` is the trial's FULL time window -- every sample from
    baseline start through response end (see fit_trial_glm below), not just
    a single instant. This function evaluates the Gaussian at EVERY one of
    those time points, so the template is a whole curve (rises toward τ,
    peaks at τ, falls away afterward) that the fit will compare against the
    whole observed curve -- τ is just where this particular curve happens
    to peak, not the only point that matters.

    INPUT:  metric ("RP", "RA", "RFR"), rf_params (config dict), tw (time array)
    OUTPUT: CRF curve evaluated at each time point in tw
    """
    tau, sigma = rf_params[metric]
    return np.exp(-((tw - tau) ** 2) / (2 * sigma ** 2))


def orthogonalize_and_normalize_basis(bs):
    """Mirrors PsPM's spm_orth.m + the normalization line every pspm_bf_r*rf_e.m
    basis function applies before returning: `bs = spm_orth(bs); bs = bs ./
    (max(bs) - min(bs))`. Applied to [CRF, dCRF/dt] ONLY -- the intercept
    column is added separately by the caller and is never orthogonalized
    against these, matching PsPM (spm_orth runs inside the basis-function
    file, before pspm_glm ever adds an intercept to the design matrix).

    Serial Gram-Schmidt (spm_orth.m lines 36-46): column 1 is kept as-is;
    each later column has its projection onto all earlier (already-kept)
    columns subtracted out, via `D - x @ (pinv(x) @ D)` -- i.e. the
    least-squares projection of D onto the span of x, removed. A column
    that becomes ~zero after that (collinear with earlier ones) is zeroed
    out (`norm(D,1) > exp(-32)` in the original; mirrored here) rather than
    kept, matching spm_orth's 'pad' default.

    Then every column (including the untouched first one) is rescaled to
    unit range (max - min), matching the normalization line that runs
    after spm_orth in every pspm_bf_r*rf_e.m file.
    """
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
    """Solve: y_actual = β₀ + β₁·CRF(t) + β₂·dCRF/dt

    For this trial, find β₀, β₁, β₂ that make the equation fit the data.

    *** SCORING METHOD -- READ THIS BEFORE TRUSTING score_rp/score_ra/score_rfr ***

    This is a regression over the trial's ENTIRE [baseline_start, response_end)
    window (see run_glm_scoring: i0 = baseline_range_sec[0], i1 =
    response_range_sec[1] -- the full epoch, not a peak-only slice), not a
    read-off of one peak sample. Every point in y_window contributes to the
    least-squares fit (`pinv(X) @ y_window` uses ALL rows of X and y_window
    together, in one solve). The CRF template (see generate_canonical_rf) also
    spans that whole window -- it rises toward its peak, peaks at τ seconds
    post-onset, then falls away -- so a high score requires the OBSERVED
    curve's whole shape (not just its highest point) to track that same
    rise-peak-fall pattern across the full window. A trial that hits a big
    peak at the wrong time, or has a big peak surrounded by a shape that
    doesn't otherwise resemble the template, will NOT automatically score
    high just because the peak sample is large.

    β₀ (intercept) absorbs the pre-stimulus baseline level, so the score
    (β₁) reflects the template-matching strength AFTER accounting for
    wherever the trial's own baseline sits -- not the raw response magnitude.

    INPUT:   y_window = observed RP/RA/RFR time series, over the FULL trial
             window (baseline start -> response end), not just the response
             half and not just a peak
             tw = time array, same full-window span as y_window
             metric, rf_params, use_derivative = CRF config

    SOLVE via regression:
      β = pinv(X) @ y_window    (where X = [intercept | CRF | derivative])
      Result: β = [β₀, β₁, β₂]

    PREDICT: Once we have β, calculate predictions at each time point
      fitted = X @ β = β₀ + β₁·CRF(t) + β₂·dCRF(t)

    EXAMPLE:
      Actual:      [1.0, 1.2, 1.5, 1.8, 1.9, 1.7, 1.4, 1.1, 0.9, 0.8]
      Solved β:    β₀=0.5, β₁=1.8, β₂=0.3
      Predicted:   [0.52, 0.78, 1.15, 1.65, 1.95, 2.00, 1.75, 1.25, 0.78, 0.55]
                   (β₁=1.8 means template needed 1.8× strength, fit across
                   ALL 10 points shown -- not derived from any single one)

    OUTPUT:
      beta_rf = 1.8              (THE SCORE: template-matching strength over
                                   the whole window, not a peak value)
      fitted_curve = [0.52, ...] (predictions: for diagnostic plots)
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

    # SOLVE: Find β that minimizes error
    beta = np.linalg.pinv(X[valid]) @ y_window[valid]
    # Result: beta = [β₀, β₁, β₂] — the coefficients

    # PREDICT: What does the model predict at each time point?
    # fitted = X @ beta = [1, CRF, dCRF] @ [β₀, β₁, β₂]
    #        = β₀ + β₁·CRF(t) + β₂·dCRF(t) at each time point
    # Shape: same as y_window (array of predicted values)
    fitted = X @ beta

    # Return β₁ (the score) and predictions (for plotting)
    return float(beta[1]), fitted


# ── Pooled session-wide GLM (PsPM pspm_glm.m architecture) ────────────────────

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

        # One event train per condition (valence label); one design column per
        # (condition, basis-column). col_is_crf[j] marks the score-bearing column.
        onsets_by_cond = {}
        for t in accepted:
            onsets_by_cond.setdefault(t["label"], []).append(t["onset_time"])

        reg_cols, col_cond, col_is_crf = [], [], []
        for cond in sorted(onsets_by_cond):
            delta = np.zeros(n_samples)
            for ot in onsets_by_cond[cond]:
                idx = int(round(ot * sfreq))
                if 0 <= idx < n_samples:
                    delta[idx] = 1.0
            for bi in range(basis.shape[1]):
                # np.convolve places the kernel's first sample (t=kernel_start) at
                # each impulse; slice from onset_off so output[p] holds the kernel
                # at relative time (p - onset)/sfreq, i.e. onset aligned to t=0.
                conv = np.convolve(delta, basis[:, bi])[onset_off:onset_off + n_samples]
                reg_cols.append(conv)
                col_cond.append(cond)
                col_is_crf.append(bi == 0)

        beta_by_cond = {c: np.nan for c in onsets_by_cond}
        full_fitted = None
        if reg_cols:
            Xr = np.column_stack(reg_cols)
            valid = np.isfinite(y) & np.all(np.isfinite(Xr), axis=1)
            if valid.sum() >= Xr.shape[1] + 2:
                # PsPM model.centering=1: mean-centre convolved columns, then
                # prepend the intercept (which absorbs the mean level, §3).
                Xc = Xr - Xr[valid].mean(axis=0, keepdims=True)
                X = np.column_stack([np.ones(n_samples), Xc])
                beta = np.linalg.pinv(X[valid]) @ y[valid]
                for j in range(len(col_cond)):
                    if col_is_crf[j]:
                        beta_by_cond[col_cond[j]] = float(beta[1 + j])
                full_fitted = X @ beta

        for t in accepted:
            t[f"score_{metric.lower()}"] = beta_by_cond.get(t["label"], np.nan)
            if metric == "RA":
                i0, i1 = t["win_start_idx"], t["win_end_idx"]
                t["series_ra"] = y[i0:i1]
                t["fitted_ra"] = full_fitted[i0:i1] if full_fitted is not None else None


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_glm_scoring(sessions_cache, config):
    """Run all 4 phases of GLM pipeline on cached data.

    Phases: 1) Filter → 2) Detect cycles → 3) Interpolate → 4) Fit GLM per trial

    INPUT:  sessions_cache (Layer 1 cache: 25Hz signal + trial metadata)
    OUTPUT: trials_data[subj][sess_key] = list of trial dicts with:
              • score_rp, score_ra, score_rfr (β₁ values)
              • rejected, rejection_reason
              • epoch_anal, times_anal (for plots)
    """
    trials_data = {}
    traces = {}
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

        for sess_key, sess in sess_dict.items():
            if f"{subj}/{sess_key}" in subjects_exclude:
                continue
            signal_raw   = sess["signal_raw"]
            native_sfreq = float(sess["sfreq"])
            trials_meta  = sess["trials_meta"]

            raw_z, sfreq = phase1_filter_downsample(
                signal_raw, native_sfreq,
                target_sfreq=config.GLM_TARGET_SFREQ,
                bp_low=config.GLM_CYCLE_BANDPASS[0],
                bp_high=config.GLM_CYCLE_BANDPASS[1],
                zscore=config.GLM_ZSCORE_RAW,
                despike=getattr(config, "GLM_DESPIKE_ENABLED", True),
                despike_k=getattr(config, "GLM_DESPIKE_K", 50.0),
                despike_max_run_sec=getattr(config, "GLM_DESPIKE_MAX_RUN_SEC", 1.0),
            )
            n_samples = len(raw_z)

            cycles = detect_cycles(
                raw_z, sfreq, signal_raw, native_sfreq,
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
            if artifact_method in ("hampel_drop_cycles", "hampel_reject_trials"):
                flags = _hampel_flag_cycles(cycles, artifact_k)
                if artifact_method == "hampel_drop_cycles":
                    cycles = [c for c, bad in zip(cycles, flags) if not bad]
                else:  # hampel_reject_trials
                    flagged_onset_times = [c["onset_time"]
                                           for c, bad in zip(cycles, flags) if bad]

            series, times_full = build_continuous_series(
                cycles, n_samples, sfreq,
                hp=config.GLM_FINAL_HP, lp=config.GLM_FINAL_LP,
            )

            # NK2 cross-check for no_cycles_found -- same detector/config the
            # BxB path already applies identically to every subject (see
            # config for why this replaced session-median ratio thresholds).
            nk2_peak_times, nk2_trough_times = _nk2_peak_trough_times(raw_z, sfreq, config)

            d110_secs = np.array([m["d110_time"] for m in trials_meta
                                   if m["d110_time"] is not None])
            traces[subj][sess_key] = {
                "signal": raw_z, "sfreq": sfreq, "startle_samps_sec": d110_secs,
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

                # cycle_gap input: longest dead stretch anywhere in the window,
                # including window-start -> first onset and last onset ->
                # window-end (a real cycle at each edge doesn't rule out a long
                # dead middle -- see config). Absolute floor, not session-relative.
                onset_edges = np.array(
                    [b0] + [c["onset_time"] for c in cycles_in_window] + [r1])
                max_gap_sec = float(np.max(np.diff(onset_edges)))

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
    """Run Phase 4: per-trial GLM fitting (two passes).

    INPUT:  trials (list of trial dicts, pre-populated with metadata + windows)
            raw_z (z-scored signal)
            series (dict: RP_series, RA_series, RFR_series — Phase 3 output)
            sfreq, config

    PASS 1: Session-level preprocessing
      • Resolve trial labels (subjective vs image metadata)
      • Compute session-level baseline stats (median std, std of stds)
      • Build shape-centroid matrix (for atypical_shape rejection gate)

    PASS 2: Per-trial GLM fitting
      • For EACH trial:
        - Build design matrix X = [intercept | CRF | dCRF/dt]
        - Solve: β = pinv(X) @ y_window (for each metric RP/RA/RFR)
        - Extract β₁ = score_rp, score_ra, score_rfr
        - Apply rejection gates
        - Store results in trial dict

    OUTPUT: Modifies trials[] in-place (adds scores, rejection flags, diagnostics)
    """
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
    sess_std_std    = float(np.nanstd(bl_stds))    if bl_stds else np.nan

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
            msd_mean, msd_std = float(np.nanmean(msd)), float(np.nanstd(msd))
            shape_cutoff = msd_mean + shape_threshold * msd_std
            for row_i, trial_i in enumerate(shape_row_idxs):
                shape_msd[trial_i] = float(msd[row_i])

    # ══ PASS 2  Rejection gates · GLM scoring ══════════════════════════════════
    for t, msd in zip(trials, shape_msd):
        i0, i1 = t["win_start_idx"], t["win_end_idx"]
        tw = t["times_wide"]
        bs, rs = t["baseline_std"], t["response_std"]

        # Reject if EITHER detector finds no real cycle here: the custom
        # zero-crossing detector found literally none, OR NK2's validated
        # khodadad2018 detector disagrees that a real breath exists in this
        # window (catches flat traces / step artifacts the custom detector's
        # lack of an amplitude floor would otherwise accept -- see config).
        no_cycles_reject = t["n_cycles_in_window"] == 0 or not t["nk2_has_cycles"]

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

        reasons = []
        if no_cycles_reject: reasons.append("no_cycles_found")
        if z_reject:         reasons.append("noisy_baseline")
        if flat_reject:      reasons.append("flat_signal")
        if rate_reject:      reasons.append("rate_artifact")
        if post_flat_reject: reasons.append("flat_response")
        if shape_reject:     reasons.append("atypical_shape")
        if gap_reject:       reasons.append("cycle_gap")
        if amp_reject:       reasons.append("amplitude_artifact")

        t["rejected"]         = bool(reasons)
        t["rejection_reason"] = ", ".join(reasons) if reasons else ""
        t["shape_msd"]        = msd

        t["epoch_anal"] = raw_z[i0:i1]
        t["times_anal"] = tw

        if t["rejected"]:
            t["score_rp"] = t["score_ra"] = t["score_rfr"] = np.nan
        elif estimation == "per_trial":
            # ── Per-trial GLM fitting: Solve linear system for each metric ──
            # For this trial, we have:
            #   y_win = observed RP/RA/RFR time series (extracted from continuous series)
            #   tw = time array for this trial window
            # For each metric, solve: y_win = β₀ + β₁·CRF(t) + β₂·dCRF/dt
            # Extract β₁ as the SCORE for this trial/metric
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

    # ── PsPM pooled session-wide GLM (config.GLM_ESTIMATION="pooled_session") ──
    # One design matrix for the whole session, one solve per metric -> one beta
    # per condition, written back onto each accepted trial (see the function).
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
