# airflow_processor.py — stateless processing for the Airflow pipeline.
#
# ── Cache architecture ─────────────────────────────────────────────────────────
#
# Layer 1  process_session()       MFF → raw signal (downsampled) + trial metadata
#                                  Written to pickle. No NK2, no filtering.
#
# Layer 2  apply_analysis_params() Everything else:
#            • NK2 khodadad2018 (or any configured method) on the full session signal
#            • Epoch cutting at current WIDE_TMIN / WIDE_TMAX
#            • Baseline, rejection, scoring
#          Returns (trials_data, traces). Never touches the MFF.
#
# What requires FORCE_RELOAD (Layer 1 changes):
#   AIRFLOW_CHANNEL, CACHE_SFREQ, trigger codes
#
# What is free to change in the notebook (Layer 2):
#   RSP_CLEAN_METHOD, WIDE_TMIN/TMAX, BASELINE_TMIN/TMAX,
#   RESPONSE_TMIN/TMAX, ANAL_TMIN/TMAX, all rejection gates,
#   AIRFLOW_SCORE_MAX, USE_SUBJECTIVE_TRIAL_TYPE
#
# ── Signal notes ───────────────────────────────────────────────────────────────
#
# NK2 method khodadad2018: Butterworth bandpass 0.05–3 Hz, 2nd order.
# Respiratory content: 0.1–0.5 Hz at rest (12–30 bpm). Running NK2 on the
# full session (not individual epochs) avoids Butterworth edge-effects — the
# filter has 20+ seconds of context before the first sample of interest.
#
# Cache sampling rate: 25 Hz (default). Nyquist = 12.5 Hz, well above the
# 3 Hz upper band. scipy.signal.resample_poly applies a Kaiser-window
# anti-aliasing filter before downsampling. NK2 runs correctly at 25 Hz.
#
# Scoring (peak_excursion_normalized):
#   score = max |RSP_Clean − baseline_mean| in response window
#           ───────────────────────────────────────────────────
#           session_breath_amp
#
#   session_breath_amp = median of per-trial median(RSP_Amplitude in baseline window)
#   Denominator is always positive; score is in units of breath amplitudes.

import os
import re
import warnings

import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

import Airflow.airflow_config as cfg

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── File discovery ─────────────────────────────────────────────────────────────

def find_startle_output_folder(subject_path):
    for name in os.listdir(subject_path):
        if name.lower() == "startle output" and os.path.isdir(os.path.join(subject_path, name)):
            return os.path.join(subject_path, name)
    if any(f.lower().endswith(".csv") for f in os.listdir(subject_path)):
        return subject_path
    return None


def find_mff_file(eeg_folder, keyword):
    for root, dirs, files in os.walk(eeg_folder):
        for name in sorted(dirs + files):
            if f"_{keyword}_" in name.lower() and name.lower().endswith(".mff"):
                return os.path.join(root, name)
    return None


def find_csv_by_suffix(startle_folder, suffix):
    for root, _, files in os.walk(startle_folder):
        for name in sorted(files):
            if "demo" not in name.lower() and name.lower().endswith(".csv"):
                if re.search(rf"{re.escape(suffix)}\.csv$", name, flags=re.IGNORECASE):
                    return os.path.join(root, name)
    return None


# ── DIN event extraction ───────────────────────────────────────────────────────

def get_events_from_eeg(raw_eeg):
    din_names = [ch for ch in raw_eeg.ch_names if ch.startswith("D")]
    if not din_names:
        return pd.DataFrame(columns=["Channel", "Sample"])
    raw_din = raw_eeg.copy().pick(din_names)
    event_data = []
    for ch_idx, ch_name in enumerate(raw_din.ch_names):
        data      = raw_din.get_data(picks=[ch_idx])[0]
        threshold = np.max(data) * 0.9
        samps     = np.where(data > threshold)[0]
        if len(samps) == 0:
            continue
        gaps   = np.where(np.diff(samps) > 1)[0]
        onsets = np.concatenate([[samps[0]], samps[gaps + 1]])
        for sample in onsets:
            event_data.append({"Channel": ch_name, "Sample": int(sample)})
    if not event_data:
        return pd.DataFrame(columns=["Channel", "Sample"])
    return pd.DataFrame(event_data).sort_values("Sample").reset_index(drop=True)


# ── Ratings loading ────────────────────────────────────────────────────────────

def load_and_classify_ratings(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["has_sound"] == True].reset_index(drop=True)

    def categorize(row):
        if row["arousalRating"] >= 7 or row["valenceRating"] <= 3:
            return 1  # Negative
        return 2      # Neutral

    df["subjective_label"] = df.apply(categorize, axis=1)
    return df


# ── Layer 1: MFF extraction (cache boundary) ──────────────────────────────────

def process_session(mff_path, ratings_df, channel=None, config=None):
    """
    Load one MFF session, extract the raw Airflow channel, downsample it,
    and store trigger positions + CSV metadata.

    CACHE BOUNDARY: no NK2, no filtering, no epoch cutting. Output is written
    to pickle and represents pure Layer 1 data. All signal processing lives
    in apply_analysis_params() so it can be reconfigured without re-reading MFFs.

    Downsampling:
      The raw signal is anti-aliased (Kaiser window) and stored at CACHE_SFREQ Hz
      (default 25 Hz). Nyquist = 12.5 Hz — well above the 3 Hz respiratory content.
      Trigger positions are converted to the downsampled time base.

    Returns a session dict or None on failure:
      {
        'raw_signal'  : float32 ndarray  (full session, CACHE_SFREQ Hz)
        'sfreq'       : float            (CACHE_SFREQ, e.g. 25.0)
        'sfreq_orig'  : float            (original MFF sample rate, for reference)
        'channel'     : str
        'startle_samps': int64 ndarray   (session-relative, CACHE_SFREQ domain)
        'trials_meta' : list of dicts    (one per trial; no signal data)
      }
    """
    if config is None:
        config = cfg
    if channel is None:
        channel = config.AIRFLOW_CHANNEL

    cache_sfreq = getattr(config, "CACHE_SFREQ", 25.0)
    print(f"    Loading {os.path.basename(mff_path)} (channel={channel}) ...")

    try:
        raw = mne.io.read_raw_egi(mff_path, preload=True, verbose=False,
                                  events_as_annotations=False)
    except Exception as e:
        print(f"    [!] Cannot read MFF ({e}) — skipping")
        return None
    sfreq_orig = raw.info["sfreq"]

    events_df   = get_events_from_eeg(raw)
    start_samps = events_df[events_df["Channel"] == f"D{config.TRIGGER_SESSION_START}"]["Sample"].values
    end_samps   = events_df[events_df["Channel"] == f"D{config.TRIGGER_SESSION_END}"]["Sample"].values
    if not len(start_samps) or not len(end_samps):
        print("    [!] Task boundaries not found — skipping")
        return None

    raw.crop(tmin=start_samps[0] / sfreq_orig, tmax=end_samps[-1] / sfreq_orig)

    if channel not in raw.ch_names:
        print(f"    [!] Channel {channel!r} not found — skipping")
        return None

    raw_channel = raw.copy().pick([channel]).get_data()[0]  # full resolution

    # Trigger positions on the cropped recording
    events_df     = get_events_from_eeg(raw)
    startle_samps = events_df[events_df["Channel"] == f"D{config.TRIGGER_STARTLE}"]["Sample"].values
    print(f"    {len(startle_samps)} startle events found.")

    # Align to CSV
    n_use         = min(len(startle_samps), len(ratings_df))
    startle_samps = startle_samps[:n_use]
    ratings_df    = ratings_df.iloc[:n_use].reset_index(drop=True)

    # Anti-alias and downsample
    down = max(1, int(round(sfreq_orig / cache_sfreq)))
    if down > 1:
        raw_ds   = resample_poly(raw_channel.astype(np.float64), 1, down).astype(np.float32)
        sfreq_ds = sfreq_orig / down
        samps_ds = np.round(startle_samps / down).astype(np.int64)
    else:
        raw_ds   = raw_channel.astype(np.float32)
        sfreq_ds = sfreq_orig
        samps_ds = startle_samps.copy().astype(np.int64)

    # Per-trial metadata (CSV; no signal)
    # image_type must come first: the CSV also has image/image_name filename columns,
    # and storing a filename in image_detail would cause label resolution to silently
    # fall back to subjective_label even when USE_SUBJECTIVE_TRIAL_TYPE = False.
    trials_meta = []
    for _, row in ratings_df.iterrows():
        img_col = next((c for c in ["image_type", "image", "image_name"]
                        if c in row.index and not pd.isna(row.get(c))), None)
        img_type_val = (str(row["image_type"]).lower().strip()
                        if "image_type" in row.index and not pd.isna(row.get("image_type"))
                        else "")
        trials_meta.append({
            "subjective_label": int(row["subjective_label"]),
            "valence":          float(row["valenceRating"]),
            "arousal":          float(row["arousalRating"]),
            "image_detail":     str(row.get(img_col, "?")) if img_col else "?",
            "image_type":       img_type_val,
        })

    return {
        "raw_signal":   raw_ds,
        "sfreq":        float(sfreq_ds),
        "sfreq_orig":   float(sfreq_orig),
        "channel":      channel,
        "startle_samps": samps_ds,
        "trials_meta":  trials_meta,
    }


# ── Layer 2: analysis (fast, no MFF access) ───────────────────────────────────

def apply_analysis_params(sessions_cache, config):
    """
    Run NK2 + epoch cutting + baseline + rejection + scoring on every session.
    Takes the raw sessions cache; returns (trials_data, traces).

    trials_data : {subj: {sess_key: [list of trial dicts]}}
                  Same shape expected by all notebook cells and plot functions.

    traces      : {subj: {sess_key: {'signal', 'sfreq', 'startle_samps_sec'}}}
                  NK2-filtered session signal for the timecourse plot.

    Rejection gates:
      nan_baseline   : baseline window has no valid samples
      noisy_baseline : baseline std > AIRFLOW_Z_SCORE_THRESHOLD SDs above session median
      amplitude_spike: max|baseline window| > AIRFLOW_AMPLITUDE_Z_THRESHOLD robust-SDs (median/MAD)
      flat_signal    : baseline std < AIRFLOW_MIN_STD_RATIO × session median std
      rate_artifact  : NK2 RSP_Rate anywhere in epoch > RSP_RATE_ARTIFACT_THRESHOLD bpm
      score_ceiling  : normalised score > AIRFLOW_SCORE_MAX

    Scoring (peak_excursion_normalized):
      max |RSP_Clean − baseline_mean| in [RESPONSE_TMIN, RESPONSE_TMAX]
      ────────────────────────────────────────────────────────────────────
      session_breath_amp  (median of per-trial median RSP_Amplitude in baseline window)
    """
    method = getattr(config, "RSP_CLEAN_METHOD", "khodadad2018")

    trials_data = {}
    traces      = {}

    for subj, sess_dict in sorted(sessions_cache.items()):
        trials_data[subj] = {}
        traces[subj]      = {}

        for sess_key, sess in sess_dict.items():

            # ── Old format: sess is already a list of trial dicts ─────────────
            if isinstance(sess, list):
                trials = [dict(t) for t in sess]
                for t in trials:
                    t.setdefault("rsp_rate_wide", None)
                    t.setdefault("image_type", "")
                    # Inject breath-amplitude proxy from baseline window range
                    # when rsp_amplitude_wide is absent (old cache pre-dates NK2 amplitude)
                    if t.get("rsp_amplitude_wide") is None:
                        tw = t["times_wide"]
                        ep = t["epoch_clean_wide"]
                        pre = (tw >= config.BASELINE_TMIN) & (tw < config.BASELINE_TMAX)
                        amp = float(np.max(ep[pre]) - np.min(ep[pre])) if np.any(pre) else 0.0
                        t["rsp_amplitude_wide"] = np.full(len(ep), amp, dtype=np.float32)
                    # Reset analysis fields so current config is applied
                    t["baseline_mean"] = None
                    t["score"]         = None
                    t["rejected"]      = None
                    t["rejection_reason"] = ""
                    t["epoch_anal"]    = None
                    t["times_anal"]    = None
                _run_analysis(trials, config)
                trials_data[subj][sess_key] = trials
                traces[subj][sess_key]      = {}
                continue

            raw_sig       = sess["raw_signal"]
            sfreq         = sess["sfreq"]
            startle_samps = sess["startle_samps"]
            trials_meta   = sess["trials_meta"]

            # ── NK2 on full session signal ────────────────────────────────────
            try:
                rsp_signals, _ = nk.rsp_process(raw_sig, sampling_rate=int(round(sfreq)),
                                                  method=method)
            except Exception as exc:
                print(f"  [!] NK2 failed for {subj} {sess_key}: {exc} — skipping")
                trials_data[subj][sess_key] = []
                continue

            clean   = rsp_signals["RSP_Clean"].to_numpy().astype(np.float32)
            nan_col = np.full(len(clean), np.nan, dtype=np.float32)
            rsp_amp  = rsp_signals["RSP_Amplitude"].to_numpy().astype(np.float32) \
                       if "RSP_Amplitude" in rsp_signals.columns else nan_col.copy()
            rsp_rate = rsp_signals["RSP_Rate"].to_numpy().astype(np.float32) \
                       if "RSP_Rate" in rsp_signals.columns else nan_col.copy()
            zeros    = np.zeros(len(clean))
            peaks    = np.flatnonzero(rsp_signals.get("RSP_Peaks",   pd.Series(zeros)).to_numpy())
            troughs  = np.flatnonzero(rsp_signals.get("RSP_Troughs", pd.Series(zeros)).to_numpy())

            # Store filtered signal for timecourse visualisation
            traces[subj][sess_key] = {
                "signal":            clean,
                "sfreq":             sfreq,
                "startle_samps_sec": startle_samps / sfreq,
            }

            # ── Cut wide epochs ───────────────────────────────────────────────
            wide_n     = int((config.WIDE_TMAX - config.WIDE_TMIN) * sfreq)
            times_wide = np.linspace(config.WIDE_TMIN, config.WIDE_TMAX,
                                     wide_n, endpoint=False).astype(np.float32)

            trials = []
            for star_samp, meta in zip(startle_samps, trials_meta):
                t0 = int(star_samp + config.WIDE_TMIN * sfreq)
                t1 = t0 + wide_n
                if t0 < 0 or t1 > len(clean):
                    continue

                ep_peaks   = np.array([p - t0 for p in peaks   if t0 <= p < t1])
                ep_troughs = np.array([t - t0 for t in troughs  if t0 <= t < t1])

                trial = {
                    "epoch_raw_wide":     raw_sig[t0:t1].copy(),
                    "epoch_clean_wide":   clean[t0:t1].copy(),
                    "rsp_amplitude_wide": rsp_amp[t0:t1].copy(),
                    "rsp_rate_wide":      rsp_rate[t0:t1].copy(),
                    "ep_peaks":           ep_peaks,
                    "ep_troughs":         ep_troughs,
                    "times_wide":         times_wide,
                    "sfreq":              sfreq,
                    "channel":            sess["channel"],
                    # CSV metadata
                    "subjective_label":   meta["subjective_label"],
                    "label":              meta["subjective_label"],
                    "valence":            meta["valence"],
                    "arousal":            meta["arousal"],
                    "image_detail":       meta["image_detail"],
                    "image_type":         meta.get("image_type", ""),
                    # Filled below
                    "baseline_mean":      None,
                    "score":              None,
                    "rejected":           None,
                    "rejection_reason":   "",
                    "epoch_anal":         None,
                    "times_anal":         None,
                }
                trials.append(trial)

            # ── Label resolution and analysis ─────────────────────────────────
            _run_analysis(trials, config)
            trials_data[subj][sess_key] = trials

    return trials_data, traces


# ── Internal: label + baseline + rejection + scoring ──────────────────────────

def _run_analysis(trials, config):
    """Resolve labels, compute baseline stats, apply all rejection gates, score."""
    if not trials:
        return

    # Pass 1 — labels, baseline, session-level statistics
    bl_stds       = []
    pre_masks     = []
    epoch_maxabs  = []
    amp_baselines = []

    for t in trials:
        # Label from image_type column (or subjective if ambiguous)
        if config.USE_SUBJECTIVE_TRIAL_TYPE:
            t["label"] = t["subjective_label"]
        else:
            img = (str(t.get("image_type") or t.get("image_detail", ""))).lower().strip()
            if "negative" in img or img == "1":
                t["label"] = 1
            elif "neutral" in img or img == "2":
                t["label"] = 2
            else:
                t["label"] = t["subjective_label"]

        tw       = t["times_wide"]
        ep       = t["epoch_clean_wide"]
        pre_mask = (tw >= config.BASELINE_TMIN) & (tw < config.BASELINE_TMAX)
        pre_masks.append(pre_mask)

        if np.any(pre_mask):
            bm = float(np.mean(ep[pre_mask]))
            bs = float(np.std(ep[pre_mask]))
        else:
            bm = bs = np.nan
        t["baseline_mean"] = bm
        bl_stds.append(bs)
        # Amplitude spike check restricted to baseline window only.
        # A spike at t>0 (response or post-response) is irrelevant to score validity —
        # we were rejecting valid startle gasps and post-trial movements.
        bl_ep = ep[pre_mask] if np.any(pre_mask) else ep
        epoch_maxabs.append(float(np.nanmax(np.abs(bl_ep))))

        # RSP_Amplitude in baseline window → session breath amplitude
        ep_amp = t.get("rsp_amplitude_wide")
        if ep_amp is not None and np.any(pre_mask):
            vals  = ep_amp[pre_mask]
            valid = vals[np.isfinite(vals) & (vals > 0)]
            if len(valid) > 0:
                amp_baselines.append(float(np.median(valid)))

    sess_std_median    = float(np.nanmedian(bl_stds))
    sess_std_std       = float(np.nanstd(bl_stds))
    sess_amp_median    = float(np.nanmedian(epoch_maxabs))
    _diffs             = np.abs(np.array(epoch_maxabs) - sess_amp_median)
    sess_amp_mad       = float(np.nanmedian(_diffs))
    session_breath_amp = float(np.nanmedian(amp_baselines)) if amp_baselines else np.nan

    # Pass 2 — rejection, score, epoch trim
    for t, pre_mask, max_abs in zip(trials, pre_masks, epoch_maxabs):
        tw = t["times_wide"]
        ep = t["epoch_clean_wide"]
        bm = t["baseline_mean"]
        bs = float(np.std(ep[pre_mask])) if np.any(pre_mask) else np.nan

        # Gate 1: baseline noise z-score
        z_reject  = False
        threshold = getattr(config, "AIRFLOW_Z_SCORE_THRESHOLD", None)
        if threshold is not None and not np.isnan(bs) and sess_std_std > 0:
            z_reject = bool((bs - sess_std_median) / sess_std_std > threshold)

        # Gate 2: amplitude spike (robust z-score: median/MAD)
        amp_reject    = False
        amp_threshold = getattr(config, "AIRFLOW_AMPLITUDE_Z_THRESHOLD", None)
        if amp_threshold is not None and sess_amp_mad > 0 and not np.isnan(max_abs):
            z_amp      = (max_abs - sess_amp_median) / (1.4826 * sess_amp_mad)
            amp_reject = bool(z_amp > amp_threshold)

        # Gate 3: flat signal
        flat_reject   = False
        min_std_ratio = getattr(config, "AIRFLOW_MIN_STD_RATIO", None)
        if min_std_ratio is not None and not np.isnan(bs) and sess_std_median > 0:
            flat_reject = bool(bs < min_std_ratio * sess_std_median)

        # Gate 4: physiological RSP_Rate ceiling
        rate_reject    = False
        rate_threshold = getattr(config, "RSP_RATE_ARTIFACT_THRESHOLD", None)
        if rate_threshold is not None:
            ep_rate     = t.get("rsp_rate_wide")
            valid_rates = ep_rate[np.isfinite(ep_rate)] if ep_rate is not None else np.array([])
            if len(valid_rates) > 0:
                rate_reject = bool(float(np.max(valid_rates)) > rate_threshold)

        reasons = []
        if np.isnan(bm): reasons.append("nan_baseline")
        if z_reject:     reasons.append("noisy_baseline")
        if amp_reject:   reasons.append("amplitude_spike")
        if flat_reject:  reasons.append("flat_signal")
        if rate_reject:  reasons.append("rate_artifact")

        t["rejected"]         = bool(reasons)
        t["rejection_reason"] = ", ".join(reasons) if reasons else ""

        anal_mask       = (tw >= config.ANAL_TMIN) & (tw < config.ANAL_TMAX)
        t["epoch_anal"] = ep[anal_mask]
        t["times_anal"] = tw[anal_mask]

        # Score: max absolute excursion / session breath amplitude
        if t["rejected"] or not config.PERFORM_SCORING:
            t["score"] = np.nan
        else:
            resp_mask = (tw >= config.RESPONSE_TMIN) & (tw < config.RESPONSE_TMAX)
            if (np.any(resp_mask)
                    and not np.isnan(bm)
                    and not np.isnan(session_breath_amp)
                    and session_breath_amp > 0):
                excursion  = float(np.max(np.abs(ep[resp_mask] - bm)))
                t["score"] = excursion / session_breath_amp
            else:
                t["score"] = np.nan

        # Gate 5: score ceiling (after scoring)
        score_max = getattr(config, "AIRFLOW_SCORE_MAX", None)
        if (score_max is not None
                and not t["rejected"]
                and t["score"] is not None
                and not np.isnan(t["score"])
                and t["score"] > score_max):
            t["rejected"]         = True
            t["score"]            = np.nan
            t["rejection_reason"] = "score_ceiling"
