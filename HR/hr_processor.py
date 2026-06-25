"""
HR Processor
====================
Stateless processing functions: MFF loading, event extraction, epoch cutting,
per-trial baseline, scoring, and rejection.

Two-layer design (mirrors Airflow pipeline):
  process_session()       -> Layer 1: raw session signal + trigger positions + CSV metadata.
                             Written to pickle. No epoch cutting, no artifact cleaning.
  apply_analysis_params() -> Layer 2: epoch cutting at current WIDE_TMIN/TMAX, artifact
                             cleaning, baseline, rejection, scoring. Called every run.

Layer 1 requires FORCE_RELOAD: HR_CHANNEL, HR_CACHE_SFREQ, trigger codes.
Layer 2 (all free to tune): WIDE_TMIN/TMAX, ANAL_TMIN/TMAX, BASELINE_*, SCORE_*,
  HR_MIN/MAX_BPM, HR_Z_SCORE_THRESHOLD, FLAT_*, SPIKE_*, SCORE_MAX/MIN_PCT.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import mne
from scipy.signal import resample_poly

warnings.filterwarnings("ignore", category=RuntimeWarning)


# -- File discovery -----------------------------------------------------------

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


# -- Event extraction ---------------------------------------------------------

def get_events_from_eeg(raw_eeg):
    """
    Extract DIN trigger onsets. Contiguous above-threshold samples are collapsed
    to the first sample of each run so a 5 ms pulse yields exactly 1 event.
    Returns DataFrame ['Channel', 'Sample'] sorted by Sample.
    """
    din_names = [ch for ch in raw_eeg.ch_names if ch.startswith("D")]
    if not din_names:
        return pd.DataFrame(columns=["Channel", "Sample"])
    raw_din = raw_eeg.copy().pick(din_names)
    event_data = []
    for ch_idx, ch_name in enumerate(raw_din.ch_names):
        data = raw_din.get_data(picks=[ch_idx])[0]
        threshold = np.max(data) * 0.9
        samps = np.where(data > threshold)[0]
        if len(samps) == 0:
            continue
        gaps   = np.where(np.diff(samps) > 1)[0]
        onsets = np.concatenate([[samps[0]], samps[gaps + 1]])
        for sample in onsets:
            event_data.append({"Channel": ch_name, "Sample": int(sample)})
    if not event_data:
        return pd.DataFrame(columns=["Channel", "Sample"])
    return pd.DataFrame(event_data).sort_values("Sample").reset_index(drop=True)


# -- CSV loading --------------------------------------------------------------

def load_and_classify_ratings(csv_path):
    """Load ratings CSV, keep only sound trials, assign subjective_label."""
    df = pd.read_csv(csv_path)
    df = df[df["has_sound"] == True].reset_index(drop=True)

    def categorize(row):
        if row["arousalRating"] >= 7 or row["valenceRating"] <= 3:
            return 1   # Negative
        return 2       # Neutral

    df["subjective_label"] = df.apply(categorize, axis=1)
    return df


# -- Layer 1: MFF extraction (cache boundary) ---------------------------------

def process_session(mff_path, ratings_df, channel, config):
    """
    Load one MFF session and return raw session data.

    CACHE BOUNDARY: no epoch cutting, no artifact cleaning. Output is written
    to pickle and represents pure Layer 1 data. All signal processing lives in
    apply_analysis_params() so WIDE_TMIN/TMAX and all analysis params can be
    tuned without re-reading MFFs.

    Downsampling: if config.HR_CACHE_SFREQ is set, the signal is anti-aliased
    (Kaiser window) and stored at that rate. 10 Hz is sufficient for SpO2-Pulse
    (0.5 Hz HR content; Nyquist = 5 Hz). Trigger positions are converted to the
    downsampled time base.

    Returns (session_dict, None) or (None, None) on failure.
      session_dict = {
        'raw_signal':    float32 ndarray  (full session at sfreq Hz)
        'sfreq':         float
        'sfreq_orig':    float            (original MFF rate, for reference)
        'channel':       str
        'startle_samps': int64 ndarray   (session-relative, downsampled domain)
        'trials_meta':   list of dicts   (one per trial; no signal data)
      }
    """
    cache_sfreq = getattr(config, "HR_CACHE_SFREQ", None)

    if config.VERBOSE:
        print(f"    Loading {os.path.basename(mff_path)} (channel={channel}) ...")

    try:
        raw = mne.io.read_raw_egi(mff_path, preload=True, verbose=False,
                                  events_as_annotations=False)
    except Exception as e:
        print(f"    [!] Cannot read MFF ({e}) — skipping")
        return None, None
    sfreq_orig = raw.info["sfreq"]

    events_df   = get_events_from_eeg(raw)
    start_samps = events_df[events_df["Channel"] == f"D{config.TRIGGER_SESSION_START}"]["Sample"].values
    end_samps   = events_df[events_df["Channel"] == f"D{config.TRIGGER_SESSION_END}"]["Sample"].values
    if len(start_samps) == 0 or len(end_samps) == 0:
        print("    [!] Task boundaries not found — skipping")
        return None, None
    raw.crop(tmin=start_samps[0] / sfreq_orig, tmax=end_samps[-1] / sfreq_orig)

    if channel not in raw.ch_names:
        print(f"    [!] Channel {channel!r} not found. Available: {raw.ch_names[:8]}")
        return None, None

    raw_channel = raw.copy().pick([channel]).get_data()[0]

    events_df     = get_events_from_eeg(raw)
    startle_samps = events_df[events_df["Channel"] == f"D{config.TRIGGER_STARTLE}"]["Sample"].values
    print(f"    {len(startle_samps)} startle events found.")

    n_use         = min(len(startle_samps), len(ratings_df))
    startle_samps = startle_samps[:n_use]
    ratings_df    = ratings_df.iloc[:n_use].reset_index(drop=True)

    # Optional downsampling
    if cache_sfreq and cache_sfreq < sfreq_orig:
        down     = max(1, int(round(sfreq_orig / cache_sfreq)))
        raw_ds   = resample_poly(raw_channel.astype(np.float64), 1, down).astype(np.float32)
        sfreq_ds = sfreq_orig / down
        samps_ds = np.round(startle_samps / down).astype(np.int64)
    else:
        raw_ds   = raw_channel.astype(np.float32)
        sfreq_ds = sfreq_orig
        samps_ds = startle_samps.copy().astype(np.int64)

    # Per-trial metadata (CSV; no signal)
    # image_type stored first — filename columns in image_detail would silently
    # fall back to subjective_label if stored instead of the objective category.
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

    session = {
        "raw_signal":    raw_ds,
        "sfreq":         float(sfreq_ds),
        "sfreq_orig":    float(sfreq_orig),
        "channel":       channel,
        "startle_samps": samps_ds,
        "trials_meta":   trials_meta,
    }
    return session, None


# -- Artifact cleaning --------------------------------------------------------

def _clean_epoch(epoch, sfreq, flat_min_sec=3.0, spike_thresh_bpm=20.0):
    """
    Detect and interpolate two artifact types in a SpO2-Pulse BPM epoch.

    SpO2-Pulse is a staircase signal (constant BPM blocks ~2 s each), so both
    detections operate on BLOCKS rather than individual samples — O(n_blocks)
    instead of O(n_samples), which is ~1000x faster than a sliding median filter.

    Type 1 — Flat plateau: block duration >= flat_min_sec (probe dropout).
    Type 2 — Spike: block value deviates > spike_thresh_bpm from the median
      of its ±3 surrounding blocks. SpO2 step transitions are 1–4 BPM; spikes
      are >>20 BPM, so the threshold cleanly separates them without touching
      legitimate HR responses.

    Both types are linearly interpolated using nearest clean neighbors as anchors.
    Returns (cleaned_epoch, artifact_mask) — boolean mask marks interpolated samples.
    """
    from scipy.interpolate import interp1d

    ep = epoch.copy().astype(float)
    n  = len(ep)
    artifact_mask = np.zeros(n, dtype=bool)

    changes      = np.concatenate([[0], np.where(np.diff(ep) != 0)[0] + 1, [n]])
    block_starts = [int(changes[i])   for i in range(len(changes) - 1)]
    block_ends   = [int(changes[i+1]) for i in range(len(changes) - 1)]
    block_vals   = np.array([ep[s] for s in block_starts], dtype=float)
    n_blocks     = len(block_vals)
    min_samp     = max(1, int(flat_min_sec * sfreq))

    for i, (bs, be, bv) in enumerate(zip(block_starts, block_ends, block_vals)):
        is_flat  = (be - bs) >= min_samp
        is_spike = False
        if spike_thresh_bpm is not None and spike_thresh_bpm > 0:
            lo = max(0, i - 3)
            hi = min(n_blocks, i + 4)
            local_med = float(np.median(block_vals[lo:hi]))
            is_spike  = abs(bv - local_med) > spike_thresh_bpm
        if is_flat or is_spike:
            artifact_mask[bs:be] = True

    if np.any(artifact_mask):
        clean_idx = np.where(~artifact_mask)[0]
        if len(clean_idx) >= 2:
            f = interp1d(clean_idx, ep[clean_idx], kind='linear',
                         bounds_error=False,
                         fill_value=(ep[clean_idx[0]], ep[clean_idx[-1]]))
            ep[artifact_mask] = f(np.where(artifact_mask)[0])
        elif len(clean_idx) == 1:
            ep[artifact_mask] = ep[clean_idx[0]]

    return ep, artifact_mask


def _window_artifact_fractions(artifact_mask, window_mask):
    """
    For a boolean window_mask over the epoch, return:
      total_frac    — fraction of window samples that were artifacts
      max_contig    — largest single contiguous artifact run / window length
    """
    from scipy.ndimage import label as ndlabel

    n_win = int(np.sum(window_mask))
    if n_win == 0:
        return 0.0, 0.0

    win_art    = artifact_mask & window_mask
    total_frac = float(np.sum(win_art)) / n_win

    if not np.any(win_art):
        return total_frac, 0.0

    labeled, n_segs = ndlabel(win_art)
    max_seg = max(int(np.sum(labeled == k)) for k in range(1, n_segs + 1))
    return total_frac, float(max_seg) / n_win


# -- Layer 2: analysis (fast, no MFF access) ----------------------------------

def apply_analysis_params(sessions_cache, config):
    """
    Run epoch cutting + artifact cleaning + baseline + rejection + scoring.
    Called unconditionally on every run. Returns (trials_data, traces).

    Accepts two cache formats:
      NEW (sessions): {subj: {sess_key: session_dict with raw_signal, startle_samps, ...}}
        → re-cuts epochs from raw signal (WIDE_TMIN/TMAX tunable)
      OLD (trials):   {subj: {sess_key: [list of trial dicts with epoch_wide pre-cut]}}
        → runs analysis on existing epochs (WIDE_TMIN/TMAX fixed at cache time)
    """
    trials_data = {}
    traces      = {}

    for subj, sess_dict in sorted(sessions_cache.items()):
        trials_data[subj] = {}
        traces[subj]      = {}

        for sess_key, sess in sess_dict.items():

            # ── Old format: sess is already a list of trial dicts ──────────
            if isinstance(sess, list):
                trials = [dict(t) for t in sess]   # shallow copy so we don't mutate cache
                # reset analysis fields before re-running
                for t in trials:
                    t["baseline_mean"]    = None
                    t["score"]            = None
                    t["rejected"]         = None
                    t["rejection_reason"] = ""
                    t["epoch_anal"]       = None
                    t["times_anal"]       = None
                    t["artifact_mask"]    = None
                    t["any_artifact"]     = False
                _run_hr_analysis(trials, config)
                trials_data[subj][sess_key] = trials
                traces[subj][sess_key]      = {}   # no raw trace available in old format
                continue

            # ── New format: sess has raw_signal + startle_samps + trials_meta ─
            raw_sig       = sess["raw_signal"]
            sfreq         = sess["sfreq"]
            startle_samps = sess["startle_samps"]
            trials_meta   = sess["trials_meta"]

            traces[subj][sess_key] = {
                "signal":            raw_sig,
                "sfreq":             sfreq,
                "startle_samps_sec": startle_samps / sfreq,
            }

            wide_n     = int((config.WIDE_TMAX - config.WIDE_TMIN) * sfreq)
            times_wide = np.linspace(config.WIDE_TMIN, config.WIDE_TMAX,
                                     wide_n, endpoint=False)

            trials = []
            for star_samp, meta in zip(startle_samps, trials_meta):
                t0 = int(star_samp + config.WIDE_TMIN * sfreq)
                t1 = t0 + wide_n
                if t0 < 0 or t1 > len(raw_sig):
                    continue
                trials.append({
                    "epoch_wide":       raw_sig[t0:t1].copy(),
                    "times_wide":       times_wide,
                    "sfreq":            sfreq,
                    "channel":          sess["channel"],
                    "subjective_label": meta["subjective_label"],
                    "label":            meta["subjective_label"],
                    "valence":          meta["valence"],
                    "arousal":          meta["arousal"],
                    "image_detail":     meta["image_detail"],
                    "image_type":       meta.get("image_type", ""),
                    "baseline_mean":    None,
                    "score":            None,
                    "rejected":         None,
                    "rejection_reason": "",
                    "epoch_anal":       None,
                    "times_anal":       None,
                })

            _run_hr_analysis(trials, config)
            trials_data[subj][sess_key] = trials

    return trials_data, traces


# -- Internal: label + baseline + rejection + scoring -------------------------

def _run_hr_analysis(trials, config):
    """Resolve labels, clean epochs, compute baseline, apply rejection gates, score."""
    if not trials:
        return

    # Pass 1 — label resolution, artifact cleaning, baseline, session std collection
    bl_stds   = []
    pre_masks = []

    for t in trials:
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

        _ep_clean, _art_mask = _clean_epoch(
            t["epoch_wide"],
            t["sfreq"],
            flat_min_sec     = getattr(config, "FLAT_MIN_SEC",      3.0),
            spike_thresh_bpm = getattr(config, "SPIKE_THRESH_BPM", 20.0),
        )
        t["epoch_wide"]    = _ep_clean
        t["artifact_mask"] = _art_mask
        t["any_artifact"]  = bool(np.any(_art_mask))

        # Optional lowpass smoothing applied after artifact cleaning.
        # Smooths the BPM staircase into a continuous curve so the max peak
        # reflects sustained HR elevation rather than the highest discrete step.
        # Artifact rejection gates still use the raw artifact_mask (not the filtered signal).
        lp_hz = getattr(config, "SIGNAL_LOWPASS_HZ", None)
        if lp_hz is not None and lp_hz > 0:
            from scipy.signal import butter, filtfilt
            sfreq = t["sfreq"]
            nyq   = sfreq / 2.0
            if lp_hz < nyq:
                b, a = butter(4, lp_hz / nyq, btype="low")
                t["epoch_filtered"] = filtfilt(b, a, _ep_clean)
            else:
                t["epoch_filtered"] = _ep_clean.copy()
        else:
            t["epoch_filtered"] = _ep_clean.copy()

        tw       = t["times_wide"]
        ep       = t["epoch_filtered"]   # filtered signal used for baseline + scoring
        pre_mask = (tw >= config.BASELINE_TMIN) & (tw < config.BASELINE_TMAX)
        pre_masks.append(pre_mask)

        if np.any(pre_mask):
            bm = float(np.mean(ep[pre_mask]) if config.BASELINE_METHOD == "mean"
                       else np.median(ep[pre_mask]))
            bs = float(np.std(ep[pre_mask]))
        else:
            bm = bs = np.nan
        t["baseline_mean"] = bm
        bl_stds.append(bs)

    sess_std_median = float(np.nanmedian(bl_stds))
    sess_std_std    = float(np.nanstd(bl_stds))

    # Pass 2 — rejection, score, trim
    for t, pre_mask in zip(trials, pre_masks):
        tw = t["times_wide"]
        ep = t["epoch_filtered"]   # use filtered signal for scoring
        bm = t["baseline_mean"]
        bs = float(np.std(ep[pre_mask])) if np.any(pre_mask) else np.nan

        range_reject = np.isnan(bm) or bm < config.HR_MIN_BPM or bm > config.HR_MAX_BPM

        z_reject  = False
        threshold = getattr(config, "HR_Z_SCORE_THRESHOLD", None)
        if threshold is not None and not np.isnan(bs) and sess_std_std > 0:
            z_reject = bool((bs - sess_std_median) / sess_std_std > threshold)

        art_mask = t.get("artifact_mask", np.zeros(len(tw), dtype=bool))

        bl_mask = (tw >= config.BASELINE_TMIN) & (tw < config.BASELINE_TMAX)
        bl_total, bl_contig = _window_artifact_fractions(art_mask, bl_mask)
        bl_reject = (
            bl_total  > getattr(config, "FLAT_BASELINE_TOTAL_MAX",      0.50) or
            bl_contig > getattr(config, "FLAT_BASELINE_CONTIGUOUS_MAX", 0.40)
        )

        sc_mask = (tw >= config.SCORE_TMIN) & (tw <= config.SCORE_TMAX)
        sc_total, sc_contig = _window_artifact_fractions(art_mask, sc_mask)
        sc_reject = (
            sc_total  > getattr(config, "FLAT_SCORE_TOTAL_MAX",      0.50) or
            sc_contig > getattr(config, "FLAT_SCORE_CONTIGUOUS_MAX", 0.40)
        )

        # New gate: dominant flat block in score window.
        # Catches trials where the BPM never updates during the response window
        # (block < FLAT_MIN_SEC so not flagged as dropout, but still uninformative).
        # Checks the raw epoch (pre-interpolation artefact) for block dominance.
        sc_dominant_reject = False
        sc_dom_thresh = getattr(config, "SCORE_FLAT_DOMINANT_MAX", None)
        if sc_dom_thresh is not None and np.any(sc_mask):
            sc_ep = t["epoch_wide"][sc_mask]   # raw staircase, not interpolated
            changes = np.concatenate([[0], np.where(np.diff(sc_ep) != 0)[0] + 1, [len(sc_ep)]])
            dominant_frac = float(np.max(np.diff(changes))) / len(sc_ep)
            sc_dominant_reject = dominant_frac > sc_dom_thresh

        reasons = []
        if range_reject:        reasons.append("bpm_range")
        if z_reject:            reasons.append("noisy_baseline")
        if bl_reject:           reasons.append("flat_baseline")
        if sc_reject:           reasons.append("flat_score_window")
        if sc_dominant_reject:  reasons.append("flat_score_dominant")
        t["rejected"]         = bool(reasons)
        t["rejection_reason"] = ", ".join(reasons) if reasons else ""

        anal_mask       = (tw >= config.ANAL_TMIN) & (tw < config.ANAL_TMAX)
        t["epoch_anal"] = ep[anal_mask]
        t["times_anal"] = tw[anal_mask]

        if t["rejected"]:
            t["score"] = np.nan
        elif config.SCORE_METHOD == "max_minus_baseline":
            score_mask = (tw >= config.SCORE_TMIN) & (tw <= config.SCORE_TMAX)
            if np.any(score_mask) and not np.isnan(bm) and bm != 0:
                t["score"] = float((np.max(ep[score_mask]) - bm) / bm * 100)
            else:
                t["score"] = np.nan
        elif config.SCORE_METHOD == "mean_minus_baseline":
            score_mask = (tw >= config.SCORE_TMIN) & (tw <= config.SCORE_TMAX)
            if np.any(score_mask) and not np.isnan(bm) and bm != 0:
                t["score"] = float((np.mean(ep[score_mask]) - bm) / bm * 100)
            else:
                t["score"] = np.nan
        else:
            t["score"] = np.nan

        score_max = getattr(config, "SCORE_MAX_PCT", None)
        if score_max is not None and not t["rejected"] and not np.isnan(t["score"]):
            if abs(t["score"]) > score_max:
                t["rejected"]         = True
                t["score"]            = np.nan
                t["rejection_reason"] = "score_too_large"

        score_min = getattr(config, "SCORE_MIN_PCT", None)
        if score_min is not None and not t["rejected"] and not np.isnan(t["score"]):
            if t["score"] < score_min:
                t["rejected"]         = True
                t["score"]            = np.nan
                t["rejection_reason"] = "score_negative"
