"""
HR Processor
====================
Stateless processing functions: MFF loading, epoch cutting, per-trial baseline,
scoring, and rejection.

Two-layer design (mirrors Airflow pipeline):
  process_session()       -> Layer 1: raw session signal + ground-truth trial
                             epochs (trial_epochs.build_trial_epochs) + CSV metadata.
                             Written to pickle. No epoch cutting, no artifact cleaning.
  apply_analysis_params() -> Layer 2: epoch cutting at the ground-truth trial
                             boundaries, artifact cleaning, baseline, rejection,
                             scoring. Called every run.

Layer 1 requires FORCE_RELOAD: HR_CHANNEL, HR_CACHE_SFREQ.
Layer 2 (all free to tune): ANAL_TMIN/TMAX, SCORE_*, HR_MIN/MAX_BPM,
  HR_Z_SCORE_THRESHOLD, FLAT_*, SPIKE_*, SCORE_MAX/MIN_PCT.

Epoch/baseline/response window timing is NOT a free parameter -- it is ground
truth from the D105 -> D{trigger_num} -> D105 trigger cycle (see
Startle/trial_epochs.py). DIN-trigger handling here is retired in favor of
trial_epochs.build_trial_epochs(), which is now the sole source of trial
boundaries for every pipeline.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import mne
from scipy.signal import resample_poly

import trial_epochs

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
            if (f"_{keyword}_" in name.lower() and name.lower().endswith(".mff")
                    and not name.startswith("._")):
                return os.path.join(root, name)
    return None


def find_csv_by_suffix(startle_folder, suffix):
    for root, _, files in os.walk(startle_folder):
        for name in sorted(files):
            if "demo" not in name.lower() and name.lower().endswith(".csv"):
                if re.search(rf"{re.escape(suffix)}\.csv$", name, flags=re.IGNORECASE):
                    return os.path.join(root, name)
    return None


# -- CSV loading --------------------------------------------------------------
# NOTE: load_and_classify_ratings keeps the has_sound filter and is still used
# by plot_subjective_negative_percentage() (a CSV-only plot, no epoching
# involved). For anything that feeds process_session(), use
# trial_epochs.load_all_trials_ratings() instead -- has_sound is not a filter
# criterion for epoching anymore, every CSV row gets a trial.

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
    Load one MFF session: ground-truth trial epochs (trial_epochs.build_trial_epochs)
    + raw signal, downsampled for caching.

    ratings_df must be the FULL, unfiltered CSV (trial_epochs.load_all_trials_ratings)
    -- every row becomes a trial, sound and no-sound alike.

    Event/epoch extraction always runs on the original full-rate, full-channel
    raw -- DIN triggers must not be resampled or have other channels dropped
    before detection. Only after build_trial_epochs() succeeds do we pick the
    analysis channel and downsample for the cache.

    CACHE BOUNDARY: no artifact cleaning here -- that's Layer 2
    (apply_analysis_params), so SCORE_*/rejection params can be tuned without
    re-reading MFFs.

    Raises trial_epochs.TriggerAlignmentError if this session's D105/D{code}
    triggers don't match its CSV -- callers must let this propagate (or
    explicitly catch and skip the session) rather than silently continuing
    with an untrustworthy trial/CSV alignment.

    Returns (session_dict, None) or (None, None) on failure.
      session_dict = {
        'signal_raw':    float32 ndarray  (full session at sfreq Hz)
        'sfreq':         float            (cache rate)
        'sfreq_orig':    float            (original MFF rate, for reference)
        'channel':       str
        'trials_meta':   list of dicts    (one per trial; no signal data)
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

    trials = trial_epochs.build_trial_epochs(raw, ratings_df)  # raises on mismatch

    if channel not in raw.ch_names:
        print(f"    [!] Channel {channel!r} not found. Available: {raw.ch_names[:8]}")
        return None, None

    raw_channel = raw.copy().pick([channel]).get_data()[0]

    # Optional downsampling
    if cache_sfreq and cache_sfreq < sfreq_orig:
        down     = max(1, int(round(sfreq_orig / cache_sfreq)))
        raw_ds   = resample_poly(raw_channel.astype(np.float64), 1, down).astype(np.float32)
        sfreq_ds = sfreq_orig / down
    else:
        raw_ds   = raw_channel.astype(np.float32)
        sfreq_ds = sfreq_orig

    # Trigger sample indices are in sfreq_orig space (from the un-resampled
    # raw); convert to seconds here so they're valid regardless of cache_sfreq.
    trials_meta = []
    for t in trials:
        row = ratings_df.iloc[t["trial_index"]]
        img_col = next((c for c in ["image_type", "image", "image_name"]
                        if c in row.index and not pd.isna(row.get(c))), None)
        img_type_val = (str(row["image_type"]).lower().strip()
                        if "image_type" in row.index and not pd.isna(row.get("image_type"))
                        else "")
        trials_meta.append({
            "trial_index":        t["trial_index"],
            "code_value":         t["code_value"],
            "has_sound":          t["d110_sample"] is not None,
            "d105_time":          t["d105_sample"] / sfreq_orig,
            "code_time":          t["code_sample"] / sfreq_orig,
            "d110_time":          (t["d110_sample"] / sfreq_orig
                                    if t["d110_sample"] is not None else None),
            "baseline_range_sec": tuple(s / sfreq_orig for s in t["baseline_range"]),
            "response_range_sec": tuple(s / sfreq_orig for s in t["response_range"]),
            "subjective_label":   int(row["subjective_label"]),
            "valence":            float(row["valenceRating"]),
            "arousal":            float(row["arousalRating"]),
            "image_detail":       str(row.get(img_col, "?")) if img_col else "?",
            "image_type":         img_type_val,
        })

    session = {
        "signal_raw":  raw_ds,
        "sfreq":       float(sfreq_ds),
        "sfreq_orig":  float(sfreq_orig),
        "channel":     channel,
        "trials_meta": trials_meta,
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
      NEW (sessions): {subj: {sess_key: session_dict with signal_raw, trials_meta, ...}}
        -> cuts epochs at the ground-truth trial boundaries from trials_meta
        (d105_time/code_time/response_range_sec) -- not a tunable window.
      OLD (trials):   {subj: {sess_key: [list of trial dicts with epoch_wide pre-cut]}}
        -> runs analysis on existing epochs (whatever window they were cut with)
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

            # ── New format: sess has signal_raw + trials_meta (ground-truth ranges) ─
            raw_sig     = sess["signal_raw"]
            sfreq       = sess["sfreq"]
            trials_meta = sess["trials_meta"]

            d110_secs = np.array([m["d110_time"] for m in trials_meta
                                   if m["d110_time"] is not None])
            traces[subj][sess_key] = {
                "signal":            raw_sig,
                "sfreq":             sfreq,
                "startle_samps_sec": d110_secs,
            }

            # ── Cut one ground-truth epoch per trial ──────────────────────────
            # Span: [d105_time, response_range_sec[1]) -- baseline + response,
            # contiguous by construction. t=0 is set at code_time (the
            # picture-identity trigger), so the baseline portion is tw<0 and
            # the response portion is tw>=0. Epoch length varies per trial --
            # this is ground truth, not a tunable WIDE_TMIN/TMAX window.
            trials = []
            for meta in trials_meta:
                d105_idx = int(round(meta["d105_time"] * sfreq))
                code_idx = int(round(meta["code_time"]  * sfreq))
                end_idx  = int(round(meta["response_range_sec"][1] * sfreq))
                if d105_idx < 0 or end_idx > len(raw_sig) or end_idx <= d105_idx:
                    continue

                times_wide = ((np.arange(d105_idx, end_idx) - code_idx)
                              / sfreq)

                trials.append({
                    "epoch_wide":       raw_sig[d105_idx:end_idx].copy(),
                    "times_wide":       times_wide,
                    "sfreq":            sfreq,
                    "channel":          sess["channel"],
                    "trial_index":      meta["trial_index"],
                    "code_value":       meta["code_value"],
                    "has_sound":        meta["has_sound"],
                    "d110_rel":         (meta["d110_time"] - meta["code_time"]
                                          if meta["d110_time"] is not None else None),
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
        # Ground truth: the entire pre-code-onset span IS the baseline (tw<0 by
        # construction -- t=0 is code_time, the epoch starts at d105_time). No
        # BASELINE_TMIN/TMAX cutoff anymore.
        pre_mask = tw < 0
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

        bl_mask = tw < 0   # ground truth, see Pass 1
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

        # Epoch length now varies per trial (ground-truth spans, no shared
        # WIDE_TMIN/TMAX) -- only trials whose epoch fully covers
        # [ANAL_TMIN, ANAL_TMAX) get a non-None epoch_anal/times_anal,
        # so every downstream consumer that stacks/averages across trials
        # (already guarded with `if epoch_anal is None: continue`) gets
        # consistently-shaped arrays instead of an inhomogeneous crash.
        # Excluded trials are mainly the last trial of a session (closed
        # early by D124) -- they still get scored/rejected normally above,
        # this only affects the display/average-timecourse window.
        anal_mask        = (tw >= config.ANAL_TMIN) & (tw < config.ANAL_TMAX)
        expected_anal_n  = int(round((config.ANAL_TMAX - config.ANAL_TMIN) * t["sfreq"]))
        if int(np.sum(anal_mask)) == expected_anal_n:
            t["epoch_anal"] = ep[anal_mask]
            t["times_anal"] = tw[anal_mask]
        else:
            t["epoch_anal"] = None
            t["times_anal"] = None

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
