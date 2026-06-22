"""
HR Processor
====================
Stateless processing functions: MFF loading, event extraction, epoch cutting,
per-trial baseline, scoring, and rejection.

Two-layer design (mirrors ECG pipeline):
  process_session()       -> MFF-derived data; written to pickle cache
  apply_analysis_params() -> config-derived data; recomputed every run

Reads the SpO2-Pulse channel (hardware BPM output) as HR. No filtering applied.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import mne

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


# -- Core session processing (cache boundary) ---------------------------------

def process_session(mff_path, ratings_df, channel, config):
    """
    Load one MFF session and return per-trial raw epoch data.

    CACHE BOUNDARY: output is written to pickle. Baseline, score, and rejection
    are NOT computed here; they are produced by apply_analysis_params() so they
    always reflect current config without reloading the MFF.

    Steps:
      1. Load full session; crop to D101-D124.
      2. Read SpO2-Pulse channel as-is (no filtering).
      3. Re-extract D110 events on cropped recording (session-relative).
      4. Align Nth trigger <-> Nth has_sound CSV row (sequential).
      5. Cut wide epoch per trial; store raw epoch + CSV metadata.

    Returns (trials, trace) where trace = {signal, sfreq, startle_samps_sec}
    for offline session plot caching. Both are None on failure.
    """
    if config.VERBOSE:
        print(f"    Loading {os.path.basename(mff_path)} (channel={channel}) ...")

    try:
        raw = mne.io.read_raw_egi(mff_path, preload=True, verbose=False,
                                  events_as_annotations=False)
    except Exception as e:
        print(f"    [!] Cannot read MFF ({e}) — skipping")
        return None, None
    sfreq = raw.info["sfreq"]

    events_df   = get_events_from_eeg(raw)
    start_samps = events_df[events_df["Channel"] == f"D{config.TRIGGER_SESSION_START}"]["Sample"].values
    end_samps   = events_df[events_df["Channel"] == f"D{config.TRIGGER_SESSION_END}"]["Sample"].values
    if len(start_samps) == 0 or len(end_samps) == 0:
        print("    [!] Task boundaries not found - skipping")
        return None, None
    raw.crop(tmin=start_samps[0] / sfreq, tmax=end_samps[-1] / sfreq)

    if channel not in raw.ch_names:
        print(f"    [!] Channel {channel!r} not found. Available: {raw.ch_names[:8]}")
        return None, None

    signal = raw.copy().pick([channel]).get_data()[0]

    events_df     = get_events_from_eeg(raw)
    startle_samps = events_df[events_df["Channel"] == f"D{config.TRIGGER_STARTLE}"]["Sample"].values
    print(f"    {len(startle_samps)} startle events found.")

    n_use         = min(len(startle_samps), len(ratings_df))
    startle_samps = startle_samps[:n_use]
    ratings_df    = ratings_df.iloc[:n_use].reset_index(drop=True)

    wide_n     = int((config.WIDE_TMAX - config.WIDE_TMIN) * sfreq)
    times_wide = np.linspace(config.WIDE_TMIN, config.WIDE_TMAX,
                             wide_n, endpoint=False)

    results = []
    for i, s_idx in enumerate(startle_samps):
        t0 = int(s_idx + config.WIDE_TMIN * sfreq)
        t1 = t0 + wide_n
        if t0 < 0 or t1 > len(signal):
            continue

        row          = ratings_df.iloc[i]
        img_col      = next((c for c in ["image_type", "file_name", "image", "image_name"]
                             if c in row.index and not pd.isna(row[c])), None)
        img_type_val = str(row[img_col]) if img_col else ""

        results.append({
            "epoch_wide":       signal[t0:t1].astype(np.float32),
            "times_wide":       times_wide,
            "sfreq":            float(sfreq),
            "channel":          channel,
            # CSV metadata (cached)
            "subjective_label": int(row["subjective_label"]),
            "image_type":       img_type_val,
            "label":            int(row["subjective_label"]),
            "valence":          float(row["valenceRating"]),
            "arousal":          float(row["arousalRating"]),
            "image_detail":     str(row.get(img_col, "?")),
            "csv_row":          row.to_dict(),
            # Filled by apply_analysis_params() -- never persisted in pickle
            "baseline_mean":    None,
            "score":            None,
            "rejected":         None,
            "epoch_anal":       None,
            "times_anal":       None,
        })

    trace = {
        "signal":            signal.astype(np.float32),
        "sfreq":             float(sfreq),
        "startle_samps_sec": startle_samps / sfreq,
    }
    return results, trace


# -- Analysis parameters (fast, no MFF access) --------------------------------

def apply_analysis_params(subj_data_dict, config):
    """
    Recompute baseline, score, rejection, and trimmed epoch for every trial.
    Called unconditionally on every run regardless of cache state.

    Moving baseline: each trial uses its own pre-trigger window defined by
    BASELINE_TMIN/TMAX, automatically correcting for slow HR drift across
    the session without needing a session-level detrend.

    Rejection (two gates, both must pass):
      1. Range gate: baseline mean outside [HR_MIN_BPM, HR_MAX_BPM].
      2. Variability z-score: baseline std > HR_Z_SCORE_THRESHOLD SDs above
         the session median std — catches probe instability where the mean
         looks plausible but the signal is jumping around.
         Disabled when HR_Z_SCORE_THRESHOLD is None.

    Score (SCORE_METHOD = 'max_minus_baseline'):
      peak HR in [SCORE_TMIN, SCORE_TMAX] minus baseline mean.
    """
    for subj, sessions in subj_data_dict.items():
        for sess_key, trials in sessions.items():
            if not trials:
                continue

            # Pass 1 — label resolution, compute baseline, collect session stds
            bl_stds = []
            pre_masks = []
            for t in trials:
                if "subjective_label" not in t:
                    t["subjective_label"] = t.get("label", 2)

                if config.USE_SUBJECTIVE_TRIAL_TYPE:
                    t["label"] = t["subjective_label"]
                else:
                    _id = t.get("image_detail", "")
                    img_type = str(_id if _id and _id != "?" else t.get("image_type", "")).lower().strip()
                    if "negative" in img_type or img_type == "1":
                        t["label"] = 1
                    elif "neutral" in img_type or img_type == "2":
                        t["label"] = 2
                    else:
                        t["label"] = t["subjective_label"]

                tw = t["times_wide"]
                ep = t["epoch_wide"]
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
                ep = t["epoch_wide"]
                bm = t["baseline_mean"]
                bs = float(np.std(ep[pre_mask])) if np.any(pre_mask) else np.nan

                range_reject = np.isnan(bm) or bm < config.HR_MIN_BPM or bm > config.HR_MAX_BPM

                z_reject = False
                threshold = getattr(config, "HR_Z_SCORE_THRESHOLD", None)
                if threshold is not None and not np.isnan(bs) and sess_std_std > 0:
                    z = (bs - sess_std_median) / sess_std_std
                    z_reject = bool(z > threshold)

                t["rejected"] = bool(range_reject or z_reject)

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
                else:
                    t["score"] = np.nan

                score_max = getattr(config, "SCORE_MAX_PCT", None)
                if score_max is not None and not t["rejected"] and not np.isnan(t["score"]):
                    if abs(t["score"]) > score_max:
                        t["rejected"] = True
                        t["score"]    = np.nan
