"""
ECG Signal Processor
====================
Core processing functions: MFF loading, session cropping, ECG filtering,
R-peak detection, IBI computation, epoch extraction, and CSV alignment.

Two-layer cache design (mirrors emg_raw_potentiation.py):
  - process_session()       → returns MFF-derived data; written to pickle
  - apply_analysis_params() → returns parameter-derived data; never written to pickle

Filter params, R-peak settings, and epoch windows require FORCE_RELOAD_ECG = True.
Rejection thresholds, score windows, and analysis windows take effect immediately.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy import signal
import mne

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── File discovery ────────────────────────────────────────────────────────────

def find_startle_output_folder(subject_path):
    for name in os.listdir(subject_path):
        if name.lower() == "startle output" and os.path.isdir(os.path.join(subject_path, name)):
            return os.path.join(subject_path, name)
    csv_files = [f for f in os.listdir(subject_path) if f.lower().endswith(".csv")]
    if csv_files:
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


# ── Event extraction ──────────────────────────────────────────────────────────

def get_events_from_eeg(raw_eeg):
    """
    Extract DIN trigger onset samples from an MNE Raw object.

    Each DIN channel is thresholded at 90% of its maximum. Contiguous runs of
    above-threshold samples are collapsed to the first sample of each run —
    a 5 ms pulse at 1000 Hz yields exactly 1 event, not 5.  Without this
    collapsing, every trigger pulse floods the event list and corrupts the
    trial-to-CSV row alignment.

    Returns DataFrame with columns ['Channel', 'Sample'], sorted by Sample.
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
        # Collapse contiguous runs → keep only the first sample of each pulse
        gaps   = np.where(np.diff(samps) > 1)[0]
        onsets = np.concatenate([[samps[0]], samps[gaps + 1]])
        for sample in onsets:
            event_data.append({"Channel": ch_name, "Sample": int(sample)})

    if not event_data:
        return pd.DataFrame(columns=["Channel", "Sample"])
    return pd.DataFrame(event_data).sort_values("Sample").reset_index(drop=True)


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_and_classify_ratings(csv_path):
    """Load a ratings CSV, keep only sound trials, assign subjective_label."""
    df = pd.read_csv(csv_path)
    df = df[df["has_sound"] == True].reset_index(drop=True)

    def categorize(row):
        if row["arousalRating"] >= 7 or row["valenceRating"] <= 3:
            return 1   # Negative
        return 2       # Neutral

    df["subjective_label"] = df.apply(categorize, axis=1)
    return df


# ── R-peak detection ──────────────────────────────────────────────────────────

def detect_r_peaks(ecg_signal, sfreq, config):
    """
    Detect R-peaks in a filtered full-session ECG signal.

    Uses scipy.signal.find_peaks with:
      - minimum inter-peak distance from config.RPEAK_MIN_DISTANCE_MS
      - prominence threshold = config.RPEAK_PROMINENCE_FRAC × peak-to-peak amplitude

    Auto-detects polarity: tries both positive and negative sides of the signal
    and keeps whichever yields more peaks.  This handles EGI recordings where
    the ECG at E256 can be either upright or inverted across subjects without
    requiring a manual RPEAK_INVERT flag.
    """
    min_dist   = int(config.RPEAK_MIN_DISTANCE_MS / 1000.0 * sfreq)
    # Robust PTP: 1st–99th percentile so a single artifact spike doesn't inflate
    # the prominence threshold and wipe out all genuine R-peaks.
    ptp        = np.percentile(ecg_signal, 99) - np.percentile(ecg_signal, 1)
    prominence = config.RPEAK_PROMINENCE_FRAC * ptp

    pos, _ = signal.find_peaks( ecg_signal, distance=min_dist, prominence=prominence)
    neg, _ = signal.find_peaks(-ecg_signal, distance=min_dist, prominence=prominence)
    return neg if len(neg) > len(pos) else pos


# ── IBI computation ───────────────────────────────────────────────────────────

def compute_ibi(r_peaks, sfreq):
    """
    Compute inter-beat intervals from consecutive R-peak sample indices.

    Returns (ibi_times_s, ibi_ms):
      ibi_times_s — midpoint time of each RR interval, session-relative (seconds)
      ibi_ms      — RR interval duration (milliseconds)
    Both arrays have shape (len(r_peaks) - 1,).
    Returns two empty arrays if fewer than 2 R-peaks provided.
    """
    if len(r_peaks) < 2:
        return np.array([]), np.array([])

    ibi_times_s = (r_peaks[:-1] + r_peaks[1:]) / 2.0 / sfreq
    ibi_ms      = (r_peaks[1:]  - r_peaks[:-1]) / sfreq * 1000.0
    return ibi_times_s, ibi_ms


# ── HR interpolation ──────────────────────────────────────────────────────────

def interpolate_hr_timeseries(ibi_times_trigger, ibi_ms, target_times):
    """
    Interpolate a uniform HR timecourse (bpm) from sparse IBI data.

    ibi_times_trigger — IBI midpoint times in seconds, trigger-relative.
    ibi_ms            — IBI durations in ms at each midpoint.
    target_times      — Uniform time axis in seconds, trigger-relative.

    Returns HR array in bpm, shape (len(target_times),).
    Returns array of NaN if fewer than 2 IBI points provided.
    """
    if len(ibi_times_trigger) < 2:
        return np.full(len(target_times), np.nan)

    hr_at_midpoints = 60000.0 / ibi_ms
    return np.interp(target_times, ibi_times_trigger, hr_at_midpoints)


# ── Core session processing ───────────────────────────────────────────────────

def process_session(mff_path, ratings_df, channel, config):
    """
    Load one MFF session, apply ECG pipeline, return per-trial data.

    CACHE BOUNDARY: everything this function returns is written to the pickle.
    Rejection, scoring, and analysis-window epochs are NOT computed here;
    they are produced by apply_analysis_params() on every run.

    Steps (mirror emg_raw_potentiation.py structure):
      1. Load full session; extract DIN events; crop to D101–D124.
      2. Pick ECG channel; HP+LP filter the cropped signal.
      3. Detect R-peaks and compute IBIs on full filtered session.
      4. Re-extract DIN events on cropped recording (now session-relative).
      5. Align: Nth D110 trigger ↔ Nth has_sound CSV row (sequential).
      6. For each trial: cut wide epoch, extract epoch R-peaks and IBIs,
         compute baseline mean HR, store metadata from CSV row.

    Returns list of trial dicts, or None if session boundaries not found.
    """
    if config.VERBOSE:
        print(f"    Loading {os.path.basename(mff_path)} (channel={channel}) ...")

    raw   = mne.io.read_raw_egi(mff_path, preload=True, verbose=False,
                                events_as_annotations=False)
    sfreq = raw.info["sfreq"]

    # Pass 1: find session boundaries, then crop (sample 0 = session start after crop).
    events_df   = get_events_from_eeg(raw)
    start_samps = events_df[events_df["Channel"] == f"D{config.TRIGGER_SESSION_START}"]["Sample"].values
    end_samps   = events_df[events_df["Channel"] == f"D{config.TRIGGER_SESSION_END}"]["Sample"].values
    if len(start_samps) == 0 or len(end_samps) == 0:
        print("    [!] Task boundaries not found - skipping")
        return None
    raw.crop(tmin=start_samps[0] / sfreq, tmax=end_samps[-1] / sfreq)

    if channel not in raw.ch_names:
        print(f"    [!] Channel {channel} not found - skipping")
        return None
    ch_data_raw = raw.copy().pick([channel]).get_data()[0]

    # Welch PSD on unfiltered signal — QC only, stored in every trial dict.
    nperseg           = int(min(len(ch_data_raw), 2048))
    psd_freqs, psd_values = signal.welch(ch_data_raw, fs=sfreq, nperseg=nperseg)

    # HP → LP filter. No rectification — ECG waveform must be preserved for R-peak detection.
    nyquist    = sfreq / 2.0
    b_hp, a_hp = signal.butter(config.HP_FILTER_ORDER, config.HP_CUTOFF_HZ / nyquist, btype="high")
    sig_hp     = signal.filtfilt(b_hp, a_hp, ch_data_raw)
    if config.APPLY_LP_FILTER:
        b_lp, a_lp = signal.butter(config.LP_FILTER_ORDER, config.LP_CUTOFF_HZ / nyquist, btype="low")
        sig_filt   = signal.filtfilt(b_lp, a_lp, sig_hp)
    else:
        sig_filt = sig_hp

    # R-peaks and IBIs on full filtered session.
    r_peaks_session              = detect_r_peaks(sig_filt, sfreq, config)
    ibi_times_s_session, ibi_ms_session = compute_ibi(r_peaks_session, sfreq)

    if config.VERBOSE:
        mean_hr = float(np.nanmean(60000.0 / ibi_ms_session)) if len(ibi_ms_session) > 0 else 0.0
        print(f"    {len(r_peaks_session)} R-peaks detected | mean HR: {mean_hr:.1f} bpm")

    # Pass 2: re-extract D110 events on cropped recording — indices are session-relative.
    events_df     = get_events_from_eeg(raw)
    startle_samps = events_df[events_df["Channel"] == f"D{config.TRIGGER_STARTLE}"]["Sample"].values
    print(f"    {len(startle_samps)} startle events found.")

    # Sequential alignment: Nth trigger ↔ Nth has_sound CSV row.
    n_use         = min(len(startle_samps), len(ratings_df))
    startle_samps = startle_samps[:n_use]
    ratings_df    = ratings_df.iloc[:n_use].reset_index(drop=True)

    wide_n        = int((config.WIDE_TMAX - config.WIDE_TMIN) * sfreq)
    times_snippet = np.linspace(config.WIDE_TMIN, config.WIDE_TMAX, wide_n, endpoint=False)
    baseline_mask = (times_snippet >= config.BASELINE_TMIN) & (times_snippet < config.BASELINE_TMAX)

    # 2 s IBI margin beyond epoch edges prevents boundary-clamping artifacts
    # when interpolating HR at the edges of the epoch window.
    IBI_MARGIN_S = 2.0

    results = []
    for i, s_idx in enumerate(startle_samps):
        t0 = int(s_idx + config.WIDE_TMIN * sfreq)
        t1 = t0 + wide_n
        if t0 < 0 or t1 > len(sig_filt):
            continue

        ecg_raw_snippet = ch_data_raw[t0:t1]
        ecg_snippet     = sig_filt[t0:t1]

        # R-peaks in this epoch, converted to epoch-relative indices (0 = t0).
        mask_peaks    = (r_peaks_session >= t0) & (r_peaks_session < t1)
        r_peaks_epoch = r_peaks_session[mask_peaks] - t0

        # IBI data for this trial in trigger-relative time.
        trigger_time_s    = s_idx / sfreq
        ibi_times_trigger = ibi_times_s_session - trigger_time_s
        ibi_select_mask   = (
            (ibi_times_trigger >= config.WIDE_TMIN - IBI_MARGIN_S) &
            (ibi_times_trigger <= config.WIDE_TMAX + IBI_MARGIN_S)
        )
        ibi_times_s = ibi_times_trigger[ibi_select_mask]
        ibi_ms      = ibi_ms_session[ibi_select_mask]

        # Baseline mean HR — cached at process time.
        if len(ibi_times_s) >= 1:
            target_baseline    = times_snippet[baseline_mask]
            hr_baseline_interp = interpolate_hr_timeseries(ibi_times_s, ibi_ms, target_baseline)
            baseline_mean_bpm  = float(np.nanmean(hr_baseline_interp))
        else:
            baseline_mean_bpm = np.nan

        # CSV metadata — mirrors emg_raw_potentiation.py L244-250 exactly.
        row     = ratings_df.iloc[i]
        img_col = ("image"      if "image"      in row.index else
                   "image_name" if "image_name" in row.index else
                   "image_type")
        img_type_val = ""
        for col in ["image_type", "file_name", "image", "image_name"]:
            if col in row.index and not pd.isna(row[col]):
                img_type_val = str(row[col])
                break

        results.append({
            # ── MFF-derived signals (cached) ──────────────────────────────────
            "ecg_raw_snippet":  ecg_raw_snippet,
            "ecg_snippet":      ecg_snippet,
            "times_snippet":    times_snippet,
            "r_peaks_epoch":    r_peaks_epoch,
            "ibi_times_s":      ibi_times_s,
            "ibi_ms":           ibi_ms,
            "baseline_mean":    baseline_mean_bpm,
            "sfreq":            sfreq,
            "channel":          channel,
            "psd_freqs":        psd_freqs,
            "psd_values":       psd_values,
            # ── CSV metadata (cached) ─────────────────────────────────────────
            "subjective_label": int(row["subjective_label"]),
            "image_type":       img_type_val,
            "label":            int(row["subjective_label"]),
            "valence":          float(row["valenceRating"]),
            "arousal":          float(row["arousalRating"]),
            "image_detail":     str(row.get(img_col, "?")),
            "csv_row":          row.to_dict(),
            # ── filled in by apply_analysis_params(), never persisted ─────────
            "hr_timecourse":    None,
            "times_hr":         None,
            "baseline_hr":      None,
            "hr_score":         None,
            "rejected":         None,
        })

    return results


# ── Analysis parameters (fast — no MFF access) ────────────────────────────────

def apply_analysis_params(subj_data_dict, config):
    """
    Recompute rejection, HR score, and analysis-window timecourse for every trial.
    Called unconditionally on every run — mirrors emg_raw_potentiation.py apply_analysis_params.
    Results are never written to the pickle.

    Two-pass structure (mirrors EMG):
      Pass 1: interpolate HR onto uniform grid; compute baseline_hr per trial;
              collect session HR statistics (median + std).
      Pass 2: label resolution, rejection, hr_score, trim to analysis window.

    Rejection criteria:
      baseline_hr outside [HR_MIN_BPM, HR_MAX_BPM]   — absolute physiological gate
      baseline_hr z-score > HR_Z_SCORE_THRESHOLD      — session-level dynamic gate
      r_peaks_epoch count < MIN_RPEAKS_EPOCH           — too few beats for interpolation
      baseline_hr is NaN                               — insufficient IBI data in baseline
    """
    wide_n       = int((config.WIDE_TMAX - config.WIDE_TMIN) * config.HR_INTERP_FS)
    target_times = np.linspace(config.WIDE_TMIN, config.WIDE_TMAX, wide_n, endpoint=False)

    baseline_mask = (target_times >= config.BASELINE_TMIN) & (target_times <  config.BASELINE_TMAX)
    score_mask    = (target_times >= config.SCORE_TMIN)    & (target_times <= config.SCORE_TMAX)
    anal_mask     = (target_times >= config.ANAL_TMIN)     & (target_times <  config.ANAL_TMAX)

    for subj, sessions in subj_data_dict.items():
        for sess_key, trials in sessions.items():
            if not trials:
                continue

            # Pass 1: interpolate HR, compute baseline_hr, collect session stats.
            baseline_hr_list = []
            for t in trials:
                hr_full          = interpolate_hr_timeseries(t["ibi_times_s"], t["ibi_ms"], target_times)
                t["_hr_full"]    = hr_full
                baseline_vals    = hr_full[baseline_mask]
                t["baseline_hr"] = float(np.nanmean(baseline_vals)) if baseline_mask.any() else np.nan
                baseline_hr_list.append(t["baseline_hr"])

            session_hr_median = np.nanmedian(baseline_hr_list)
            session_hr_std    = np.nanstd(baseline_hr_list)

            # Pass 2: label resolution, rejection, score, trim.
            for t in trials:
                if "subjective_label" not in t:
                    t["subjective_label"] = t.get("label", 2)

                # Label resolution — identical logic to EMG L316-326.
                if config.USE_SUBJECTIVE_TRIAL_TYPE:
                    t["label"] = t["subjective_label"]
                else:
                    img_type = str(t.get("image_detail", t.get("image_type", ""))).lower().strip()
                    if "negative" in img_type or img_type == "1":
                        t["label"] = 1
                    elif "neutral" in img_type or img_type == "2":
                        t["label"] = 2
                    else:
                        t["label"] = t["subjective_label"]

                bhr     = t["baseline_hr"]
                hr_full = t.pop("_hr_full")

                hr_z = 0.0
                if session_hr_std > 0 and not np.isnan(bhr):
                    hr_z = (bhr - session_hr_median) / session_hr_std

                t["rejected"] = bool(
                    np.isnan(bhr)
                    or bhr < config.HR_MIN_BPM
                    or bhr > config.HR_MAX_BPM
                    or hr_z > config.HR_Z_SCORE_THRESHOLD
                    or len(t["r_peaks_epoch"]) < config.MIN_RPEAKS_EPOCH
                )

                # HR score: peak HR in score window minus baseline.
                if t["rejected"]:
                    t["hr_score"] = np.nan
                else:
                    t["hr_score"] = float(np.nanmax(hr_full[score_mask]) - bhr)

                t["hr_timecourse"] = hr_full[anal_mask]
                t["times_hr"]      = target_times[anal_mask]
