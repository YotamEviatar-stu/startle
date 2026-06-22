# airflow_processor.py — stateless processing for the Airflow gasp-ratio pipeline.
#
# Two-layer cache design (mirrors HR pipeline):
#   process_session()       → MFF-derived data; written to pickle cache
#   apply_analysis_params() → baseline, score, rejection; recomputed every run

import os
import re
import warnings

import mne
import neurokit2 as nk
import numpy as np
import pandas as pd

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


# ── Gasp scoring ───────────────────────────────────────────────────────────────

def score_respiration_gasp(trigger_in_epoch, epoch_clean, ep_peaks, ep_troughs, sfreq,
                            tmin_baseline, tmax_baseline,
                            tmin_response, tmax_response):
    """
    Gasp ratio = abs(response amplitude / baseline amplitude).

    All indices are relative to the start of epoch_clean.
    trigger_in_epoch = sample index within the epoch where t=0 (the startle).
    Amplitude = peak-to-trough range; falls back to max-min if no peaks/troughs.
    Returns 1.0 on any failure.
    """
    try:
        n          = len(epoch_clean)
        base_start = max(0, trigger_in_epoch + int(tmin_baseline * sfreq))
        base_end   = min(n, trigger_in_epoch + int(tmax_baseline * sfreq))
        resp_start = max(0, trigger_in_epoch + int(tmin_response * sfreq))
        resp_end   = min(n, trigger_in_epoch + int(tmax_response * sfreq))

        bl_peaks   = [p for p in ep_peaks   if base_start <= p <= base_end]
        bl_troughs = [t for t in ep_troughs if base_start <= t <= base_end]
        if bl_peaks and bl_troughs:
            baseline_amp = (np.mean(epoch_clean[bl_peaks])
                            - np.mean(epoch_clean[bl_troughs]))
        else:
            w = epoch_clean[base_start:base_end]
            baseline_amp = float(np.max(w) - np.min(w)) if w.size else 0.0

        rs_peaks   = [p for p in ep_peaks   if resp_start <= p <= resp_end]
        rs_troughs = [t for t in ep_troughs if resp_start <= t <= resp_end]
        if rs_peaks and rs_troughs:
            gasp_amp = epoch_clean[rs_peaks[0]] - epoch_clean[rs_troughs[0]]
        else:
            w = epoch_clean[resp_start:resp_end]
            gasp_amp = float(np.max(w) - np.min(w)) if w.size else 0.0

        return abs(gasp_amp / baseline_amp) if baseline_amp != 0 else 1.0

    except Exception:
        return 1.0


# ── Layer 1: MFF extraction (cache boundary) ──────────────────────────────────

def process_session(mff_path, ratings_df, channel=None, config=None):
    """
    Load one MFF session, run NeuroKit2 on the Airflow channel, extract
    per-trial wide epochs with epoch-relative peak/trough indices.

    CACHE BOUNDARY: output is written to pickle. Baseline, score, and
    rejection are NOT computed here — see apply_analysis_params().

    Returns (trials, trace) or (None, None) on failure.
    trace = {'signal', 'sfreq', 'startle_samps_sec'} for session timecourse.
    """
    if config is None:
        config = cfg
    if channel is None:
        channel = config.AIRFLOW_CHANNEL

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
    if not len(start_samps) or not len(end_samps):
        print("    [!] Task boundaries not found — skipping")
        return None, None

    raw.crop(tmin=start_samps[0] / sfreq, tmax=end_samps[-1] / sfreq)

    if channel not in raw.ch_names:
        print(f"    [!] Channel {channel!r} not found — skipping")
        return None, None

    raw_channel = raw.copy().pick([channel]).get_data()[0]

    try:
        rsp_signals, _ = nk.rsp_process(raw_channel, sampling_rate=int(sfreq))
    except Exception as exc:
        print(f"    [!] nk.rsp_process() failed: {exc}")
        return None, None

    if "RSP_Clean" not in rsp_signals.columns:
        print("    [!] NeuroKit2 output missing RSP_Clean")
        return None, None

    clean_signal = rsp_signals["RSP_Clean"].to_numpy()
    zeros        = np.zeros(len(clean_signal))
    peaks   = np.flatnonzero(rsp_signals.get("RSP_Peaks",   pd.Series(zeros)).to_numpy())
    troughs = np.flatnonzero(rsp_signals.get("RSP_Troughs", pd.Series(zeros)).to_numpy())

    events_df     = get_events_from_eeg(raw)
    startle_samps = events_df[events_df["Channel"] == f"D{config.TRIGGER_STARTLE}"]["Sample"].values
    print(f"    {len(startle_samps)} startle events found.")

    n_use         = min(len(startle_samps), len(ratings_df))
    startle_samps = startle_samps[:n_use]
    ratings_df    = ratings_df.iloc[:n_use].reset_index(drop=True)

    wide_n     = int((config.WIDE_TMAX - config.WIDE_TMIN) * sfreq)
    times_wide = np.linspace(config.WIDE_TMIN, config.WIDE_TMAX, wide_n, endpoint=False)

    results = []
    for i, s_idx in enumerate(startle_samps):
        t0 = int(s_idx + config.WIDE_TMIN * sfreq)
        t1 = t0 + wide_n
        if t0 < 0 or t1 > len(clean_signal):
            continue

        ep_raw   = raw_channel[t0:t1]
        ep_clean = clean_signal[t0:t1]

        ep_peaks   = np.array([p - t0 for p in peaks   if t0 <= p < t1])
        ep_troughs = np.array([t - t0 for t in troughs if t0 <= t < t1])

        row     = ratings_df.iloc[i]
        img_col = next((c for c in ["image", "image_name", "image_type"]
                        if c in row.index), None)

        results.append({
            "epoch_raw_wide":   ep_raw.astype(np.float32),
            "epoch_clean_wide": ep_clean.astype(np.float32),
            "ep_peaks":         ep_peaks,
            "ep_troughs":       ep_troughs,
            "times_wide":       times_wide,
            "sfreq":            float(sfreq),
            "channel":          channel,
            "subjective_label": int(row["subjective_label"]),
            "label":            int(row["subjective_label"]),
            "valence":          float(row["valenceRating"]),
            "arousal":          float(row["arousalRating"]),
            "image_detail":     str(row.get(img_col, "?")) if img_col else "?",
            # Filled by apply_analysis_params — not persisted in pickle
            "baseline_mean":    None,
            "score":            None,
            "rejected":         None,
            "epoch_anal":       None,
            "times_anal":       None,
        })

    trace = {
        "signal":            clean_signal.astype(np.float32),
        "sfreq":             float(sfreq),
        "startle_samps_sec": startle_samps / sfreq,
    }
    return results, trace


# ── Layer 2: analysis parameters (fast, no MFF access) ────────────────────────

def apply_analysis_params(subj_data_dict, config):
    """
    Recompute baseline, gasp score, rejection, and trimmed epoch for every trial.
    Called unconditionally on every run regardless of cache state.

    Changing BASELINE_TMIN/TMAX, RESPONSE_TMIN/TMAX, rejection thresholds,
    or USE_SUBJECTIVE_TRIAL_TYPE takes effect immediately without reloading MFF.

    Rejection (two gates):
      1. Z-score on baseline std: if the baseline window is unusually variable
         relative to other trials in the session, the signal was likely unstable.
         Controlled by AIRFLOW_Z_SCORE_THRESHOLD (None = disabled).
      2. Score ceiling: if gasp_ratio exceeds AIRFLOW_SCORE_MAX the trial is
         almost certainly an artifact. Controlled by AIRFLOW_SCORE_MAX (None = disabled).
    """
    for subj, sessions in subj_data_dict.items():
        for sess_key, trials in sessions.items():
            if not trials:
                continue

            # Pass 1 — label resolution, baseline, collect session stds
            bl_stds   = []
            pre_masks = []
            for t in trials:
                if "subjective_label" not in t:
                    t["subjective_label"] = t.get("label", 2)

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

                tw = t["times_wide"]
                ep = t["epoch_clean_wide"]
                pre_mask = (tw >= config.BASELINE_TMIN) & (tw < config.BASELINE_TMAX)
                pre_masks.append(pre_mask)

                if np.any(pre_mask):
                    bm = float(np.mean(ep[pre_mask]))
                    bs = float(np.std(ep[pre_mask]))
                else:
                    bm = bs = np.nan
                t["baseline_mean"] = bm
                bl_stds.append(bs)

            sess_std_median = float(np.nanmedian(bl_stds))
            sess_std_std    = float(np.nanstd(bl_stds))

            # Pass 2 — rejection, score, trim
            for t, pre_mask in zip(trials, pre_masks):
                tw      = t["times_wide"]
                ep      = t["epoch_clean_wide"]
                bm      = t["baseline_mean"]
                peaks   = t["ep_peaks"]
                troughs = t["ep_troughs"]
                sfreq   = t["sfreq"]
                bs = float(np.std(ep[pre_mask])) if np.any(pre_mask) else np.nan

                z_reject = False
                threshold = getattr(config, "AIRFLOW_Z_SCORE_THRESHOLD", None)
                if threshold is not None and not np.isnan(bs) and sess_std_std > 0:
                    z = (bs - sess_std_median) / sess_std_std
                    z_reject = bool(z > threshold)

                t["rejected"] = bool(np.isnan(bm) or z_reject)

                anal_mask       = (tw >= config.ANAL_TMIN) & (tw < config.ANAL_TMAX)
                t["epoch_anal"] = ep[anal_mask]
                t["times_anal"] = tw[anal_mask]

                if t["rejected"]:
                    t["score"] = np.nan
                elif config.PERFORM_SCORING:
                    trigger_in_epoch = int(-config.WIDE_TMIN * sfreq)
                    t["score"] = score_respiration_gasp(
                        trigger_in_epoch, ep, peaks, troughs, sfreq,
                        config.BASELINE_TMIN, config.BASELINE_TMAX,
                        config.RESPONSE_TMIN, config.RESPONSE_TMAX,
                    )
                else:
                    t["score"] = np.nan

                score_max = getattr(config, "AIRFLOW_SCORE_MAX", None)
                if score_max is not None and not t["rejected"] and not np.isnan(t["score"]):
                    if t["score"] > score_max:
                        t["rejected"] = True
                        t["score"]    = np.nan
