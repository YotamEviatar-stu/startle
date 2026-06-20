# resp_processor.py — stateless processing for the RESP pipeline.
#
# Two-layer cache design:
#   Layer 1 (cached):  raw wide epochs + NeuroKit2-derived signals extracted
#                      from MFF — process_session()
#   Layer 2 (live):    baseline, rejection, score selection — apply_analysis_params()
#
# NeuroKit2 runs on the full session signal (not per epoch) so that RSP_Rate
# and RSP_Amplitude are interpolated across the whole recording and every sample
# has a valid value.  Epochs are then cut from the derived full-session arrays.
#
# Changing score_signal, baseline window, score window, or rejection thresholds
# in resp_config.py takes effect immediately with no MFF reload.
# Changing RESP_CHANNELS, WIDE_TMIN, or WIDE_TMAX requires FORCE_PREPROCESSING.

import os
import re
import warnings
import numpy as np
import pandas as pd
import mne
import neurokit2 as nk
from scipy import signal as sp_signal

from . import resp_config as cfg

warnings.filterwarnings("ignore", category=nk.misc.NeuroKitWarning
                        if hasattr(nk, "misc") else UserWarning)


# ── File discovery ─────────────────────────────────────────────────────────────

def find_startle_output_folder(subject_path):
    for name in os.listdir(subject_path):
        if name.lower() == "startle output" and os.path.isdir(os.path.join(subject_path, name)):
            return os.path.join(subject_path, name)
    return None


def find_mff_file(eeg_folder, keyword):
    for name in sorted(os.listdir(eeg_folder)):
        if f"_{keyword}_" in name.lower() and name.endswith(".mff"):
            return os.path.join(eeg_folder, name)
    return None


def find_csv_by_suffix(startle_folder, suffix):
    for name in os.listdir(startle_folder):
        if name.endswith(".csv") and "demo" not in name.lower():
            if re.search(rf"{suffix}\.csv$", name):
                return os.path.join(startle_folder, name)
    return None


# ── DIN event extraction ───────────────────────────────────────────────────────

def get_events_from_eeg(raw_eeg):
    din_names = [ch for ch in raw_eeg.ch_names if ch.startswith("D")]
    raw_din = raw_eeg.copy().pick(din_names)
    event_data = []
    for ch_idx, ch_name in enumerate(raw_din.ch_names):
        data = raw_din.get_data(picks=[ch_idx])[0]
        threshold = np.max(data) * 0.9
        for sample in np.where(data > threshold)[0]:
            event_data.append({"Channel": ch_name, "Sample": int(sample)})
    if not event_data:
        return pd.DataFrame(columns=["Channel", "Sample"])
    return pd.DataFrame(event_data).sort_values("Sample").reset_index(drop=True)


# ── Ratings classification ─────────────────────────────────────────────────────

def load_and_classify_ratings(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df[df["has_sound"] == True].reset_index(drop=True)

    def categorize(row):
        if row["arousalRating"] >= 7 or row["valenceRating"] <= 3:
            return 1  # Negative
        return 2      # Neutral

    df["subjective_label"] = df.apply(categorize, axis=1)
    return df


# ── Layer 1: MFF extraction (cached) ──────────────────────────────────────────

def _run_nk(signal_1d, sfreq):
    """
    Run nk.rsp_process on a full-session signal.  Returns dict of derived arrays
    (RSP_Clean, RSP_Amplitude, RSP_Rate, RSP_Phase), all same length as input.
    Falls back to NaN arrays on failure.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, _ = nk.rsp_process(signal_1d, sampling_rate=int(sfreq))
        return {
            "clean":     out["RSP_Clean"].to_numpy(),
            "amplitude": out["RSP_Amplitude"].to_numpy(),
            "rate":      out["RSP_Rate"].to_numpy(),
            "phase":     out["RSP_Phase"].to_numpy(),
        }
    except Exception as e:
        print(f"      [!] nk.rsp_process failed: {e} — using NaN fallback")
        nan = np.full(len(signal_1d), np.nan)
        return {"clean": signal_1d.copy(), "amplitude": nan, "rate": nan, "phase": nan}


def process_session(mff_path, ratings_df):
    """
    Load one session MFF, crop to task, run NeuroKit2 on each channel's full
    session signal, then cut wide epochs from raw + derived arrays.

    Cached per trial:
      epoch_raw_wide       — unfiltered signal (V)
      epoch_clean_wide     — nk RSP_Clean
      epoch_amplitude_wide — nk RSP_Amplitude (tidal volume proxy)
      epoch_rate_wide      — nk RSP_Rate (breaths/min)
      epoch_phase_wide     — nk RSP_Phase (0=exhale, 1=inhale)

    Not cached (filled by apply_analysis_params every run):
      epoch_proc_wide, epoch_proc_anal, baseline_mean,
      rejected, score, scores, times_anal, phase_at_onset
    """
    print(f"    Loading {os.path.basename(mff_path)} ...")
    raw = mne.io.read_raw_egi(mff_path, preload=True, verbose=False,
                              events_as_annotations=False)
    sfreq = raw.info["sfreq"]

    # Crop to task
    events_df   = get_events_from_eeg(raw)
    start_samps = events_df[events_df["Channel"] == f"D{cfg.TRIGGER_SESSION_START}"]["Sample"].values
    end_samps   = events_df[events_df["Channel"] == f"D{cfg.TRIGGER_SESSION_END}"]["Sample"].values
    if not len(start_samps) or not len(end_samps):
        print("    [!] Task boundaries not found — skipping")
        return None
    raw.crop(tmin=start_samps[0] / sfreq, tmax=end_samps[-1] / sfreq)

    # Pick available channels
    available = [ch for ch in cfg.RESP_CHANNELS if ch in raw.ch_names]
    missing   = [ch for ch in cfg.RESP_CHANNELS if ch not in raw.ch_names]
    if not available:
        print("    [!] No RESP channels found in file — skipping")
        return None
    if missing:
        print(f"    Channels not in file (skipped): {missing}")

    ch_signals = {ch: raw.copy().pick([ch]).get_data()[0] for ch in available}
    min_len    = min(len(v) for v in ch_signals.values())

    # Run NeuroKit2 on the full session signal per channel
    print(f"    Running nk.rsp_process on {available} ...")
    nk_derived = {ch: _run_nk(ch_signals[ch], sfreq) for ch in available}

    # Re-extract events post-crop
    events_df     = get_events_from_eeg(raw)
    startle_samps = events_df[events_df["Channel"] == f"D{cfg.TRIGGER_STARTLE}"]["Sample"].values

    n_use         = min(len(startle_samps), len(ratings_df))
    startle_samps = startle_samps[:n_use]
    ratings_df    = ratings_df.iloc[:n_use].reset_index(drop=True)

    wide_n     = int((cfg.WIDE_TMAX - cfg.WIDE_TMIN) * sfreq)
    times_wide = np.linspace(cfg.WIDE_TMIN, cfg.WIDE_TMAX, wide_n, endpoint=False)

    results = []
    for i, s_idx in enumerate(startle_samps):
        t0 = int(s_idx + cfg.WIDE_TMIN * sfreq)
        t1 = t0 + wide_n
        if t0 < 0 or t1 > min_len:
            continue

        row = ratings_df.iloc[i]
        img_type_val = ""
        for col in ["image_type", "file_name", "image", "image_name"]:
            if col in row.index and not pd.isna(row[col]):
                img_type_val = str(row[col])
                break
        img_col = next(
            (c for c in ["image", "image_name", "image_type"] if c in row.index),
            "image_type",
        )

        results.append({
            # ── cached: raw + nk-derived wide epochs ──
            "epoch_raw_wide":       {ch: ch_signals[ch][t0:t1].copy()              for ch in available},
            "epoch_clean_wide":     {ch: nk_derived[ch]["clean"][t0:t1].copy()     for ch in available},
            "epoch_amplitude_wide": {ch: nk_derived[ch]["amplitude"][t0:t1].copy() for ch in available},
            "epoch_rate_wide":      {ch: nk_derived[ch]["rate"][t0:t1].copy()      for ch in available},
            "epoch_phase_wide":     {ch: nk_derived[ch]["phase"][t0:t1].copy()     for ch in available},
            "times_wide":           times_wide,
            "sfreq":                sfreq,
            "channels":             available,
            # ── cached: metadata ──
            "subjective_label":     int(row["subjective_label"]),
            "image_type":           img_type_val,
            "label":                int(row["subjective_label"]),
            "valence":              float(row["valenceRating"]),
            "arousal":              float(row["arousalRating"]),
            "image_detail":         str(row.get(img_col, "?")),
            # ── filled by apply_analysis_params — never persisted ──
            "epoch_proc_wide":  None,   # set to epoch_clean_wide for visualisation
            "epoch_proc_anal":  None,
            "baseline_mean":    None,   # {ch: float} of score_signal
            "rejected":         None,
            "score":            None,
            "scores":           None,
            "times_anal":       None,
            "phase_at_onset":   None,   # RSP_Phase value at t=0 (0=exhale, 1=inhale)
        })

    return results


# ── Layer 2: analysis params (always live, no MFF access) ─────────────────────

def apply_analysis_params(subj_data_dict):
    """
    Apply current resp_config parameters to cached epochs.

    For each trial:
      - epoch_proc_wide  = RSP_Clean (for visualisation, always)
      - baseline_mean    = mean of score_signal in baseline window
      - rejection        = z-score of RSP_Clean baseline std (consistent, artifact-sensitive)
      - score            = nanmean(score_signal in score_win) - baseline_mean
      - phase_at_onset   = RSP_Phase at t=0 sample (0=exhale, 1=inhale, nan=unknown)
    """
    _signal_key = {
        "clean":     "epoch_clean_wide",
        "amplitude": "epoch_amplitude_wide",
        "rate":      "epoch_rate_wide",
    }

    for subj, sessions in subj_data_dict.items():
        p = {**cfg.DEFAULT_PARAMS, **cfg.SUBJECT_PARAMS.get(subj, {})}
        sig_key = _signal_key.get(p.get("score_signal", "amplitude"), "epoch_amplitude_wide")

        for sess_key, trials in sessions.items():
            if not trials:
                continue

            # Check whether nk-derived signals are present (old cache compat)
            has_nk = "epoch_clean_wide" in trials[0] and trials[0]["epoch_clean_wide"] is not None

            # Pass 1 — set proc epochs and collect RSP_Clean baseline stds for z-score
            bl_clean_stds: dict[str, list] = {ch: [] for ch in trials[0]["channels"]}

            for t in trials:
                tw = t["times_wide"]

                if has_nk:
                    t["epoch_proc_wide"] = t["epoch_clean_wide"]
                else:
                    # Fallback: manual bandpass (old cache without nk signals)
                    sfreq   = t["sfreq"]
                    nyquist = sfreq / 2.0
                    proc = {}
                    for ch in t["channels"]:
                        sig = t["epoch_raw_wide"][ch].copy()
                        if p.get("hp_cutoff") and 0 < p["hp_cutoff"] < nyquist:
                            b, a = sp_signal.butter(p["filter_order"], p["hp_cutoff"] / nyquist, btype="high")
                            sig  = sp_signal.filtfilt(b, a, sig)
                        if p.get("lp_cutoff") and 0 < p["lp_cutoff"] < nyquist:
                            b, a = sp_signal.butter(p["filter_order"], p["lp_cutoff"] / nyquist, btype="low")
                            sig  = sp_signal.filtfilt(b, a, sig)
                        proc[ch] = sig
                    t["epoch_proc_wide"] = proc
                    # No nk signals — copy clean into all derived fields
                    t["epoch_clean_wide"]     = proc
                    t["epoch_amplitude_wide"] = {ch: np.full_like(proc[ch], np.nan) for ch in t["channels"]}
                    t["epoch_rate_wide"]      = {ch: np.full_like(proc[ch], np.nan) for ch in t["channels"]}
                    t["epoch_phase_wide"]     = {ch: np.full_like(proc[ch], np.nan) for ch in t["channels"]}

                # Collect RSP_Clean baseline std for rejection
                bl_mask = (tw >= p["baseline_tmin"]) & (tw < p["baseline_tmax"])
                for ch in t["channels"]:
                    clean_seg = t["epoch_clean_wide"][ch][bl_mask]
                    bl_clean_stds[ch].append(float(np.nanstd(clean_seg)))

            # Session-level stats of baseline stds (for z-score rejection)
            sess_med = {ch: float(np.nanmedian(v)) for ch, v in bl_clean_stds.items() if v}
            sess_std = {ch: float(np.nanstd(v))    for ch, v in bl_clean_stds.items() if v}

            primary = cfg.PRIMARY_CHANNEL

            # Pass 2 — label, rejection, baseline, score, trimmed epoch
            for t in trials:
                tw = t["times_wide"]

                # Active label
                if cfg.USE_SUBJECTIVE_TRIAL_TYPE:
                    t["label"] = t["subjective_label"]
                else:
                    img = str(t.get("image_type", "")).lower().strip()
                    if "negative" in img or img == "1":
                        t["label"] = 1
                    elif "neutral" in img or img == "2":
                        t["label"] = 2
                    else:
                        t["label"] = t["subjective_label"]

                bl_mask    = (tw >= p["baseline_tmin"]) & (tw < p["baseline_tmax"])
                score_mask = (tw >= p["score_tmin"])    & (tw <= p["score_tmax"])
                anal_mask  = (tw >= p["anal_tmin"])     & (tw <  p["anal_tmax"])

                # Rejection — always on RSP_Clean std for consistency
                ref_ch = primary if primary in t["channels"] else t["channels"][0]
                bl_std = float(np.nanstd(t["epoch_clean_wide"][ref_ch][bl_mask]))
                med    = sess_med.get(ref_ch, 0.0)
                ssd    = sess_std.get(ref_ch, 1.0)
                z      = (bl_std - med) / ssd if ssd > 0 else 0.0

                is_z_outlier  = abs(z) > p["z_score_threshold"]
                is_abs_outlier = (
                    p.get("absolute_max") is not None
                    and any(
                        np.nanmax(np.abs(t["epoch_clean_wide"][ch][bl_mask])) > p["absolute_max"]
                        for ch in t["channels"]
                    )
                )
                t["rejected"] = bool(is_z_outlier or is_abs_outlier)

                # Baseline mean of score_signal per channel
                score_epochs = t.get(sig_key) or t["epoch_clean_wide"]
                t["baseline_mean"] = {
                    ch: float(np.nanmean(score_epochs[ch][bl_mask])) for ch in t["channels"]
                }

                # Score per channel
                scores = {}
                for ch in t["channels"]:
                    if t["rejected"]:
                        scores[ch] = np.nan
                    else:
                        scores[ch] = float(
                            np.nanmean(score_epochs[ch][score_mask])
                            - t["baseline_mean"][ch]
                        )
                t["scores"] = scores
                t["score"]  = scores.get(primary, scores.get(t["channels"][0], np.nan))

                # Trimmed epoch for plots — always RSP_Clean
                t["epoch_proc_anal"] = {
                    ch: t["epoch_clean_wide"][ch][anal_mask] for ch in t["channels"]
                }
                t["times_anal"] = tw[anal_mask]

                # Phase at startle onset (t=0)
                onset_idx = np.argmin(np.abs(tw))
                phase_arr = t["epoch_phase_wide"].get(ref_ch)
                t["phase_at_onset"] = (
                    float(phase_arr[onset_idx])
                    if phase_arr is not None and not np.isnan(phase_arr[onset_idx])
                    else np.nan
                )
