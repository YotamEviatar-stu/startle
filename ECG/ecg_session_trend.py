#!/usr/bin/env python3
"""
ECG Session Trend
=================
Loads both the evening and morning sessions for one subject, computes
beat-to-beat heart rate from channel E256, and plots both trends on a
shared figure with each startle event (D110) marked as a vertical line.

Usage:
    .venv/bin/python scripts/ecg_session_trend.py          # default subject
    .venv/bin/python scripts/ecg_session_trend.py AB22
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import mne
from scipy.signal import butter, filtfilt, find_peaks

# ── Configuration ──────────────────────────────────────────────────────────────

RAW_DATA_DIR       = r"/Volumes/My Passport/startle_raw"
DEFAULT_SUBJECT    = "DA01"
ECG_CHANNEL        = "E256"        # vertex electrode — strong cardiac signal on EGI hardware

TRIGGER_START      = 101           # D101 = session start
TRIGGER_END        = 124           # D124 = session end
TRIGGER_STARTLE    = 110           # D110 = startle onset

HP_CUTOFF_HZ       = 0.5
LP_CUTOFF_HZ       = 40.0
FILTER_ORDER       = 3

MIN_RR_S           = 0.5          # 0.5 s → 120 bpm ceiling; avoids T-wave false positives
PROMINENCE_FRAC    = 0.10         # 10% of peak-to-peak amplitude

HR_INTERP_FS       = 4.0          # Hz — uniform HR timecourse output rate

SESSIONS = [
    ("eve", "Evening"),
    ("mor", "Morning"),
]

EVENT_COLOR   = "steelblue"
SESSION_COLOR = {"eve": "crimson", "mor": "darkorange"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def find_mff(eeg_folder, keyword):
    for f in sorted(os.listdir(eeg_folder)):
        if f"_{keyword}_" in f.lower() and f.lower().endswith(".mff"):
            return os.path.join(eeg_folder, f)
    return None


def get_din_events(raw):
    """Return dict {channel_name: sorted array of sample indices} for all D-channels."""
    din_chs = [ch for ch in raw.ch_names if ch.startswith("D")]
    if not din_chs:
        return {}
    raw_din = raw.copy().pick(din_chs)
    events = {}
    for idx, name in enumerate(raw_din.ch_names):
        data = raw_din.get_data(picks=[idx])[0]
        thr  = data.max() * 0.9
        samps = np.where(data > thr)[0]
        if len(samps):
            # collapse contiguous runs to first sample only
            gaps = np.where(np.diff(samps) > 1)[0]
            starts = np.concatenate([[samps[0]], samps[gaps + 1]])
            events[name] = starts
    return events


def detect_peaks_e256(data, sfreq):
    """Bandpass-filter E256 and detect R-peaks with scipy."""
    b, a = butter(FILTER_ORDER, [HP_CUTOFF_HZ, LP_CUTOFF_HZ], btype="bandpass", fs=sfreq)
    filt = filtfilt(b, a, data)

    ptp        = filt.max() - filt.min()
    prominence = PROMINENCE_FRAC * ptp
    min_dist   = int(MIN_RR_S * sfreq)

    pos, _ = find_peaks( filt, distance=min_dist, prominence=prominence)
    neg, _ = find_peaks(-filt, distance=min_dist, prominence=prominence)
    return neg if len(neg) > len(pos) else pos


def compute_hr_timecourse(r_peaks, sfreq, duration_s):
    """Interpolate beat-to-beat HR onto a uniform grid at HR_INTERP_FS."""
    if len(r_peaks) < 2:
        n = int(duration_s * HR_INTERP_FS)
        return np.linspace(0, duration_s, n), np.full(n, np.nan)

    peak_times  = r_peaks / sfreq
    rr          = np.diff(peak_times)
    hr_bpm      = 60.0 / rr
    hr_times    = peak_times[:-1] + rr / 2.0   # midpoint of each RR interval

    target = np.arange(0, duration_s, 1.0 / HR_INTERP_FS)
    hr_interp = np.interp(target, hr_times, hr_bpm)
    return target, hr_interp


# ── Session processor ──────────────────────────────────────────────────────────

def process_session(subject, session_kw, eeg_folder):
    path = find_mff(eeg_folder, session_kw)
    if path is None:
        print(f"  [{session_kw}] No MFF file found — skipping.")
        return None

    print(f"  [{session_kw}] Loading {os.path.basename(path)} ...")
    raw   = mne.io.read_raw_egi(path, preload=True, verbose=False, events_as_annotations=True)
    sfreq = raw.info["sfreq"]

    if ECG_CHANNEL not in raw.ch_names:
        print(f"  [{session_kw}] Channel {ECG_CHANNEL} not found — skipping.")
        return None

    # Crop to task window
    din = get_din_events(raw)
    start_samps = din.get(f"D{TRIGGER_START}", np.array([]))
    end_samps   = din.get(f"D{TRIGGER_END}",   np.array([]))

    if len(start_samps) == 0 or len(end_samps) == 0:
        print(f"  [{session_kw}] Task boundary triggers not found — using full session.")
        t_start, t_end = 0.0, raw.times[-1]
    else:
        t_start = start_samps[0]  / sfreq
        t_end   = end_samps[-1]   / sfreq
        raw.crop(tmin=t_start, tmax=t_end)
        print(f"  [{session_kw}] Cropped to task: {t_start:.1f}s → {t_end:.1f}s  ({(t_end - t_start)/60:.1f} min)")

    # Re-extract DIN events on cropped recording (now session-relative)
    din_cropped    = get_din_events(raw)
    startle_samps  = din_cropped.get(f"D{TRIGGER_STARTLE}", np.array([]))
    startle_times  = startle_samps / sfreq
    print(f"  [{session_kw}] {len(startle_times)} startle events found.")

    # HR detection on E256
    ecg_data = raw.get_data(picks=[ECG_CHANNEL])[0]
    r_peaks  = detect_peaks_e256(ecg_data, sfreq)
    duration = raw.times[-1]
    t_hr, hr = compute_hr_timecourse(r_peaks, sfreq, duration)

    n_beats  = len(r_peaks)
    mean_hr  = float(np.nanmean(hr))
    print(f"  [{session_kw}] {n_beats} R-peaks detected | mean HR: {mean_hr:.1f} bpm")

    return {
        "t_hr":          t_hr,
        "hr":            hr,
        "startle_times": startle_times,
        "mean_hr":       mean_hr,
        "n_beats":       n_beats,
        "duration":      duration,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    subject    = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SUBJECT
    eeg_folder = os.path.join(RAW_DATA_DIR, subject, "EEG")

    if not os.path.isdir(eeg_folder):
        print(f"[!] EEG folder not found: {eeg_folder}")
        sys.exit(1)

    print(f"\nSubject: {subject}  |  Channel: {ECG_CHANNEL}")
    print("=" * 60)

    results = {}
    for kw, label in SESSIONS:
        results[kw] = process_session(subject, kw, eeg_folder)

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_panels = sum(1 for v in results.values() if v is not None)
    if n_panels == 0:
        print("[!] No sessions processed — nothing to plot.")
        sys.exit(1)

    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 4 * n_panels), sharex=False)
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0
    for kw, label in SESSIONS:
        res = results.get(kw)
        if res is None:
            continue

        ax    = axes[ax_idx]
        color = SESSION_COLOR[kw]

        ax.plot(res["t_hr"], res["hr"], color=color, linewidth=0.9,
                label=f"HR ({ECG_CHANNEL})")
        ax.axhline(res["mean_hr"], color=color, linewidth=1, linestyle="--",
                   alpha=0.6, label=f"Mean {res['mean_hr']:.1f} bpm")

        # Mark startle events
        for i, st in enumerate(res["startle_times"]):
            ax.axvline(st, color=EVENT_COLOR, linewidth=0.8, alpha=0.7,
                       label="Startle" if i == 0 else None)

        ax.set_ylabel("Heart Rate (bpm)")
        ax.set_xlabel("Time (s, task-relative)")
        ax.set_title(f"{subject} — {label} session  |  {res['n_beats']} beats")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)

        # Sensible y-limits: median ± 40 bpm, at least 40–120
        med = np.nanmedian(res["hr"])
        ax.set_ylim(max(20, med - 40), min(220, med + 40))

        ax_idx += 1

    fig.suptitle(f"Heart Rate Trend — {subject}  (channel {ECG_CHANNEL})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
