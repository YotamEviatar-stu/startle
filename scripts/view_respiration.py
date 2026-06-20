#!/usr/bin/env python3
"""
Respiration Raw Viewer — interactive matplotlib
================================================
Channels: Airflow, Chest_belt, Abdomen_belt, Pleth (raw, normalized).

Event shading:
  Orange = trial with sound   (timing from D110 trigger)
  Grey   = silent control     (timing reconstructed from CSV abs_s + fixation_duration)
  Red dashed line = D110 startle onset (sound trials only)

Fixes applied vs naive approach:
  1. Pre-task D110 triggers (demo/practice before D101) are excluded.
  2. D110 triggers aligned only to has_sound=True rows.
  3. Silent trial timing reconstructed via:
       pic_onset_task = abs_s + fixation_duration - D101_script_time
     where D101_script_time is calibrated from the sound trials (mean ~1.19s, std 23ms).

Controls:
    ← →   scroll one full page
    [ ]   scroll half a page
    + −   zoom in / out
    slider   jump anywhere

Usage:
    .venv/bin/python scripts/view_respiration.py              # DA01 eve
    .venv/bin/python scripts/view_respiration.py AB22 mor
"""

import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider
import pandas as pd
import mne

# ── Config ─────────────────────────────────────────────────────────────────────
RAW_DATA_DIR    = r"/Volumes/My Passport/startle_raw"
DEFAULT_SUBJECT = "DA01"
DEFAULT_SESSION = "eve"

TRIGGER_START   = 101
TRIGGER_END     = 124
TRIGGER_STARTLE = 110

RESP_CHANNELS = ["Airflow", "Chest_belt", "Abdomen_belt", "Pleth"]
CSV_SUFFIX    = {"eve": "_1", "mor": "_2"}

WINDOW_SEC = 30.0

CH_COLORS      = ["#2980b9", "#27ae60", "#8e44ad", "#d35400"]
SOUND_COLOR    = "#e67e22"
NOSOUND_COLOR  = "#7f8c8d"
STARTLE_COLOR  = "#e74c3c"


# ── Helpers ────────────────────────────────────────────────────────────────────

def find_mff(eeg_folder, session_kw):
    for f in sorted(os.listdir(eeg_folder)):
        if f"_{session_kw}_" in f.lower() and f.lower().endswith(".mff"):
            return os.path.join(eeg_folder, f)
    return None


def find_csv(subject_path, session_kw):
    suffix = CSV_SUFFIX.get(session_kw, "_1")
    for name in os.listdir(subject_path):
        if name.lower() == "startle output" and os.path.isdir(os.path.join(subject_path, name)):
            search = os.path.join(subject_path, name)
            break
    else:
        search = subject_path
    for root, _, files in os.walk(search):
        for f in sorted(files):
            if "demo" not in f.lower() and f.lower().endswith(".csv"):
                if re.search(rf"{re.escape(suffix)}\.csv$", f, re.IGNORECASE):
                    return os.path.join(root, f)
    return None


def get_din_onsets(raw, ch_name):
    """Return first sample of each trigger pulse for one DIN channel."""
    if ch_name not in raw.ch_names:
        return np.array([])
    d = raw.get_data(picks=[ch_name])[0]
    if d.max() == 0:
        return np.array([])
    samps = np.where(d > d.max() * 0.9)[0]
    gaps  = np.where(np.diff(samps) > 1)[0]
    return np.concatenate([[samps[0]], samps[gaps + 1]])


def normalize(data):
    mn, mx = data.min(), data.max()
    if mx == mn:
        return np.zeros_like(data)
    return 2.0 * (data - mn) / (mx - mn) - 1.0


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    subject    = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SUBJECT
    session_kw = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SESSION

    subject_path = os.path.join(RAW_DATA_DIR, subject)
    eeg_folder   = os.path.join(subject_path, "EEG")

    path = find_mff(eeg_folder, session_kw)
    if not path:
        print(f"[!] No MFF found for {subject}/{session_kw}"); sys.exit(1)

    csv_path = find_csv(subject_path, session_kw)
    if not csv_path:
        print(f"[!] No CSV found for {subject}/{session_kw}"); sys.exit(1)

    print(f"Loading : {os.path.basename(path)}")
    print(f"CSV     : {os.path.basename(csv_path)}")

    raw   = mne.io.read_raw_egi(path, preload=True, verbose=False,
                                 events_as_annotations=True)
    sfreq = raw.info["sfreq"]

    # ── Parse CSV ────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    df["abs_s"] = pd.to_timedelta(df["absolute_duration"]).dt.total_seconds()
    df_sound  = df[df["has_sound"] == True].reset_index(drop=True)
    df_silent = df[df["has_sound"] == False].reset_index(drop=True)
    print(f"CSV: {len(df)} trials  ({len(df_sound)} sound, {len(df_silent)} silent)")

    # ── Crop to task ─────────────────────────────────────────────────────────
    raw_din = raw.copy().pick([ch for ch in raw.ch_names if ch.startswith("D")])

    d101_samps = get_din_onsets(raw_din, f"D{TRIGGER_START}")
    d124_samps = get_din_onsets(raw_din, f"D{TRIGGER_END}")

    if len(d101_samps) and len(d124_samps):
        t_start = d101_samps[0] / sfreq
        t_end   = d124_samps[-1] / sfreq
        raw.crop(tmin=t_start, tmax=t_end)
        print(f"Cropped : {t_start:.1f}s → {t_end:.1f}s  ({(t_end-t_start)/60:.1f} min)")
    else:
        t_start = 0.0
        print("[!] Boundary triggers not found — using full recording.")

    # ── D110 triggers (task-relative, exclude pre-task) ───────────────────────
    # Re-pick DIN from cropped raw (sample 0 = t_start)
    raw_din2 = raw.copy().pick([ch for ch in raw.ch_names if ch.startswith("D")])
    d110_raw  = get_din_onsets(raw_din, f"D{TRIGGER_STARTLE}")          # absolute
    d110_task = d110_raw[d110_raw / sfreq > t_start] / sfreq - t_start  # task-relative, post-D101

    n_use = min(len(d110_task), len(df_sound))
    d110_task  = d110_task[:n_use]
    df_sound_u = df_sound.iloc[:n_use].reset_index(drop=True)

    print(f"D110 triggers (post-task-start): {len(d110_task)}  |  sound rows: {len(df_sound_u)}")

    # ── Calibrate CSV abs_s → task-relative ──────────────────────────────────
    # Formula:  D110_task = abs_s + fixation_duration + startle_start - D101_script_time
    d101_script_times = (df_sound_u["abs_s"].values
                         + df_sound_u["fixation_duration"].values
                         + df_sound_u["startle_start"].values
                         - d110_task)
    d101_st = float(np.mean(d101_script_times))
    print(f"D101 script offset: {d101_st:.4f}s  (std {np.std(d101_script_times):.4f}s)")

    # ── Build trial list ──────────────────────────────────────────────────────
    trials = []

    # Sound trials (accurate — from D110)
    for i, row in df_sound_u.iterrows():
        d110_t     = d110_task[i]
        pic_onset  = d110_t - float(row["startle_start"])
        pic_offset = pic_onset + float(row["picture_duration"])
        trials.append({
            "pic_onset":  max(pic_onset, 0),
            "pic_offset": pic_offset,
            "d110_t":     d110_t,
            "has_sound":  True,
        })

    # Silent trials (reconstructed from abs_s)
    for _, row in df_silent.iterrows():
        pic_onset  = float(row["abs_s"]) + float(row["fixation_duration"]) - d101_st
        pic_offset = pic_onset + float(row["picture_duration"])
        trials.append({
            "pic_onset":  max(pic_onset, 0),
            "pic_offset": pic_offset,
            "d110_t":     None,
            "has_sound":  False,
        })

    trials.sort(key=lambda t: t["pic_onset"])

    # ── Pick channels ─────────────────────────────────────────────────────────
    available = [ch for ch in RESP_CHANNELS if ch in raw.ch_names]
    missing   = [ch for ch in RESP_CHANNELS if ch not in raw.ch_names]
    if missing:
        print(f"Not in file (skipped): {missing}")
    if not available:
        print("[!] No target channels found."); sys.exit(1)
    print(f"Channels: {available}")

    all_times = raw.times
    duration  = all_times[-1]
    ch_data   = {ch: normalize(raw.get_data(picks=[ch])[0]) for ch in available}
    n_ch      = len(available)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 2.4 * n_ch + 2.0))
    fig.patch.set_facecolor("#1a1a2e")

    gs = gridspec.GridSpec(
        n_ch, 1,
        hspace=0.06,
        top=0.92, bottom=0.18, left=0.10, right=0.97
    )
    axes    = [fig.add_subplot(gs[i]) for i in range(n_ch)]
    ax_slid = fig.add_axes([0.10, 0.06, 0.87, 0.04])

    lines = []

    for i, ch in enumerate(available):
        ax    = axes[i]
        color = CH_COLORS[i % len(CH_COLORS)]
        ax.set_facecolor("#0d0d1a")

        for tr in trials:
            fc = SOUND_COLOR if tr["has_sound"] else NOSOUND_COLOR
            ax.axvspan(tr["pic_onset"], tr["pic_offset"],
                       color=fc, alpha=0.18, linewidth=0)
            if tr["d110_t"] is not None:
                ax.axvline(tr["d110_t"], color=STARTLE_COLOR,
                           linewidth=0.9, linestyle="--", alpha=0.8)

        ln, = ax.plot([], [], color=color, linewidth=0.8)
        lines.append(ln)

        ax.set_ylim(-1.2, 1.2)
        ax.set_yticks([])
        ax.set_ylabel(ch, color=color, fontsize=9, rotation=0,
                      labelpad=65, va="center")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        ax.tick_params(colors="#aaaaaa")
        if i < n_ch - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (s, task-relative)", color="#aaaaaa")
            ax.tick_params(axis="x", colors="#aaaaaa", labelsize=8)

    legend_handles = [
        mpatches.Patch(color=SOUND_COLOR,   alpha=0.5, label="Trial — with sound"),
        mpatches.Patch(color=NOSOUND_COLOR, alpha=0.5, label="Trial — silent control"),
        plt.Line2D([0], [0], color=STARTLE_COLOR, linestyle="--",
                   linewidth=1.2, label="D110 startle onset"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right",
                   fontsize=8, facecolor="#1a1a2e", labelcolor="white",
                   edgecolor="#444466")

    fig.suptitle(
        f"{subject}  {session_kw}  —  respiration channels (raw, normalized)\n"
        f"← → scroll  |  [ ] half-page  |  + − zoom",
        color="white", fontsize=10
    )

    slider = Slider(ax_slid, "t (s)", 0.0, max(0.1, duration - WINDOW_SEC),
                    valinit=0.0, color="#4a4a8a", track_color="#2a2a4a")
    ax_slid.set_facecolor("#1a1a2e")
    slider.label.set_color("white")
    slider.valtext.set_color("white")

    state = {"window": WINDOW_SEC}

    def update(t0):
        t0 = float(t0)
        t1 = t0 + state["window"]
        s0 = int(t0 * sfreq)
        s1 = int(t1 * sfreq)
        seg = all_times[s0:s1]
        for i, ch in enumerate(available):
            lines[i].set_data(seg, ch_data[ch][s0:s1])
            axes[i].set_xlim(t0, t1)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_key(event):
        w   = state["window"]
        t0  = slider.val
        dur = duration - w
        if event.key == "right":
            slider.set_val(min(t0 + w, dur))
        elif event.key == "left":
            slider.set_val(max(t0 - w, 0))
        elif event.key == "]":
            slider.set_val(min(t0 + w / 2, dur))
        elif event.key == "[":
            slider.set_val(max(t0 - w / 2, 0))
        elif event.key in ("+", "="):
            state["window"] = max(5.0, w / 1.5)
            slider.valmax = max(0.1, duration - state["window"])
            slider.ax.set_xlim(slider.valmin, slider.valmax)
            update(slider.val)
        elif event.key == "-":
            state["window"] = min(duration, w * 1.5)
            slider.valmax = max(0.1, duration - state["window"])
            slider.ax.set_xlim(slider.valmin, slider.valmax)
            update(slider.val)

    fig.canvas.mpl_connect("key_press_event", on_key)
    update(0.0)
    plt.show()


if __name__ == "__main__":
    main()
