"""
MFF File Viewer
===============
Follows the EMG raw logic exactly:
  1. Load with mne.io.read_raw_egi
  2. Extract DIN events by amplitude threshold
  3. Crop to task segment (DIN 101 → DIN 124)
  4. Display raw with MNE viewer — no filtering, no processing

Usage:
    .venv/bin/python scripts/view_mff.py              # default subject
    .venv/bin/python scripts/view_mff.py YR08 eve
    .venv/bin/python scripts/view_mff.py NB03 mor
    .venv/bin/python scripts/view_mff.py /full/path/to/EEG/folder
"""

import os
import sys
import numpy as np
import pandas as pd
import mne

# ── Config ────────────────────────────────────────────────────────────────────

RAW_DATA_DIR = r"/Volumes/My Passport/startle_raw"
DEFAULT_SUBJECT = "DA01"

TRIGGER_SESSION_START = 101
TRIGGER_SESSION_END   = 124

# Number of channels visible at once in the scrollable viewer
N_CHANNELS_VISIBLE = 20

# Seconds of data per screen
DURATION_SEC = 10.0


# ── DIN event extraction (same as EMG script) ─────────────────────────────────

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


# ── File helpers ──────────────────────────────────────────────────────────────

def find_mff_files(folder):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".mff")
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    arg1 = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SUBJECT
    session_kw = sys.argv[2] if len(sys.argv) > 2 else None

    if os.sep not in arg1 and not arg1.startswith("/"):
        folder = os.path.join(RAW_DATA_DIR, arg1, "EEG")
    else:
        folder = arg1

    if not os.path.isdir(folder):
        print(f"[!] Folder not found: {folder}")
        sys.exit(1)

    mff_files = find_mff_files(folder)
    if not mff_files:
        print(f"[!] No .mff files found in: {folder}")
        sys.exit(1)

    if session_kw:
        matches = [f for f in mff_files if f"_{session_kw}_" in os.path.basename(f).lower()]
        if not matches:
            print(f"[!] No .mff matching '{session_kw}'. Available:")
            for f in mff_files:
                print(f"  {os.path.basename(f)}")
            sys.exit(1)
        path = matches[0]
    else:
        if len(mff_files) > 1:
            print("Multiple .mff files found — loading first. Pass session keyword (eve/mor/sleep) to pick one.")
            for i, f in enumerate(mff_files):
                print(f"  [{i}] {os.path.basename(f)}")
        path = mff_files[0]

    print(f"\nLoading: {os.path.basename(path)} ...")
    raw = mne.io.read_raw_egi(path, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]

    # Crop to task segment exactly as the EMG script does
    events_df = get_events_from_eeg(raw)
    start_samps = events_df[events_df["Channel"] == f"D{TRIGGER_SESSION_START}"]["Sample"].values
    end_samps   = events_df[events_df["Channel"] == f"D{TRIGGER_SESSION_END}"]["Sample"].values

    if len(start_samps) == 0 or len(end_samps) == 0:
        print("[!] Task boundary triggers not found — showing full session.")
    else:
        t_start = start_samps[0] / sfreq
        t_end   = end_samps[-1] / sfreq
        raw.crop(tmin=t_start, tmax=t_end)
        print(f"Cropped to task: {t_start:.1f}s → {t_end:.1f}s  ({(t_end-t_start)/60:.1f} min)")

    # Drop DIN channels from the plot (keep only EEG signal channels)
    eeg_chs = [ch for ch in raw.ch_names if not ch.startswith("D")]
    raw_plot = raw.copy().pick(eeg_chs)

    print(f"Channels : {len(raw.ch_names)} total  |  {len(eeg_chs)} EEG plotted")
    print(f"Sfreq    : {sfreq} Hz")
    print(f"Duration : {raw_plot.times[-1]:.1f} s")
    print(f"\nOpening raw viewer — arrows to scroll, +/- to scale, ? for help.\n")

    raw_plot.plot(
        duration=DURATION_SEC,
        n_channels=N_CHANNELS_VISIBLE,
        title=os.path.basename(path),
        show=True,
        block=True,
    )


if __name__ == "__main__":
    main()
