"""
airflow_mff_show.py — presents ONE raw .mff recording, straight off the drive.

No Layer 1/2 processing: no bandpass, no resample, no NK2. This is for
answering "does this recording actually have signal / triggers?" for a
subject/session that failed to make it into the cache (see airflow_main.py's
TriggerAlignmentError / mne read-failure handling) — a diagnostic tool, not
part of the analysis pipeline.

Run with:
    .venv/bin/python Startle/Airflow/airflow_mff_show.py
"""

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import mne

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # vs_code root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))        # Startle/
sys.path.insert(0, os.path.dirname(__file__))                            # Startle/Airflow/

import airflow_config as cfg

# emg_raw_potentiation.py (the canonical EMG reference script) does its own
# `matplotlib.use("Agg")` at MODULE level, unconditionally -- not inside an
# `if __name__ == "__main__"` guard. Importing it here to reuse its helpers
# therefore silently force-switches the WHOLE process's matplotlib backend
# the moment this line runs. If a caller (e.g. airflow_glm_show.ipynb, via
# its View 0) already set an inline backend for live plotting, that gets
# clobbered and every subsequent `plt.show()` in the notebook goes silently
# nowhere -- no error, just no figure. Save/restore around the import so this
# module never has that side effect on whoever imports it.
_backend_before_emg_import = matplotlib.get_backend()
from Startle.extras.emg_raw_potentiation import find_startle_output_folder, find_mff_file, get_events_from_eeg
matplotlib.use(_backend_before_emg_import)

# ── What to show ───────────────────────────────────────────────────────────────
# MFF_PATH overrides SUBJECT/SESSION lookup — set this explicitly when the
# session you want isn't the one find_mff_file() would pick (e.g. a subject
# has two "*_eve_*" recordings on disk and you want the non-default one).
MFF_PATH = '/Volumes/My Passport/startle_raw/MG14/EEG/MG14_mor_20250928_075517.mff'
SUBJECT  = "MG14"        # only used if MFF_PATH is None
SESSION  = "mor"          # only used if MFF_PATH is None

CHANNEL         = cfg.AIRFLOW_CHANNEL   # None = show every channel that isn't a DIN
SHOW_TRIGGERS   = True                  # overlay D-channel event markers
MAX_PLOT_POINTS = 8000                  # decimation target — keeps MNE/matplotlib responsive
OUTPUT_DIR      = os.path.join(cfg.OUTPUT_DIR, "mff_show")


# ── Peak-preserving decimation ──────────────────────────────────────────────────
# Naive stride-based downsampling (signal[::k]) can silently step over a real
# spike or dropout depending on phase. This bins the signal and keeps both the
# min AND max of every bin, then plots the min/max envelope — any excursion in
# the raw data still shows up, just at a display-safe sample count.
def _decimate_minmax(sig, times, target_points):
    n = len(sig)
    if n <= target_points * 2:
        return times, sig, sig
    bin_size = int(np.ceil(n / target_points))
    n_bins   = int(np.ceil(n / bin_size))
    pad      = n_bins * bin_size - n
    sig_p   = np.pad(sig, (0, pad), mode="edge")
    times_p = np.pad(times, (0, pad), mode="edge")
    sig_r  = sig_p.reshape(n_bins, bin_size)
    time_r = times_p.reshape(n_bins, bin_size)
    return time_r.mean(axis=1), sig_r.min(axis=1), sig_r.max(axis=1)


def show_mff(mff_path, channel=None, show_triggers=True,
             max_points=MAX_PLOT_POINTS, output_dir=None, show=False, out_name=None):
    print(f"Reading {mff_path} ...")
    try:
        raw = mne.io.read_raw_egi(mff_path, preload=True, verbose=False)
    except Exception as e:
        print(f"[!] Could not read {mff_path}: {e}")
        return None

    sfreq = raw.info["sfreq"]
    print(f"  sfreq={sfreq} Hz  duration={raw.times[-1]:.1f}s  "
          f"n_channels={len(raw.ch_names)}")

    if channel is not None:
        target = next((c for c in raw.ch_names if channel.lower() in c.lower()), None)
        if target is None:
            print(f"[!] Channel '{channel}' not found. Available: {raw.ch_names}")
            return None
        chans_to_plot = [target]
    else:
        chans_to_plot = [c for c in raw.ch_names if not c.startswith("D")]

    times = raw.times
    events_df = None
    if show_triggers:
        events_df = get_events_from_eeg(raw)
        if events_df.empty:
            print("  No DIN trigger events decoded on any D-channel.")
        else:
            print(f"  {len(events_df)} DIN events decoded "
                  f"across {events_df['Channel'].nunique()} channels.")

    fig, axes = plt.subplots(len(chans_to_plot), 1,
                              figsize=(14, 3.2 * len(chans_to_plot)),
                              sharex=True, squeeze=False)
    axes = axes[:, 0]

    for ax, ch_name in zip(axes, chans_to_plot):
        sig = raw.get_data(picks=[ch_name])[0]
        t_dec, lo, hi = _decimate_minmax(sig, times, max_points)
        ax.fill_between(t_dec, lo, hi, color="#2C3E50", linewidth=0, alpha=0.9)
        ax.set_ylabel(ch_name, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

        if events_df is not None and not events_df.empty:
            for _, row in events_df.iterrows():
                ax.axvline(row["Sample"] / sfreq, color="#E74C3C",
                           linewidth=0.6, alpha=0.5, zorder=0)

    axes[-1].set_xlabel("Time in session (s)")
    fig.suptitle(f"{os.path.basename(mff_path)}  (raw, unfiltered)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # out_name lets the caller pin the saved filename to the SUBJECT/session
        # it actually asked for -- the fallback (mff basename) instead reflects
        # whatever file find_mff_file() happened to pick, which silently
        # mislabels the PNG if that lookup ever resolves to an unexpected file.
        fname = os.path.join(output_dir, out_name or
                              os.path.basename(mff_path).replace(".mff", ".png"))
        fig.savefig(fname, dpi=cfg.PLOT_DPI, bbox_inches="tight")
        print(f"  Saved -> {fname}")

    if show:
        plt.show()
    plt.close(fig)
    return raw


if __name__ == "__main__":
    # Forcing the Agg backend must stay INSIDE this __main__ guard, not at
    # module import time: airflow_glm_show.ipynb imports this file to call
    # show_mff() inline, and matplotlib.use() switches the backend for the
    # WHOLE process the moment the module is imported — it would silently
    # kill every other inline plot in the notebook (Views 1-6), not just this
    # one. A standalone script run has no inline backend to protect, so Agg
    # is safe (and desired, since it never calls plt.show()) here only.
    matplotlib.use("Agg")

    path = MFF_PATH
    if path is None:
        eeg_folder = os.path.join(cfg.RAW_DATA_DIR, SUBJECT, "EEG")
        path = find_mff_file(eeg_folder, SESSION)
        if path is None:
            raise FileNotFoundError(
                f"No '*_{SESSION}_*.mff' found under {eeg_folder}")

    show_mff(path, channel=CHANNEL, show_triggers=SHOW_TRIGGERS,
              output_dir=OUTPUT_DIR)
