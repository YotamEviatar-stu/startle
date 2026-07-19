"""
Dump the full raw trigger stream (task block [D101, D124]) for every
subject/session to a text file, mirroring the format of the existing
AS09_eve_trigger_list.txt / DA01_mor_trigger_list.txt files at the repo
root. Triggers are read directly from the .mff (via
emg_raw_potentiation.get_events_from_eeg), NOT from the CSV -- this is a
raw-hardware-log dump, independent of any CSV alignment logic.

Output layout:
    OUTPUT_DIR/<SubjectID>/eve.txt
    OUTPUT_DIR/<SubjectID>/mor.txt

Usage: python -m Startle.dump_trigger_lists
"""

import os

import mne

import Startle.extras.emg_raw_potentiation as emg
from Startle.trial_epochs import _classify_channel, _merge_contiguous

RAW_DATA_DIR = "/Volumes/My Passport/startle_raw"
SESSION_MAP = emg.SESSION_MAP
OUTPUT_DIR = "/Users/yotameviatar/vs_code/trigger_lists"

_ANNOTATIONS = {105: "<- fixation onset", 110: "<- SOUND"}

# ES29 eve is a known crash/restart: the original recording
# (ES29_eve_20260605_125423.mff) crashed at 2026-06-05T03:54:52.405683+03:00
# (its last logged event, mid-trial, right after a D110) -- before it could
# write epochs.xml, so MNE can't load it at all. The restart
# (ES29_eve_2_20260605_010726.mff) began 25.055s later and continues the same
# task without resending D101 (confirmed via both files' raw
# "Events_255 DINs.xml" logs + info.xml recordTime, bypassing MNE for the
# broken file). Decision: do not attempt binary-level recovery of the
# pre-crash ~8min segment -- use the restart file alone, with its own first
# event as the practical window start instead of requiring D101.
_NO_D101_OVERRIDE = {("ES29", "eve")}


_UNANNOTATED_STRUCTURAL = {101, 124}   # session boundaries -- no note, matches reference format


def _annotate(channel):
    kind, value = _classify_channel(channel)
    if kind == "structural":
        if value in _UNANNOTATED_STRUCTURAL:
            return None
        note = _ANNOTATIONS.get(value)
        return note if note is not None else "<- unrecognized (non-task) code"
    if kind == "picture":
        return "<- image code"
    return None


def format_trigger_list(subj, sess_key, window, note_header=None):
    lines = [f"{subj} {sess_key} -- full trigger list, task block [D101, D124]"]
    if note_header:
        lines.append(f"NOTE: {note_header}")
    for i, ev in enumerate(window):
        ch, t = ev["Channel"], ev["Sample"] / ev["sfreq"]
        note = _annotate(ch)
        line = f"{i:3d}  {ch:<9} t={t:8.3f}s"
        if note:
            line += f"  {note}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def build_window(mff_path, require_d101=True):
    raw = mne.io.read_raw_egi(mff_path, preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])

    events_df = emg.get_events_from_eeg(raw)
    merged = _merge_contiguous(events_df)
    channels = [m["Channel"] for m in merged]

    if "D124" not in channels:
        raise ValueError("D124 not found in this recording")
    d124_i = channels.index("D124")

    if require_d101:
        if "D101" not in channels:
            raise ValueError("D101 not found in this recording")
        start_i = channels.index("D101")
    else:
        start_i = 0

    window = merged[start_i:d124_i + 1]
    for ev in window:
        ev["sfreq"] = sfreq
    return window


def main():
    subject_folders = sorted([
        d for d in os.listdir(RAW_DATA_DIR)
        if os.path.isdir(os.path.join(RAW_DATA_DIR, d))
    ])

    n_written, n_failed = 0, 0

    for subj in subject_folders:
        subj_path = os.path.join(RAW_DATA_DIR, subj)
        eeg_folder = os.path.join(subj_path, "EEG")
        if not os.path.isdir(eeg_folder):
            continue

        subj_out_dir = os.path.join(OUTPUT_DIR, subj)

        for sess_key in SESSION_MAP:
            candidates = emg.find_mff_candidates(eeg_folder, sess_key)
            if not candidates:
                continue

            require_d101 = (subj, sess_key) not in _NO_D101_OVERRIDE
            note_header = None
            if not require_d101:
                note_header = (
                    "original recording crashed mid-session (last event "
                    "2026-06-05T03:54:52+03:00, right after a D110, before "
                    "epochs.xml could be written); this is the restart file "
                    "(began 25.055s later), used alone -- D101 was never "
                    "resent, so this window starts at the restart's first "
                    "event instead of D101. Pre-crash segment not recovered."
                )

            # find_mff_file only returns candidates[0] (alphabetical), which
            # picks the ORIGINAL attempt over a "_2" restart if a session was
            # aborted and re-recorded (e.g. ES29 eve) -- try every candidate
            # in order until one actually loads, per find_mff_candidates'
            # own docstring.
            window, text, last_err = None, None, None
            for mff_path in candidates:
                try:
                    window = build_window(mff_path, require_d101=require_d101)
                    text = format_trigger_list(subj, sess_key, window, note_header=note_header)
                    break
                except Exception as e:
                    last_err = e
                    continue

            if text is None:
                print(f"[{subj} {sess_key}] FAILED: {last_err}")
                n_failed += 1
                continue

            os.makedirs(subj_out_dir, exist_ok=True)
            out_path = os.path.join(subj_out_dir, f"{sess_key}.txt")
            with open(out_path, "w") as f:
                f.write(text)
            print(f"[{subj} {sess_key}] -> {out_path}")
            n_written += 1

    print(f"\n{n_written} trigger list(s) written, {n_failed} session(s) failed.")


if __name__ == "__main__":
    main()
