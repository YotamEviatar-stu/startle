"""
Session-wide trigger-order audit.

Walks every session's raw D101->D124 trigger stream (independent of any
per-trial CSV alignment logic in trial_epochs.py) and checks that
consecutive "meaningful" events never repeat the same category. The only
two legal per-trial sequences are:

    1. D105 -> code -> D105        (no-sound trial)
    2. D105 -> code -> D110 -> D105 (sound trial)

so category-adjacent pairs are never allowed:
    fixation-fixation (D105 followed by D105 with no code in between)
    image-image       (code followed by code with no D105/D110 in between)
    sound-sound       (D110 followed by D110)

This is a raw-stream check, not a value-matched one: unlike
trial_epochs.build_trial_epochs (which discards a spurious DI/DIN blip if
exactly one candidate matches the CSV's trigger_num), this script flags
every back-to-back same-category pair regardless of whether a later CSV
match would silently absorb it. That is the point -- it is meant to catch
the raw mistake (e.g. AS09: DI73 then DI27, 7.2s apart, no D105 between
them) directly in the trigger log, not rely on downstream tolerance.

Usage: python -m Startle.inspect_triggers
"""

import os

import mne

import Startle.extras.emg_raw_potentiation as emg
from Startle.trial_epochs import _classify_channel, _merge_contiguous

RAW_DATA_DIR = "/Volumes/My Passport/startle_raw"
SESSION_MAP = emg.SESSION_MAP


def audit_session(mff_path, sfreq_hint=None):
    """Return a list of violation dicts for one session's raw trigger stream."""
    raw = mne.io.read_raw_egi(mff_path, preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])

    events_df = emg.get_events_from_eeg(raw)
    merged = _merge_contiguous(events_df)
    channels = [m["Channel"] for m in merged]

    if "D101" not in channels or "D124" not in channels:
        return [{"error": "D101 or D124 not found"}], sfreq

    d101_i = channels.index("D101")
    d124_i = channels.index("D124", d101_i)
    window = merged[d101_i:d124_i + 1]

    # Reduce to the category-relevant event stream only: fixation (D105),
    # picture code (DIN/DI), startle probe (D110). D101/D124 are the block
    # boundaries, not trial-cycle events, so they're excluded from the
    # adjacency check itself.
    seq = []
    for ev in window:
        kind, value = _classify_channel(ev["Channel"])
        if kind == "structural" and value == 105:
            seq.append(("fixation", ev["Channel"], ev["Sample"]))
        elif kind == "structural" and value == 110:
            seq.append(("sound", ev["Channel"], ev["Sample"]))
        elif kind == "picture":
            seq.append(("image", ev["Channel"], ev["Sample"]))
        # everything else (other D-codes, noise channels) is ignored here

    violations = []
    for i in range(1, len(seq)):
        prev_cat, prev_ch, prev_s = seq[i - 1]
        cur_cat, cur_ch, cur_s = seq[i]
        if prev_cat == cur_cat:
            violations.append({
                "category": f"{prev_cat}-{prev_cat}",
                "prev_channel": prev_ch, "prev_sample": prev_s,
                "cur_channel": cur_ch, "cur_sample": cur_s,
                "gap_sec": (cur_s - prev_s) / sfreq,
                "prev_t": prev_s / sfreq, "cur_t": cur_s / sfreq,
            })

    return violations, sfreq


def main():
    subject_folders = sorted([
        d for d in os.listdir(RAW_DATA_DIR)
        if os.path.isdir(os.path.join(RAW_DATA_DIR, d))
    ])

    total_violations = 0
    total_sessions = 0
    clean_sessions = 0

    for subj in subject_folders:
        subj_path = os.path.join(RAW_DATA_DIR, subj)
        eeg_folder = os.path.join(subj_path, "EEG")
        if not os.path.isdir(eeg_folder):
            continue

        for sess_key, sess_info in SESSION_MAP.items():
            mff_path = emg.find_mff_file(eeg_folder, sess_key)
            if mff_path is None:
                continue

            total_sessions += 1
            try:
                violations, sfreq = audit_session(mff_path)
            except Exception as e:
                print(f"[{subj} {sess_key}] FAILED TO LOAD: {e}")
                continue

            if violations and "error" in violations[0]:
                print(f"[{subj} {sess_key}] {violations[0]['error']}")
                continue

            if not violations:
                clean_sessions += 1
                continue

            total_violations += len(violations)
            print(f"\n[{subj} {sess_key}] {mff_path}")
            for v in violations:
                print(
                    f"  {v['category']}: {v['prev_channel']} t={v['prev_t']:.3f}s "
                    f"-> {v['cur_channel']} t={v['cur_t']:.3f}s "
                    f"(gap={v['gap_sec']:.3f}s)"
                )

    print(
        f"\n{clean_sessions}/{total_sessions} sessions clean, "
        f"{total_violations} total violations across "
        f"{total_sessions - clean_sessions} affected sessions."
    )


if __name__ == "__main__":
    main()
