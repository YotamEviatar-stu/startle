import argparse
import glob
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import mne
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hr import config as cfg

SFREQ = 250.0
EXTRA_CHANNELS = ["SpO2-Pulse"]


def candidates(subj, sess):
    token = cfg.SESSION_FILE_ALIASES.get(f"{subj}/{sess}", sess)
    pat = re.compile(rf"_{token}_", re.IGNORECASE)
    files = sorted(glob.glob(os.path.join(cfg.RAW_DATA_DIR, subj, "EEG", "*.mff")) + glob.glob(os.path.join(cfg.RAW_DATA_DIR, subj, "*.mff")))
    return [f for f in files if pat.search(os.path.basename(f)) and not os.path.basename(f).startswith("._")]


def din_channels(raw):
    return [c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == "stim" and c.startswith("D")]


def din_annotations(raw, chans):
    data = raw.get_data(picks=chans)
    onsets, names = [], []
    for ch, x in zip(chans, data):
        mx = x.max()
        if mx <= 0:
            continue
        on = np.flatnonzero(np.diff(np.r_[False, x >= 0.9 * mx].astype(np.int8)) == 1)
        onsets += list(on / raw.info["sfreq"])
        names += [ch] * len(on)
    return mne.Annotations(onsets, 0.0, names, orig_time=raw.annotations.orig_time)


def process(subj, sess, force):
    out_dir = os.path.join(cfg.DATA_250_DIR, subj)
    out = os.path.join(out_dir, f"{subj}_{sess}_raw.fif")
    if not force and os.path.exists(out):
        return subj, sess, "exists", ""
    errors = []
    for f in candidates(subj, sess):
        try:
            raw = mne.io.read_raw_egi(f, preload=True, verbose="error")
            dins = din_channels(raw)
            ann = din_annotations(raw, dins)
            extra = [c for c in EXTRA_CHANNELS if c in raw.ch_names]
            eeg = [c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"]
            raw.pick(eeg + extra + dins)
            native = raw.info["sfreq"]
            raw.resample(SFREQ, verbose="error")
            raw.set_annotations(ann)
            os.makedirs(out_dir, exist_ok=True)
            raw.save(out, fmt="single", overwrite=True, verbose="error")
            missing = sorted(set(EXTRA_CHANNELS) - set(extra))
            return subj, sess, "ok", (f"{os.path.basename(f)} {native:g}Hz {raw.times[-1]:.0f}s "
                                      f"eeg={len(eeg)} din={len(dins)} markers={len(ann)}"
                                      + (f" missing={missing}" if missing else ""))
        except Exception as e:
            errors.append(f"{os.path.basename(f)}: {type(e).__name__}: {str(e)[:120]}")
    return subj, sess, "fail", "; ".join(errors) or "no matching mff"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(cfg.DATA_250_DIR, exist_ok=True)
    subjects = a.subjects or sorted(d for d in os.listdir(cfg.RAW_DATA_DIR) if os.path.isdir(os.path.join(cfg.RAW_DATA_DIR, d)))
    subjects = [s for s in subjects if s not in cfg.SUBJECTS_EXCLUDE]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(process, s, x, a.force) for s in subjects for x in cfg.SESSIONS]
        for fu in as_completed(futs):
            r = fu.result()
            rows.append(r)
            print(*r, flush=True)
    log_path = os.path.join(cfg.DATA_250_DIR, "downsample_log.csv")
    log = pd.DataFrame(rows, columns=["subject", "session", "status", "detail"])
    if os.path.exists(log_path):
        old = pd.read_csv(log_path, keep_default_na=False)
        log = pd.concat([old, log[log.status != "exists"]]).drop_duplicates(["subject", "session"], keep="last")
    log = log.sort_values(["subject", "session"])
    log.to_csv(log_path, index=False)
    print(log.status.value_counts().to_string())


if __name__ == "__main__":
    main()
