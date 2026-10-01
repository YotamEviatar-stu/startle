import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "_cache" / "caracas"
DEPS = HERE.parent / "_cache" / "deps"
MEASURES = ("sk", "ku", "RR", "Rampl", "bpm", "RPeakstoNoise")
MATLAB_NEED_GB = 5.0


def list_sessions():
    out = []
    for subj_dir in sorted(Path(config.RAW_DATA_DIR).iterdir()):
        eeg_dir = subj_dir / "EEG"
        if not eeg_dir.is_dir():
            continue
        subject = config.SUBJECT_ID.get(subj_dir.name, subj_dir.name)
        for mff in sorted(eeg_dir.glob("*.mff")):
            if mff.name.startswith("._"):
                continue
            session = next((s for s in config.SESSIONS if f"_{s}_" in mff.name), None)
            if session is not None:
                out.append((subject, session, mff))
    return out


def heavy_others(min_gb=1.0):
    mine = {os.getpid()} | {p.pid for p in psutil.Process().children(recursive=True)}
    out = []
    for p in psutil.process_iter(["pid", "name", "memory_info", "cmdline"]):
        try:
            if p.pid in mine or p.info["memory_info"] is None:
                continue
            gb = p.info["memory_info"].rss / 1e9
            name = (p.info["name"] or "").lower()
            if gb >= min_gb and "matlab" in name:
                out.append(f"{p.pid} {' '.join(p.info['cmdline'] or [name])[:80]} {gb:.1f}GB")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def wait_for_memory(need_gb=MATLAB_NEED_GB):
    while True:
        avail = psutil.virtual_memory().available / 1e9
        others = heavy_others()
        if avail >= need_gb and not others:
            return
        print(f"  waiting: {avail:.1f}GB available (need {need_gb}), heavy: {others}", flush=True)
        time.sleep(120)


def run_matlab(mff, out_mat, log):
    call = (f"addpath('{HERE}'); "
            f"caracas_session('{mff}', '{out_mat}', '{DEPS}')")
    with open(log, "w") as fh:
        subprocess.run([config.MATLAB_BIN, "-batch", call], stdout=fh, stderr=subprocess.STDOUT, check=True)


def mat_to_npz(out_mat, npz, subject, session, mff):
    m = loadmat(out_mat, squeeze_me=False)
    cardiac_idx = m["cardiac_idx"].ravel().astype(int) - 1
    signal = np.asarray(m["cardiac_signal"], dtype=np.float32)
    sfreq = float(m["fsample"].squeeze())
    np.savez(
        npz,
        subject=subject, session=session, mff=mff.name,
        signal=signal, sfreq=sfreq, t0=float(m["t0"].squeeze()),
        cardiac_idx=cardiac_idx,
        is_cardiac=m["is_cardiac"].ravel().astype(bool),
        not_cardiac=m["NotCardiac"].astype(bool),
        **{k: m[k].ravel().astype(float) for k in MEASURES},
        thresh_sk=float(m["thresh_sk"].squeeze()), thresh_ku=float(m["thresh_ku"].squeeze()),
        thresh_RR=float(m["thresh_RR"].squeeze()), thresh_Rampl=float(m["thresh_Rampl"].squeeze()),
        thresh_bpm=m["thresh_bpm"].ravel().astype(float),
        unmixing=m["unmixing"], topo=m["topo"],
        topolabel=np.array([str(x[0]) for x in m["topolabel"].ravel()]),
        ncomp=int(m["ncomp"].squeeze()),
    )


def main(only=None):
    CACHE.mkdir(parents=True, exist_ok=True)
    for subject, session, mff in list_sessions():
        if only and f"{subject}/{session}" not in only:
            continue
        stem = mff.stem
        out_dir = CACHE / subject
        out_dir.mkdir(exist_ok=True)
        npz = out_dir / f"{stem}.npz"
        if npz.exists():
            continue
        out_mat = out_dir / f"{stem}.mat"
        log = out_dir / f"{stem}.log"
        wait_for_memory()
        print(f"{subject}/{session}  {mff.name}", flush=True)
        try:
            run_matlab(mff, out_mat, log)
            mat_to_npz(out_mat, npz, subject, session, mff)
            out_mat.unlink()
        except Exception as e:
            print(f"  FAILED: {e!r} (see {log})", flush=True)


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
