import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mne
import numpy as np
from mne_icalabel.iclabel import iclabel_label_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hr import config as cfg
from hr.caracas_py import caracas

DATA = Path(cfg.DS_DATA_DIR)
CACHE = Path(__file__).resolve().parent / "_cache" / "ica_all"
CLASSES = ("brain", "muscle artifact", "eye blink", "heart beat", "line noise", "channel noise", "other")
SEED = 97


def sessions():
    return sorted((f.parent.name, f.stem.split("_")[1], f) for f in DATA.glob("*/*_raw.fif"))


def run(subj, sess, fif):
    out = CACHE / f"{subj}_{sess}.npz"
    if out.exists():
        return f"{subj} {sess}: cached"
    t0 = time.time()
    raw = mne.io.read_raw_fif(fif, preload=True, verbose="error")
    raw.pick("eeg")
    raw.filter(l_freq=1.0, h_freq=100.0, verbose="error")
    raw.set_eeg_reference("average", verbose="error")
    rank = mne.compute_rank(raw, rank=None, verbose="error")["eeg"]
    n = int(round(rank / 5))
    ica = mne.preprocessing.ICA(n_components=n, method="infomax", fit_params=dict(extended=True),
                                max_iter="auto", random_state=SEED, verbose="error")
    ica.fit(raw, verbose="error")
    ica.save(CACHE / f"{subj}_{sess}-ica.fif", overwrite=True, verbose="error")
    t_fit = time.time() - t0
    proba = iclabel_label_components(raw, ica, backend="onnx")
    src = ica.get_sources(raw).get_data()
    meas = caracas(src, raw.info["sfreq"])
    keys = list(meas[0])
    np.savez(out, proba=proba, classes=np.array(CLASSES), rank=rank, ncomp=n, n_iter=ica.n_iter_,
             dur_s=raw.times[-1], fit_s=t_fit, meas_keys=np.array(keys),
             meas=np.array([[float(m[k]) for k in keys] for m in meas]))
    return f"{subj} {sess}: rank {rank}, {n} ICs, {ica.n_iter_} iter, fit {t_fit:.0f}s, total {time.time() - t0:.0f}s"


def safe_run(*a):
    try:
        return run(*a)
    except Exception:
        return f"{a[0]} {a[1]}: FAILED\n{traceback.format_exc()}"


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = sessions()
    if len(sys.argv) > 2:
        todo = [s for s in todo if s[0] == sys.argv[1] and s[1] == sys.argv[2]]
    workers = int(os.environ.get("ICA_WORKERS", "6"))
    print(f"{len(todo)} sessions, {workers} workers", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for f in as_completed([ex.submit(safe_run, *s) for s in todo]):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
