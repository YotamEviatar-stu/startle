import os
import sys
import time
import traceback
from pathlib import Path

import mne
import numpy as np
from mne.preprocessing import ICA
from mne_icalabel.iclabel import iclabel_label_components
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caracas.run import list_sessions

CACHE = Path(__file__).resolve().parents[1] / "_cache" / "goal"
SCRATCH = Path(os.environ["GOAL_SCRATCH"])
CLASSES = ("brain", "muscle artifact", "eye blink", "heart beat", "line noise", "channel noise", "other")

CONFIGS = {
    "c1_readme": dict(sfreq=250.0, l_freq=1.0, h_freq=100.0, method="infomax",
                      fit_params=dict(extended=True), ncomp="rank/5"),
    "c2_fullrank_picard": dict(sfreq=250.0, l_freq=1.0, h_freq=100.0, method="picard",
                               fit_params=dict(ortho=False, extended=True), ncomp="rank"),
}


def n_components(rule, rank):
    if rule == "rank/5":
        return int(round(rank / 5))
    if rule == "rank":
        return int(rank)
    return int(rule)


def preprocess(mff, cfg):
    raw = mne.io.read_raw_egi(mff, preload=True, verbose="error")
    raw.pick("eeg")
    raw.resample(cfg["sfreq"], method="polyphase", verbose="error")
    raw.filter(l_freq=cfg["l_freq"], h_freq=cfg["h_freq"], verbose="error")
    raw.set_eeg_reference("average", verbose="error")
    return raw


def run_session(tag, subject, session, mff, max_pending=6):
    cfg = CONFIGS[tag]
    out_dir = CACHE / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = SCRATCH / tag
    src_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{subject}_{session}__{mff.stem}"
    res = out_dir / f"{stem}.npz"
    if res.exists():
        return
    while len(list(src_dir.glob("*.mat"))) >= max_pending:
        time.sleep(20)
    t0 = time.time()
    raw = preprocess(mff, cfg)
    rank = mne.compute_rank(raw, rank=None, verbose="error")["eeg"]
    n = n_components(cfg["ncomp"], rank)
    ica = ICA(n_components=n, method=cfg["method"], fit_params=cfg["fit_params"],
              max_iter="auto", random_state=97, verbose="error")
    ica.fit(raw, verbose="error")
    proba = iclabel_label_components(raw, ica, backend="onnx")
    src = ica.get_sources(raw).get_data().astype(np.float32)
    tmp = src_dir / f"{stem}.mat.part"
    unmixing = ica.unmixing_matrix_ @ ica.pca_components_[:n] / ica.pre_whitener_.T
    pos = np.array([ch["loc"][:3] for ch in raw.info["chs"]])
    savemat(tmp, dict(src=src, fs=raw.info["sfreq"], mixing=ica.get_components(), unmixing=unmixing,
                      chlabel=np.array(raw.ch_names, dtype=object), chpos=pos),
            do_compression=False, format="5")
    tmp.rename(src_dir / f"{stem}.mat")
    np.savez(res, subject=subject, session=session, mff=mff.name, tag=tag, rank=rank, ncomp=n,
             n_iter=getattr(ica, "n_iter_", -1), proba=proba, classes=np.array(CLASSES),
             dur_s=raw.times[-1], secs=time.time() - t0)
    print(f"{stem}: rank {rank} ncomp {n} iter {getattr(ica, 'n_iter_', -1)} {time.time() - t0:.0f}s", flush=True)


def local_sessions(mff_dir):
    out = []
    for mff in sorted(Path(mff_dir).glob("*.mff")):
        parts = mff.stem.split("_")
        out.append((parts[0], parts[1], mff))
    return out


def main(tag, worker=0, nworkers=1, only=None):
    sessions = local_sessions(os.environ["GOAL_MFF_DIR"]) if "GOAL_MFF_DIR" in os.environ else list_sessions()
    for i, (subject, session, mff) in enumerate(sessions):
        if i % nworkers != worker:
            continue
        if only and f"{subject}/{session}" not in only:
            continue
        try:
            run_session(tag, subject, session, mff)
        except Exception:
            print(f"{subject}/{session} {mff.name} FAILED\n{traceback.format_exc()}", flush=True)
            (CACHE / tag / f"{subject}_{session}__{mff.stem}.failed").write_text(traceback.format_exc())


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), set(sys.argv[4:]) or None)
