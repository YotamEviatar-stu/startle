import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne_icalabel.iclabel import iclabel_label_components

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hr import config as cfg

SUBJ, SESS = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("AB22", "eve")
NCOMP = sys.argv[3] if len(sys.argv) > 3 else "rank/5"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out_ica_test")
SEED = 123
CLASSES = ("brain", "muscle", "eye", "heart", "line", "chan", "other")
N_SHOW = 8


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = mne.io.read_raw_fif(os.path.join(cfg.DATA_250_DIR, SUBJ, f"{SUBJ}_{SESS}_raw.fif"), preload=True, verbose="error")
    raw.pick("eeg")
    raw.filter(l_freq=1.0, h_freq=100.0, method="fir", fir_design="firwin", verbose="error")
    raw.set_eeg_reference("average", verbose="error")
    rank = mne.compute_rank(raw, rank=None, verbose="error")["eeg"]
    n_comp = rank if NCOMP == "rank" else round(rank / 5) if NCOMP == "rank/5" else int(NCOMP)
    tag = f"{SUBJ}_{SESS}_n{n_comp}"
    ica = mne.preprocessing.ICA(n_components=n_comp, method="infomax", fit_params=dict(extended=True),
                                random_state=SEED, max_iter="auto")
    t0 = time.time()
    ica.fit(raw, verbose="error")
    fit_s = time.time() - t0
    proba = iclabel_label_components(raw, ica, backend="onnx")
    ica.save(os.path.join(OUT_DIR, f"{tag}-ica.fif"), overwrite=True, verbose="error")
    np.save(os.path.join(OUT_DIR, f"{tag}_iclabel.npy"), proba)

    heart = proba[:, CLASSES.index("heart")]
    order = np.argsort(heart)[::-1][:N_SHOW]
    print(f"{tag}: rank {rank}, {n_comp} ICs, {ica.n_iter_} iter, fit {fit_s:.0f} s, "
          f"n heart>0.5: {(heart > 0.5).sum()}, top heart: "
          + ", ".join(f"IC{i} {heart[i]:.2f}" for i in order), flush=True)

    d102 = [o for o, d in zip(raw.annotations.onset, raw.annotations.description) if d == "D102"]
    t_start = d102[0] + 10 if d102 else 60.0
    src = ica.get_sources(raw).get_data(picks=order.tolist(), tmin=t_start, tmax=t_start + 10)
    t = t_start + np.arange(src.shape[1]) / raw.info["sfreq"]
    fig, axes = plt.subplots(N_SHOW, 1, figsize=(14, 1.1 * N_SHOW), sharex=True)
    for ax, i, s in zip(axes, order, src):
        ax.plot(t, s, "k", lw=0.5)
        ax.set_yticks([])
        top = np.argmax(proba[i])
        ax.set_ylabel(f"IC{i}\nheart {heart[i]:.2f}\n{CLASSES[top]} {proba[i, top]:.2f}",
                      rotation=0, ha="right", va="center", fontsize=7)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    axes[-1].set_xlabel("s")
    fig.suptitle(f"{SUBJ} {SESS}, {n_comp} ICs: top {N_SHOW} by ICLabel heart p, 10 s pre-task rest (raw, 250 Hz)", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{tag}_top_heart_sources.png"), dpi=90)

    fig = ica.plot_components(picks=order.tolist(), show=False, title=f"{tag} top heart-p ICs")
    fig.savefig(os.path.join(OUT_DIR, f"{tag}_top_heart_topos.png"), dpi=90)


if __name__ == "__main__":
    main()
