import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caracas.run import list_sessions

CACHE = Path(__file__).resolve().parents[1] / "_cache" / "iclabel"
OUT = Path(__file__).resolve().parents[1] / "_out" / "iclabel"


def plot_session(subject, session, mff, t0=300.0, dur=10.0):
    d = np.load(CACHE / subject / f"{mff.stem}.npz")
    ica = mne.preprocessing.read_ica(CACHE / subject / f"{mff.stem}-ica.fif", verbose="error")
    raw = mne.io.read_raw_egi(mff, preload=False, verbose="error").pick("eeg")
    proba, heart, sfreq = d["proba"], int(d["heart_idx"]), float(d["sfreq"])
    classes = list(d["classes"])

    fig = plt.figure(figsize=(13, 9), layout="constrained")
    gs = fig.add_gridspec(4, 4)
    ax = fig.add_subplot(gs[0, :3])
    hp = proba[:, classes.index("heart beat")]
    ax.bar(np.arange(len(hp)), hp, color=["#C0392B" if i == heart else "#95A5A6" for i in range(len(hp))])
    for i in range(len(hp)):
        ax.text(i, hp[i] + 0.01, classes[int(np.argmax(proba[i]))].split()[0], ha="center", fontsize=7, rotation=90, va="bottom")
    ax.set_xticks(np.arange(len(hp)), [f"IC{i}" for i in range(len(hp))], fontsize=8)
    ax.set_ylabel("ICLabel P(heart beat)")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"{subject}/{session} — heart prob per IC (label above = argmax class)")
    axt = fig.add_subplot(gs[0, 3])
    ica.plot_components(picks=[heart], axes=axt, show=False, colorbar=False)
    axt.set_title(f"IC{heart} topography")

    sig = d["signal"]
    i0, i1 = int(t0 * sfreq), int((t0 + dur) * sfreq)
    a = fig.add_subplot(gs[1, :])
    a.plot(np.arange(i0, i1) / sfreq, sig[i0:i1], lw=0.7, color="#C0392B")
    a.set_title(f"IC{heart}  P(heart)={hp[heart]:.2f} — raw source, {dur:.0f} s crop")
    a.set_xlabel("time (s)")
    n = 4000
    blk = len(sig) // n
    env = sig[: n * blk].reshape(n, blk)
    tb = (np.arange(n) * blk + blk / 2) / sfreq / 60
    a2 = fig.add_subplot(gs[2:, :])
    a2.fill_between(tb, env.min(1), env.max(1), color="#C0392B", lw=0)
    a2.set_title(f"IC{heart} whole session, min/max envelope ({len(sig)/sfreq/60:.1f} min)")
    a2.set_xlabel("time (min)")
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{subject}_{session}.png"
    fig.savefig(png, dpi=110)
    return png


if __name__ == "__main__":
    for subject, session, mff in list_sessions():
        if f"{subject}/{session}" in sys.argv[1:]:
            print(plot_session(subject, session, mff))
