import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hr.ica_label_all import CACHE, DATA, sessions

OUT = Path(__file__).resolve().parent / "_out_ica_all"
HEART = 3


def table():
    rows = []
    for subj, sess, _ in sessions():
        f = CACHE / f"{subj}_{sess}.npz"
        if not f.exists():
            rows.append(dict(session=f"{subj}_{sess}", status="missing"))
            continue
        d = np.load(f)
        p, keys, m = d["proba"], list(d["meas_keys"]), d["meas"]
        col = {k: m[:, i] for i, k in enumerate(keys)}
        icl = np.flatnonzero(p.argmax(1) == HEART)
        car = np.flatnonzero(col["cardiac"] == 1)
        best = int(p[:, HEART].argmax())
        rows.append(dict(
            session=f"{subj}_{sess}", status="ok", ncomp=int(d["ncomp"]), n_iter=int(d["n_iter"]),
            iclabel="yes" if icl.size else "no", iclabel_ics=" ".join(map(str, icl)),
            iclabel_best_ic=best, iclabel_best_p=round(float(p[best, HEART]), 3),
            caracas="yes" if car.size else "no", caracas_ics=" ".join(map(str, car)),
            caracas_bpm=" ".join(f"{col['bpm'][i]:.1f}" for i in car),
            caracas_sk=" ".join(f"{col['sk'][i]:.2f}" for i in car),
            caracas_RR=" ".join(f"{col['RR'][i]:.3f}" for i in car),
            caracas_p_heart=" ".join(f"{p[i, HEART]:.2f}" for i in car),
            same_ic=bool(set(icl) & set(car)),
        ))
    return pd.DataFrame(rows)


def sources(subj, sess, fif):
    raw = mne.io.read_raw_fif(fif, preload=True, verbose="error")
    raw.pick("eeg")
    raw.filter(l_freq=1.0, h_freq=100.0, verbose="error")
    raw.set_eeg_reference("average", verbose="error")
    ica = mne.preprocessing.read_ica(CACHE / f"{subj}_{sess}-ica.fif", verbose="error")
    d102 = [o for o, a in zip(raw.annotations.onset, raw.annotations.description) if a == "D102"]
    t0 = d102[0] + 10 if d102 else 60.0
    src = ica.get_sources(raw).get_data(tmin=t0, tmax=t0 + 10)
    return t0 + np.arange(src.shape[1]) / raw.info["sfreq"], src


def plot(df):
    fifs = {f"{s}_{e}": f for s, e, f in sessions()}
    show = df[(df.iclabel == "yes") | (df.caracas == "yes")]
    for _, r in show.iterrows():
        subj, sess = r.session.split("_")
        who = {}
        for name, col in (("ICLabel", r.iclabel_ics), ("CARACAS", r.caracas_ics)):
            for ic in map(int, col.split()):
                who.setdefault(ic, []).append(name)
        ics = [(ic, " + ".join(w)) for ic, w in sorted(who.items())]
        t, src = sources(subj, sess, fifs[r.session])
        fig, axes = plt.subplots(len(ics), 1, figsize=(14, 1.6 * len(ics) + 0.6), sharex=True, squeeze=False)
        for ax, (ic, who) in zip(axes[:, 0], ics):
            ax.plot(t, src[ic], "k", lw=0.6)
            ax.set_yticks([])
            ax.set_ylabel(f"IC{ic:02d}\n{who}", rotation=0, ha="right", va="center", fontsize=8)
        axes[-1, 0].set_xlabel("s")
        fig.suptitle(f"{r.session}: ICLabel {r.iclabel}, CARACAS {r.caracas} — 10 s of pre-task rest, raw IC")
        fig.tight_layout()
        fig.savefig(OUT / f"{r.session}_flagged.png", dpi=80)
        plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    df = table()
    df.to_csv(OUT / "labels.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(df.drop(columns=["status"]).to_string(index=False))
    ok = df[df.status == "ok"]
    print()
    print(pd.crosstab(ok.iclabel, ok.caracas, rownames=["ICLabel"], colnames=["CARACAS"], margins=True))
    if "--plot" in sys.argv:
        plot(ok)


if __name__ == "__main__":
    main()
