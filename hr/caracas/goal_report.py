import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caracas.run import list_sessions

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "_cache" / "goal"
LOG = HERE.parent / "_out_caracas" / "goal_log.md"
CRIT = ("sk", "ku", "RR", "Rampl", "bpm")

DOCS = {
    "c1_readme": dict(
        method="Extended Infomax, PCA-reduced to round(rank/5) components (CARACAS README example), "
               "1–100 Hz FIR band-pass, average reference, 250 Hz",
        citation="CARACAS README example `pop_runica(...,'extended',1,'pca',round(dataRank/5))` "
                 "(github.com/PierreChampetier/Cardiac_IC_labelling; SASICA CARACAS); extended Infomax: Lee, Girolami & "
                 "Sejnowski 1999 Neural Comput 11:417, impl. MNE `ICA(method='infomax', extended=True)`; 1 Hz high-pass: "
                 "Winkler et al. 2015 IEEE EMBC; 1–100 Hz + average reference: ICLabel input spec (Pion-Tonachini et "
                 "al. 2019 NeuroImage 198:181; mne-icalabel docs); fs 250 Hz > 200 Hz required by heart_peak_detect "
                 "lpfreq 100 Hz",
        params="mne 1.8 `read_raw_egi` → pick EEG → `resample(250, 'polyphase')` → `filter(1, 100)` (MNE default "
               "firwin) → `set_eeg_reference('average')` → rank = `compute_rank` → `ICA(n_components=round(rank/5), "
               "method='infomax', extended=True, max_iter='auto', random_state=97)`; CARACAS = `eeg_SASICA` with "
               "`SASICA('getdefs')` CARACAS defaults (sk≥1.0, ku≥5.3, RR≤0.31, Rampl≤0.23, bpm 34–97), all else disabled",
    ),
    "c2_fullrank_picard": dict(
        method="Full-rank ICA (no PCA reduction, n_components = data rank), Picard fitting the extended-Infomax "
               "model; preprocessing as c1",
        citation="No PCA reduction: Artoni, Delorme & Makeig 2018 NeuroImage 175:176 (PCA reduction degrades ICA "
                 "decomposition); Picard: Ablin, Cardoso & Gramfort 2018 IEEE TSP 66:4040, impl. python-picard 0.8.2 "
                 "via MNE `ICA(method='picard', fit_params=dict(ortho=False, extended=True))` (MNE docs: equivalent "
                 "model to extended Infomax, faster convergence); filters/reference/fs as c1",
        params="as c1 except `ICA(n_components=rank, method='picard', fit_params=dict(ortho=False, extended=True), "
               "max_iter='auto', random_state=97)`; CARACAS unchanged",
    ),
}


def session_stems(tag):
    by_session = {}
    try:
        for subject, session, mff in list_sessions():
            by_session.setdefault(f"{subject}/{session}", []).append(f"{subject}_{session}__{mff.stem}")
        return by_session, False
    except FileNotFoundError:
        for f in sorted((CACHE / tag).glob("*__*")):
            if f.suffix in (".npz", ".failed"):
                key = "/".join(f.stem.split("__")[0].split("_"))
                by_session.setdefault(key, []).append(f.stem)
        return by_session, True


def session_rows(tag):
    by_session, partial = session_stems(tag)
    rows = []
    for key, stems in by_session.items():
        row = dict(session=key, passed=False, ics=[], heart=[], note="")
        notes = []
        for stem in stems:
            npz, mat = CACHE / tag / f"{stem}.npz", CACHE / tag / "caracas" / f"{stem}.mat"
            if (CACHE / tag / f"{stem}.failed").exists():
                last = (CACHE / tag / f"{stem}.failed").read_text().strip().splitlines()[-1]
                notes.append(f"{stem.split('__')[1]}: ICA error `{last[:80]}`")
                continue
            if not npz.exists() or not mat.exists():
                notes.append(f"{stem.split('__')[1]}: missing" if not (CACHE / tag / "caracas" / f"{stem}.caracas_failed").exists()
                             else f"{stem.split('__')[1]}: CARACAS error")
                continue
            r, m = np.load(npz), loadmat(mat)
            heart = r["proba"][:, 3]
            is_c = m["is_cardiac"].ravel().astype(bool)
            if is_c.any():
                row["passed"] = True
                row["ics"] += [int(i) for i in np.flatnonzero(is_c)]
                row["heart"] += [float(heart[i]) for i in np.flatnonzero(is_c)]
            else:
                nc = m["NotCardiac"].astype(bool)
                b = int(heart.argmax())
                fails = ",".join(c for c, f in zip(CRIT, nc[b]) if f)
                notes.append(f"top-ICLabel IC{b} (Heart {heart[b]:.2f}) fails {fails}: bpm {m['bpm'].ravel()[b]:.1f}, "
                             f"sk {m['sk'].ravel()[b]:.2f}, ku {m['ku'].ravel()[b]:.1f}, "
                             f"RR {m['RR'].ravel()[b]:.2f}, Rampl {m['Rampl'].ravel()[b]:.2f}")
        row["note"] = "; ".join(notes)
        rows.append(row)
    return rows, partial


def append_log(tag, title=None):
    rows, partial = session_rows(tag)
    n = len(rows)
    k = sum(r["passed"] for r in rows)
    hp = [h for r in rows for h in r["heart"]]
    d = DOCS[tag]
    lines = [f"\n## {title or tag} — {date.today()}\n",
             *(["- **PARTIAL:** raw-data drive unmounted; only sessions processed so far are counted"] if partial else []),
             f"- **Method:** {d['method']}",
             f"- **Citation:** {d['citation']}",
             f"- **Parameters:** {d['params']}",
             f"- **Pass:** {k}/{n} = {100 * k / n:.1f}%",
             f"- **Median ICLabel Heart p of passing ICs:** {np.median(hp):.3f} (n = {len(hp)} ICs)" if hp else
             "- **Median ICLabel Heart p of passing ICs:** n/a",
             "", "| session | pass | CARACAS IC(s) | ICLabel Heart p | note |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['session']} | {'PASS' if r['passed'] else 'fail'} | "
                     f"{', '.join(map(str, r['ics']))} | {', '.join(f'{h:.2f}' for h in r['heart'])} | {r['note']} |")
    if not LOG.exists():
        LOG.write_text("# Goal log — cardiac IC in ≥75% of sessions\n\nPass criterion (fixed): `eeg_SASICA` CARACAS "
                       "block with `SASICA('getdefs')` defaults selects ≥1 IC. ICLabel Heart p logged, not required. "
                       "IC indices are 0-based.\n")
    with open(LOG, "a") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{tag}: {k}/{n} = {100 * k / n:.1f}%")


if __name__ == "__main__":
    append_log(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
