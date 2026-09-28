#!/usr/bin/env python
"""
baseline.py — Red-team baseline for the HR and Airflow startle pipelines.

Loads the EXISTING cache pickle, runs Layer 2 (apply_analysis_params) in memory,
and prints the diagnostics a peer reviewer needs to GROUND a finding in real
numbers instead of speculation:

  - per-condition n accepted / n total and mean score
  - per-subject Evening vs Morning means + Wilcoxon signed-rank p (the actual claim)
  - rejection breakdown overall, split by session (Eve vs Mor) AND by label (Neg vs Neu)
  - the asymmetry ratio that exposes condition-correlated rejection — the single
    most important hidden-flaw detector, because a gate that is individually
    justified on every trial can still bias the Eve>Mor comparison if it fires
    more often in one condition than the other.

It NEVER sets FORCE_RELOAD and NEVER reads an MFF. It only loads the pickle and
runs the fast in-memory Layer 2. If the pickle is missing it says so and exits —
it does not rebuild. This is the guarantee that makes the skill safe to run.

Usage (run from the repo root):
    PYTHONPATH=<repo> <repo>/.venv/bin/python baseline.py airflow
    PYTHONPATH=<repo> <repo>/.venv/bin/python baseline.py hr
"""
import sys
import os
import pickle
import importlib

import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

PIPELINES = {
    "airflow": ("Airflow.airflow_config", "Airflow.airflow_amp_processor", "airflow_cache.pkl"),
    "hr":      ("HR.hr_config",           "HR.hr_processor",           "hr_cache.pkl"),
}

LABEL_NAME = {1: "Neg", 2: "Neu"}


def _accepted_score(t):
    """Return the trial score if the trial is accepted and the score is finite, else None."""
    if t.get("rejected"):
        return None
    s = t.get("score")
    if s is None:
        return None
    try:
        s = float(s)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(s):
        return None
    return s


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in PIPELINES:
        print("usage: baseline.py [airflow|hr]")
        sys.exit(1)

    name = sys.argv[1]
    cfg_mod, proc_mod, cache_name = PIPELINES[name]
    cfg = importlib.import_module(cfg_mod)
    proc = importlib.import_module(proc_mod)

    cache_file = os.path.join(cfg.OUTPUT_DIR, "_cache", cache_name)
    if not os.path.exists(cache_file):
        print(f"[!] No cache at {cache_file}")
        print("    Run the pipeline once to build it. NOT rebuilding from MFF here.")
        sys.exit(2)

    print(f"# Red-team baseline — {name.upper()}")
    print(f"# cache: {cache_file}")
    print(f"# Layer 2 in-memory only — no MFF reload, FORCE_RELOAD ignored.\n")

    with open(cache_file, "rb") as f:
        cache = pickle.load(f)
    sessions = cache.get("sessions", cache)

    trials_data, _ = proc.apply_analysis_params(sessions, cfg)

    # ── Per-condition means ──────────────────────────────────────────────────
    conds = {("eve", 1): [], ("eve", 2): [], ("mor", 1): [], ("mor", 2): []}
    cond_total = {k: 0 for k in conds}
    for subj, sess_dict in trials_data.items():
        for sk, trials in sess_dict.items():
            for t in trials:
                lbl = t.get("label")
                key = (sk, lbl)
                if key not in conds:
                    continue
                cond_total[key] += 1
                s = _accepted_score(t)
                if s is not None:
                    conds[key].append(s)

    print("## Per-condition (accepted / total, mean score)")
    for sk in ("eve", "mor"):
        for lbl in (1, 2):
            sc = conds[(sk, lbl)]
            tot = cond_total[(sk, lbl)]
            label = f"{'Eve' if sk == 'eve' else 'Mor'}-{LABEL_NAME[lbl]}"
            mean = f"{np.mean(sc):+.4f}" if sc else "  n/a "
            print(f"  {label:9s}: {len(sc):3d}/{tot:3d} accepted   mean={mean}")
    print()

    # ── Per-subject Eve vs Mor (collapsed Neg+Neu), Wilcoxon ─────────────────
    eve_means, mor_means, paired_subj = [], [], []
    for subj, sess_dict in sorted(trials_data.items()):
        e = [s for t in sess_dict.get("eve", []) if (s := _accepted_score(t)) is not None]
        m = [s for t in sess_dict.get("mor", []) if (s := _accepted_score(t)) is not None]
        if e and m:
            eve_means.append(np.mean(e))
            mor_means.append(np.mean(m))
            paired_subj.append(subj)

    print("## Overall Evening vs Morning (per-subject means, collapsed Neg+Neu)")
    n = len(paired_subj)
    print(f"  paired subjects (both sessions usable): {n}")
    if n:
        eve_arr, mor_arr = np.array(eve_means), np.array(mor_means)
        direction = "Eve>Mor" if eve_arr.mean() > mor_arr.mean() else "Mor>Eve"
        print(f"  Eve mean={eve_arr.mean():+.4f}   Mor mean={mor_arr.mean():+.4f}   ->  {direction}")
        n_eve_bigger = int(np.sum(eve_arr > mor_arr))
        print(f"  subjects with Eve>Mor: {n_eve_bigger}/{n}")
        if wilcoxon is not None and n >= 1:
            try:
                stat, p = wilcoxon(eve_arr, mor_arr)
                sig = "*" if p < 0.05 else "ns"
                print(f"  Wilcoxon signed-rank: W={stat:.1f}  p={p:.4f}  {sig}")
            except Exception as exc:
                print(f"  Wilcoxon failed: {exc}")
    print()

    # ── Rejection breakdown — overall, by session, by label ──────────────────
    # This is the asymmetry detector. A gate is only neutral if it rejects Eve and
    # Mor (and Neg and Neu) at the same rate. Unequal rates bias the comparison.
    by_reason_session = {}   # reason -> {"eve": n, "mor": n}
    by_reason_label = {}     # reason -> {1: n, 2: n}
    n_trials = {"eve": 0, "mor": 0, 1: 0, 2: 0}
    n_rejected = {"eve": 0, "mor": 0, 1: 0, 2: 0}

    for subj, sess_dict in trials_data.items():
        for sk, trials in sess_dict.items():
            for t in trials:
                lbl = t.get("label")
                if sk in n_trials:
                    n_trials[sk] += 1
                if lbl in n_trials:
                    n_trials[lbl] += 1
                if t.get("rejected"):
                    if sk in n_rejected:
                        n_rejected[sk] += 1
                    if lbl in n_rejected:
                        n_rejected[lbl] += 1
                    for r in (t.get("rejection_reason") or "").split(", "):
                        r = r.strip()
                        if not r:
                            continue
                        bs = by_reason_session.setdefault(r, {"eve": 0, "mor": 0})
                        if sk in bs:
                            bs[sk] += 1
                        bl = by_reason_label.setdefault(r, {1: 0, 2: 0})
                        if lbl in bl:
                            bl[lbl] += 1

    print("## Rejection rate by session")
    for sk in ("eve", "mor"):
        tot = n_trials[sk]
        rej = n_rejected[sk]
        pct = (100.0 * rej / tot) if tot else 0.0
        print(f"  {'Eve' if sk=='eve' else 'Mor'}: {rej}/{tot} rejected ({pct:.1f}%)")
    # asymmetry flag
    e_tot, m_tot = n_trials["eve"], n_trials["mor"]
    e_pct = (n_rejected["eve"] / e_tot) if e_tot else 0
    m_pct = (n_rejected["mor"] / m_tot) if m_tot else 0
    if max(e_pct, m_pct) > 0:
        ratio = max(e_pct, m_pct) / max(min(e_pct, m_pct), 1e-9)
        flag = "  <-- ASYMMETRIC, investigate" if ratio >= 1.5 else ""
        print(f"  session rejection asymmetry ratio: {ratio:.2f}x{flag}")
    print()

    print("## Rejection reasons (count split by session, then by label)")
    print(f"  {'reason':22s} {'Eve':>5s} {'Mor':>5s}   {'Neg':>5s} {'Neu':>5s}")
    for r in sorted(set(by_reason_session) | set(by_reason_label)):
        s = by_reason_session.get(r, {"eve": 0, "mor": 0})
        l = by_reason_label.get(r, {1: 0, 2: 0})
        flag = ""
        # flag a reason whose Eve/Mor or Neg/Neu split is lopsided
        for a, b in ((s["eve"], s["mor"]), (l[1], l[2])):
            if a + b >= 6 and max(a, b) >= 3 * max(min(a, b), 1):
                flag = "  <-- lopsided"
        print(f"  {r:22s} {s['eve']:5d} {s['mor']:5d}   {l[1]:5d} {l[2]:5d}{flag}")
    print()
    print("# Done. Every empirical finding in the review must trace back to numbers above")
    print("# (or to a targeted probe you run the same way: load pickle -> Layer 2 -> count).")


if __name__ == "__main__":
    main()
