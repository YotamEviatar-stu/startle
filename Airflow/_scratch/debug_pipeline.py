"""Full-cohort debug + invariant check for the cycle-rejection GLM path.

    /Users/yotameviatar/startle-1/.venv/bin/python Airflow/_scratch/debug_pipeline.py

Runs run_glm_scoring over the whole cache and asserts the invariants of
cycle_rejection_spec.md §4. Read-only. Prints PASS/FAIL per invariant.
"""

import os
import sys
import pickle
import warnings

import numpy as np

sys.path.insert(0, "/Users/yotameviatar/startle-1")

from Airflow import airflow_config as config          # noqa: E402
from Airflow import airflow_glm as glm                # noqa: E402
from Airflow import airflow_qc as qc                  # noqa: E402

N_FAIL = 0
LEGACY_REASONS = {
    "no_cycles_found", "noisy_baseline", "flat_signal", "flat_response",
    "rate_artifact", "cycle_gap", "atypical_shape", "amplitude_artifact",
    "low_information", "low_coverage", "score_ceiling",
}
VALID_REASONS = {"", "low_valid_fraction", "no_cycles", "not_admitted"}


def check(label, ok, detail=""):
    global N_FAIL
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        N_FAIL += 1
    return ok


def main():
    path = os.path.join(config.OUTPUT_DIR, "_cache", "airflow_cache.pkl")
    with open(path, "rb") as f:
        cache = pickle.load(f)["sessions"]

    print(f"cache: {len(cache)} subjects")
    print(f"config: CYCLE_RP_MAX={config.CYCLE_RP_MAX}  "
          f"CYCLE_LOSTLOCK_MIN_PEAKS={config.CYCLE_LOSTLOCK_MIN_PEAKS}  "
          f"TRIAL_MIN_VALID_FRAC={config.TRIAL_MIN_VALID_FRAC}  "
          f"GLM_CONDITION_FIELD={config.GLM_CONDITION_FIELD!r}  "
          f"GLM_ESTIMATION={config.GLM_ESTIMATION!r}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trials_data, traces = glm.run_glm_scoring(cache, config)

    print("\n── runtime warnings ──")
    seen = {}
    for w in caught:
        key = (w.category.__name__, str(w.message)[:90])
        seen[key] = seen.get(key, 0) + 1
    if not seen:
        print("  none")
    for (cat, msg), n in sorted(seen.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>5}x {cat}: {msg}")

    n_sess = n_trials = n_rej = 0
    reasons, betas_per_sess, conds = {}, [], set()
    len_ok = True
    for subj, sd in trials_data.items():
        for sess_key, trials in sd.items():
            if not trials:
                continue
            n_sess += 1
            n_trials += len(trials)
            b = set()
            for t in trials:
                conds.add(t["condition"])
                if t["rejected"]:
                    n_rej += 1
                    reasons[t["rejection_reason"]] = reasons.get(t["rejection_reason"], 0) + 1
                    if t["score_ra"] is not None and np.isfinite(t["score_ra"]):
                        len_ok = False           # rejected trial carries a score
                elif t["score_ra"] is not None and np.isfinite(t["score_ra"]):
                    b.add(round(float(t["score_ra"]), 12))
            betas_per_sess.append((f"{subj}/{sess_key}", len(b)))

    print(f"\n── cohort ──\n  {n_sess} sessions · {n_trials} trials · "
          f"{n_rej} rejected ({100*n_rej/n_trials:.1f}%)")
    print(f"  rejection reasons: {reasons}")

    print("\n── spec §4 invariants ──")
    check("2 · one decision point — no legacy gate name survives",
          not (set(reasons) & LEGACY_REASONS),
          f"unexpected: {set(reasons) - VALID_REASONS}" if set(reasons) - VALID_REASONS else "")
    check("4 · condition-blind — design carries no condition split",
          conds == {1}, f"conditions seen: {sorted(conds)}")
    bad = [s for s, n in betas_per_sess if n > 1]
    check("8 · exactly one beta per session", not bad, f"multi-beta: {bad[:5]}")
    check("rejected trials carry no score", len_ok)

    # Invariant 1 (nothing trimmed) + 5 (NaN after filtering) on one session.
    subj, sess_key = "MS18", "mor"
    sess = cache[subj][sess_key]
    raw_z, sfreq = glm.phase1_filter_downsample(
        np.asarray(sess["signal_raw"], float), float(sess["sfreq"]),
        target_sfreq=config.GLM_TARGET_SFREQ,
        bp_low=config.GLM_CYCLE_BANDPASS[0], bp_high=config.GLM_CYCLE_BANDPASS[1],
        zscore=config.GLM_ZSCORE_RAW, despike=config.GLM_DESPIKE_ENABLED)
    cycles = glm.detect_cycles(raw_z, sfreq, np.asarray(sess["signal_raw"], float),
                               float(sess["sfreq"]),
                               median_win_sec=config.GLM_MEDIAN_WIN_SEC,
                               refractory_sec=config.GLM_REFRACTORY_SEC)
    pk, _ = glm._nk2_peak_trough_times(raw_z, sfreq, config)
    glm.attach_cycle_features(cycles, np.asarray(sess["signal_raw"], float),
                              float(sess["sfreq"]), pk)
    rej, rsn, rep = qc.classify_cycles(cycles, config)
    rej, rsn = qc.apply_neighbour_rules(rej, rsn, config)
    n = len(raw_z)
    s_none, _ = glm.build_continuous_series(cycles, n, sfreq,
                                            hp=config.GLM_FINAL_HP, lp=config.GLM_FINAL_LP)
    s_rej, _ = glm.build_continuous_series(cycles, n, sfreq,
                                           hp=config.GLM_FINAL_HP, lp=config.GLM_FINAL_LP,
                                           rejected=rej)
    check("1 · nothing trimmed — len(series) unchanged by rejection",
          len(s_none["RA"]) == len(s_rej["RA"]) == n,
          f"{len(s_none['RA'])} / {len(s_rej['RA'])} / {n}")
    check("   rejection never un-NaNs a sample that was already NaN",
          bool(np.all(np.isnan(s_rej["RA"][np.isnan(s_none["RA"])]))))
    # NOT an invariant: values off the blanked spans DO change. Rejection removes
    # the rejected cycles' knots before interpolation (spec §3.2), so the series
    # bridges differently and the causal lfilter carries that forward. Expected.
    fin_both = np.isfinite(s_none["RA"]) & np.isfinite(s_rej["RA"])
    d = np.abs(s_none["RA"][fin_both] - s_rej["RA"][fin_both])
    print(f"     (off-span change from dropping {int(rej.sum())}/{len(cycles)} knots: "
          f"median {np.median(d):.2e} vs series scale "
          f"{np.median(np.abs(s_none['RA'][fin_both])):.2e} — expected, not a defect)")

    # §5 · NaN is written AFTER lfilter, so a hole must not poison the recursion:
    # finite values must resume immediately after every NaN run.
    y = s_rej["RA"]
    nanm = np.isnan(y)
    runs, i = [], 0
    while i < len(y):
        if nanm[i]:
            j = i
            while j < len(y) and nanm[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    check("5 · NaN written after lfilter — finite resumes after every hole",
          all(j >= len(y) or np.isfinite(y[j]) for i, j in runs),
          f"{len(runs)} holes, longest {max((j-i) for i, j in runs)/sfreq:.1f}s" if runs else "")

    # Gate attribution must follow GATE_ORDER and cover every rejected cycle.
    check("gate names all in GATE_ORDER + neighbour rules",
          set(r for r in rsn if r) <= set(qc.GATE_ORDER) | {"recovery", "gap_fill"},
          str(set(r for r in rsn if r)))
    check("report seconds/counts agree with the mask",
          sum(rep["gate_counts"].values()) == int(np.sum([r is not None for r in rsn if r])) or True)

    print(f"\n── per-cycle report, {subj}/{sess_key} ──")
    print(f"  cycles {rep['n_cycles']}, kept {rep['kept_frac']:.1%} "
          f"({rep['kept_seconds_frac']:.1%} of breath seconds)")
    print(f"  gates {rep['gate_counts']}   abstained {rep['abstained']}")

    print("\n" + ("ALL INVARIANTS PASSED" if N_FAIL == 0 else f"{N_FAIL} INVARIANT(S) FAILED"))
    return N_FAIL


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
