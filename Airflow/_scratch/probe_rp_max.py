"""What CYCLE_RP_MAX costs, per value: cycles/seconds removed, and the
seconds-weighted Evening-vs-Morning balance REPORT (never a gate).

    /Users/yotameviatar/startle-1/.venv/bin/python Airflow/_scratch/probe_rp_max.py

Read-only. Condition labels are used here and ONLY here, after the fact.
"""

import sys
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, "/Users/yotameviatar/startle-1")
sys.path.insert(0, "/Users/yotameviatar/startle-1/Airflow/_scratch")

from Airflow import airflow_config as config          # noqa: E402
from Airflow import airflow_qc as qc                  # noqa: E402
from probe_peak_gates import load_cache, session_cycles  # noqa: E402

VALUES = [None, 12.0, 10.0, 8.0, 7.0, 6.0, 5.0]


def main():
    cache = load_cache()
    excl = getattr(config, "SUBJECTS_EXCLUDE", {})
    sessions = []

    for subj in sorted(cache):
        if subj in excl:
            continue
        for sess_key, sess in sorted(cache[subj].items()):
            if f"{subj}/{sess_key}" in excl:
                continue
            cycles, _, _, _, _ = session_cycles(sess, subj, sess_key)
            sessions.append((subj, sess_key, cycles))

    rp_all = np.array([c["RP"] for _, _, cy in sessions for c in cy])
    print(f"\n{len(sessions)} sessions, {len(rp_all)} cycles, "
          f"{rp_all.sum()/3600:.2f} h of breath time")
    print("RP percentiles: " + "  ".join(
        f"p{p}={np.percentile(rp_all, p):.2f}s" for p in (50, 90, 95, 99, 99.5, 99.9)))

    print(f"\n{'RP_MAX':>7} {'gate':>26} | {'Eve % sec':>10} {'Mor % sec':>10} "
          f"{'ratio':>6} {'MWU p':>9} | {'overall % sec':>13} {'kept cyc %':>11}")
    orig = config.CYCLE_RP_MAX
    for v in VALUES:
        config.CYCLE_RP_MAX = v
        per_cond = defaultdict(list)          # sess-level % of seconds removed by this gate
        n_gate, sec_gate, n_tot, sec_tot, n_kept = 0, 0.0, 0, 0.0, 0
        for subj, sess_key, cycles in sessions:
            rej, rsn, _ = qc.classify_cycles(cycles, config)
            rej, rsn = qc.apply_neighbour_rules(rej, rsn, config)
            tot = sum(c["RP"] for c in cycles)
            s = sum(c["RP"] for c, r in zip(cycles, rsn) if r == "rate_implausible")
            per_cond[sess_key].append(100.0 * s / tot if tot else np.nan)
            n_gate += sum(1 for r in rsn if r == "rate_implausible")
            sec_gate += s
            n_tot += len(cycles)
            sec_tot += tot
            n_kept += int((~np.asarray(rej, dtype=bool)).sum())

        eve = np.array(per_cond.get("eve", []), dtype=float)
        mor = np.array(per_cond.get("mor", []), dtype=float)
        if len(eve) > 1 and len(mor) > 1:
            p = mannwhitneyu(eve, mor, alternative="two-sided").pvalue
        else:
            p = np.nan
        e, m = np.nanmean(eve), np.nanmean(mor)
        lbl = "off" if v is None else f"{v:.0f}s"
        print(f"{lbl:>7} {f'{n_gate} cyc / {sec_gate:.0f}s':>26} | "
              f"{e:10.2f} {m:10.2f} {m/e if e else np.nan:6.2f} {p:9.4f} | "
              f"{100*sec_gate/sec_tot:13.2f} {100*n_kept/n_tot:11.2f}")
    config.CYCLE_RP_MAX = orig
    print("\n(balance is a REPORT: it exposes a condition-biased gate, it never passes"
          "\n or fails one, and it is not a tuning target.)")


if __name__ == "__main__":
    main()
