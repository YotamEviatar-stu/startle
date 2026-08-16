"""Item #5: does mean/SD vs a robust centre/scale still differ on the series as it
actually reaches zscore_subject_series -- i.e. AFTER filtering, cycle rejection and
NaN blanking? If rejection already removed the extremes, the two agree and #5 is moot.

    /Users/yotameviatar/startle-1/.venv/bin/python Airflow/_scratch/probe_zscore.py

Read-only. Reproduces run_glm_scoring Pass A per subject.
"""

import sys
import numpy as np

sys.path.insert(0, "/Users/yotameviatar/startle-1")
sys.path.insert(0, "/Users/yotameviatar/startle-1/Airflow/_scratch")

from Airflow import airflow_config as config          # noqa: E402
from Airflow import airflow_glm as glm                # noqa: E402
from Airflow import airflow_qc as qc                  # noqa: E402
from probe_peak_gates import load_cache, session_cycles  # noqa: E402

METRIC = "RA"


def main():
    cache = load_cache()
    excl = getattr(config, "SUBJECTS_EXCLUDE", {})
    rows = []

    for subj in sorted(cache):
        if subj in excl:
            continue
        pooled = []
        for sess_key, sess in sorted(cache[subj].items()):
            if f"{subj}/{sess_key}" in excl:
                continue
            cycles, rejected, reason, report, _ = session_cycles(sess, subj, sess_key)
            rejected, reason = qc.apply_neighbour_rules(rejected, reason, config)

            native_sfreq = float(sess["sfreq"])
            n_native = len(sess["signal_raw"])
            sfreq = config.GLM_TARGET_SFREQ
            n_samples = int(round(n_native / native_sfreq * sfreq))

            manual = [(float(a), float(b)) for a, b in
                      getattr(config, "MANUAL_BAD_SPANS", {}).get(f"{subj}/{sess_key}", [])]
            series, _ = glm.build_continuous_series(
                cycles, n_samples, sfreq,
                hp=config.GLM_FINAL_HP, lp=config.GLM_FINAL_LP,
                missing_spans=manual, rejected=rejected)
            y = series[METRIC]
            pooled.append(y[np.isfinite(y)])

        if not pooled:
            continue
        v = np.concatenate(pooled)
        if v.size < 2:
            continue
        mu, sd = float(np.mean(v)), float(np.std(v))
        med, rsd = float(np.median(v)), float(qc.robust_sd(v))
        rows.append((subj, v.size, mu, med, sd, rsd,
                     sd / rsd if rsd > 0 else np.nan,
                     abs(mu - med) / rsd if rsd > 0 else np.nan))

    print(f"\nmetric={METRIC}  (series AFTER filter + cycle rejection + NaN blanking)")
    print(f"{'subj':<7}{'n':>8}{'mean':>11}{'median':>11}{'SD':>11}{'robustSD':>11}"
          f"{'SD/rSD':>9}{'|mu-med|/rSD':>14}")
    for r in rows:
        print(f"{r[0]:<7}{r[1]:>8}{r[2]:>11.4f}{r[3]:>11.4f}{r[4]:>11.4f}{r[5]:>11.4f}"
              f"{r[6]:>9.3f}{r[7]:>14.3f}")

    ratio = np.array([r[6] for r in rows])
    shift = np.array([r[7] for r in rows])
    print(f"\nSD / robustSD   : median {np.median(ratio):.3f}  "
          f"p90 {np.percentile(ratio,90):.3f}  max {ratio.max():.3f} "
          f"({rows[int(np.argmax(ratio))][0]})")
    print(f"|mean-median| in robust-SD units: median {np.median(shift):.3f}  "
          f"max {shift.max():.3f} ({rows[int(np.argmax(shift))][0]})")
    print(f"\nA ratio near 1.0 means the two estimators agree and #5 is moot;"
          f"\nlarge values mean outliers survive rejection and still inflate the scale.")


if __name__ == "__main__":
    main()
