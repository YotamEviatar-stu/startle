"""Item #2 evidence: what `no_inspiration` (n_peaks==0) and `lost_lock` (n_peaks>=K)
actually fire on, cohort-wide, from the Layer-1 cache.

    /Users/yotameviatar/startle-1/.venv/bin/python Airflow/_scratch/probe_peak_gates.py

Mirrors run_glm_scoring Pass A exactly (QC_ENABLED path included). Read-only.
"""

import os
import sys
import pickle
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/Users/yotameviatar/startle-1")

from Airflow import airflow_config as config          # noqa: E402
from Airflow import airflow_glm as glm                # noqa: E402
from Airflow import airflow_qc as qc                  # noqa: E402


def load_cache():
    path = os.path.join(config.OUTPUT_DIR, "_cache", "airflow_cache.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)["sessions"]


def session_cycles(sess, subj, sess_key):
    signal_raw = np.asarray(sess["signal_raw"], dtype=float)
    native_sfreq = float(sess["sfreq"])
    env_min, env_max = sess.get("env_min"), sess.get("env_max")

    if getattr(config, "QC_ENABLED", False):
        q = qc.qc_session(signal_raw, native_sfreq, config)
        signal_for_pipeline = q["signal_clean"]
        if env_min is not None and env_max is not None:
            env_min, env_max = np.array(env_min), np.array(env_max)
            for s0, s1 in q["masked_spans"]:
                j0, j1 = int(round(s0 * native_sfreq)), int(round(s1 * native_sfreq))
                env_min[j0:j1] = signal_for_pipeline[j0:j1]
                env_max[j0:j1] = signal_for_pipeline[j0:j1]
    else:
        signal_for_pipeline = signal_raw

    raw_z, sfreq = glm.phase1_filter_downsample(
        signal_for_pipeline, native_sfreq,
        target_sfreq=config.GLM_TARGET_SFREQ,
        bp_low=config.GLM_CYCLE_BANDPASS[0], bp_high=config.GLM_CYCLE_BANDPASS[1],
        zscore=config.GLM_ZSCORE_RAW,
        despike=(not getattr(config, "QC_ENABLED", False))
                and getattr(config, "GLM_DESPIKE_ENABLED", True),
        despike_k=getattr(config, "GLM_DESPIKE_K", 50.0),
        despike_max_run_sec=getattr(config, "GLM_DESPIKE_MAX_RUN_SEC", 1.0),
    )

    cycles = glm.detect_cycles(
        raw_z, sfreq, signal_for_pipeline, native_sfreq,
        median_win_sec=config.GLM_MEDIAN_WIN_SEC,
        refractory_sec=config.GLM_REFRACTORY_SEC,
        env_min=env_min, env_max=env_max,
    )
    pk, _ = glm._nk2_peak_trough_times(raw_z, sfreq, config)
    glm.attach_cycle_features(cycles, signal_for_pipeline, native_sfreq, pk)

    manual = [(float(a), float(b)) for a, b in
              getattr(config, "MANUAL_BAD_SPANS", {}).get(f"{subj}/{sess_key}", [])]
    if manual:
        cycles = [c for c in cycles
                  if not any(c["onset_time"] < b and c["assign_time"] > a for a, b in manual)]

    rejected, reason, report = qc.classify_cycles(cycles, config)
    return cycles, rejected, reason, report, len(raw_z) / sfreq


def main():
    cache = load_cache()
    excl = getattr(config, "SUBJECTS_EXCLUDE", {})
    min_pk = getattr(config, "CYCLE_LOSTLOCK_MIN_PEAKS", 3)

    per_sess = []
    hits = defaultdict(list)
    npeak_hist = defaultdict(int)
    n_cycles_total = 0
    dur_total = 0.0

    for subj in sorted(cache):
        if subj in excl:
            continue
        for sess_key, sess in sorted(cache[subj].items()):
            if f"{subj}/{sess_key}" in excl:
                continue
            cycles, rejected, reason, report, sess_sec = session_cycles(sess, subj, sess_key)
            n_cycles_total += len(cycles)
            dur_total += sess_sec
            counts = {g: 0 for g in ("no_inspiration", "lost_lock")}
            secs = {g: 0.0 for g in ("no_inspiration", "lost_lock")}
            for c, r in zip(cycles, reason):
                npeak_hist[int(c.get("n_peaks", -1))] += 1
                if r in counts:
                    counts[r] += 1
                    secs[r] += c["RP"]
                    hits[r].append((subj, sess_key, c["onset_time"], c["RP"],
                                    c["RA"], int(c.get("n_peaks", -1))))
            per_sess.append((subj, sess_key, len(cycles), sess_sec, counts, secs,
                             report["gate_counts"]))

    print(f"\n=== cohort: {len(per_sess)} sessions, {n_cycles_total} cycles, "
          f"{dur_total/60:.1f} min ===")
    print(f"CYCLE_LOSTLOCK_MIN_PEAKS = {min_pk}\n")

    print("--- n_peaks distribution over ALL cycles ---")
    for k in sorted(npeak_hist):
        n = npeak_hist[k]
        print(f"  n_peaks={k:>3}: {n:>6}  ({100*n/n_cycles_total:5.2f}%)")

    for g in ("no_inspiration", "lost_lock"):
        rows = hits[g]
        tot_sec = sum(r[3] for r in rows)
        n_sess_firing = len({(r[0], r[1]) for r in rows})
        print(f"\n--- {g}: {len(rows)} cycles / {tot_sec:.1f} s "
              f"({100*len(rows)/n_cycles_total:.2f}% of cycles, "
              f"{100*tot_sec/dur_total:.2f}% of time) "
              f"in {n_sess_firing}/{len(per_sess)} sessions ---")
        per = defaultdict(int)
        for r in rows:
            per[(r[0], r[1])] += 1
        if per:
            v = np.array(list(per.values()))
            print(f"    per firing session: median {np.median(v):.0f}, "
                  f"max {v.max()} ({max(per, key=per.get)})")
        print("    worst 12 by RP:")
        for r in sorted(rows, key=lambda x: -x[3])[:12]:
            print(f"      {r[0]}/{r[1]:<4} t={r[2]:8.2f}s  RP={r[3]:6.2f}s  "
                  f"RA={r[4]:9.2f}  n_peaks={r[5]}")

    print("\n--- per-session gate counts (all gates) ---")
    for subj, sk, ncyc, sec, counts, secs, gc in per_sess:
        print(f"  {subj}/{sk:<4} ncyc={ncyc:>4} {sec/60:6.1f}min  "
              f"no_insp={counts['no_inspiration']:>3} ({secs['no_inspiration']:6.1f}s)  "
              f"lost_lock={counts['lost_lock']:>3} ({secs['lost_lock']:6.1f}s)   {gc}")


if __name__ == "__main__":
    main()
