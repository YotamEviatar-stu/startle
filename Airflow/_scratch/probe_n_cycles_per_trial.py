import sys, collections
import numpy as np
sys.path.insert(0, "/Users/yotameviatar/startle-1")
sys.path.insert(0, "/Users/yotameviatar/startle-1/Airflow/_scratch")
from Airflow import airflow_config as config
import verify_spec_numbers as V

W = 12.0
cache = V.load_cache()
excl = set(getattr(config, "SUBJECTS_EXCLUDE", []) or [])
hist = collections.Counter()
singles = []
n_tr = 0
for subj in sorted(cache):
    if subj in excl:
        continue
    for sess in sorted(cache[subj]):
        if f"{subj}/{sess}" in excl:
            continue
        try:
            S = V.session_state(cache, subj, sess)
        except Exception as e:
            print("skip", subj, sess, e); continue
        for k, m in enumerate(cache[subj][sess]["trials_meta"], 1):
            ct = m.get("code_time")
            if ct is None:
                continue
            ov = V.overlapping(S, ct, W)
            n_tr += 1
            hist[len(ov)] += 1
            if len(ov) <= 2:
                rps = [round(c["RP"], 1) for _, c in ov]
                singles.append((subj, sess, k, round(ct, 1), len(ov), rps))
print(f"\ntrials scanned: {n_tr}   window W={W}s")
for n in sorted(hist):
    print(f"  {n} cycle(s) in window: {hist[n]:>5}  ({100*hist[n]/n_tr:5.2f}%)")
print(f"\ntrials with <=2 cycles ({len(singles)}):")
for row in singles[:40]:
    print("   ", row)
