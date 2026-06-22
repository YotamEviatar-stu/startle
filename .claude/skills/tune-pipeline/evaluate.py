#!/usr/bin/env python
"""
Pipeline evaluator — reports key metrics for HR or Airflow pipeline.

Usage (run from /Users/yotameviatar/vs_code):
  .venv/bin/python .claude/skills/tune-pipeline/evaluate.py hr
  .venv/bin/python .claude/skills/tune-pipeline/evaluate.py airflow
"""

import sys
import os
import pickle
import math

import numpy as np
from scipy import stats

REPO = os.path.expanduser("~/vs_code")
sys.path.insert(0, REPO)

pipeline = sys.argv[1].lower() if len(sys.argv) > 1 else "hr"

if pipeline == "hr":
    from HR import hr_config as config, hr_processor as processor
    cache_path = os.path.join(config.OUTPUT_DIR, "_cache", "hr_cache.pkl")
    label_min = config.HR_MIN_BPM
    label_max = config.HR_MAX_BPM
    score_unit = "% change from baseline"
    print(f"=== HR Pipeline Evaluation ===")
    print(f"Config:  SCORE_TMIN={config.SCORE_TMIN}  SCORE_TMAX={config.SCORE_TMAX}")
    print(f"         BASELINE_TMIN={config.BASELINE_TMIN}  BASELINE_TMAX={config.BASELINE_TMAX}")
    print(f"         SCORE_MAX_PCT={getattr(config,'SCORE_MAX_PCT',None)}  "
          f"HR_Z_SCORE_THRESHOLD={getattr(config,'HR_Z_SCORE_THRESHOLD',None)}")
elif pipeline == "airflow":
    from Airflow import airflow_config as config, airflow_processor as processor
    cache_path = os.path.join(config.OUTPUT_DIR, "_cache", "airflow_cache.pkl")
    label_min = None
    label_max = None
    score_unit = "gasp ratio"
    print(f"=== Airflow Pipeline Evaluation ===")
    print(f"Config:  RESPONSE_TMIN={config.RESPONSE_TMIN}  RESPONSE_TMAX={config.RESPONSE_TMAX}")
    print(f"         BASELINE_TMIN={config.BASELINE_TMIN}  BASELINE_TMAX={config.BASELINE_TMAX}")
    print(f"         AIRFLOW_Z_SCORE_THRESHOLD={getattr(config,'AIRFLOW_Z_SCORE_THRESHOLD',None)}  "
          f"AIRFLOW_SCORE_MAX={getattr(config,'AIRFLOW_SCORE_MAX',None)}")
else:
    print(f"Unknown pipeline: {pipeline}. Use 'hr' or 'airflow'.")
    sys.exit(1)

# ── Load cache ────────────────────────────────────────────────────────────────
assert os.path.exists(cache_path), f"Cache not found: {cache_path}\nRun the pipeline once first."
with open(cache_path, "rb") as f:
    data = pickle.load(f)
trials_cache = data["trials"]

processor.apply_analysis_params(trials_cache, config)

# ── Compute metrics ───────────────────────────────────────────────────────────
conds = [("eve", 1), ("eve", 2), ("mor", 1), ("mor", 2)]
labels_map = {("eve",1): "Eve-Neg", ("eve",2): "Eve-Neu",
              ("mor",1): "Mor-Neg", ("mor",2): "Mor-Neu"}

n_total   = sum(len(tl) for s in trials_cache.values() for tl in s.values())
n_rejected = sum(1 for s in trials_cache.values() for tl in s.values()
                 for t in tl if t["rejected"])
print(f"\nSubjects: {len(trials_cache)}   Trials: {n_total}   "
      f"Rejected: {n_rejected}/{n_total} ({100*n_rejected/n_total:.1f}%)")

# Per-condition subject means
subj_means = {}
for (sk, cond) in conds:
    vals = []
    n_subj = 0
    n_trials = 0
    for subj, sessions in trials_cache.items():
        s_vals = [t["score"] for t in sessions.get(sk, [])
                  if not t["rejected"] and t["label"] == cond
                  and t["score"] is not None and not math.isnan(t["score"])]
        if s_vals:
            vals.append(np.mean(s_vals))
            n_subj += 1
            n_trials += len(s_vals)
    subj_means[(sk, cond)] = vals
    mean_str = f"{np.mean(vals):.3f} ± {np.std(vals):.3f}" if vals else "—"
    print(f"  {labels_map[(sk,cond)]:<10}  {mean_str}  ({n_subj} subjects, {n_trials} trials)")

# Eve vs Mor overall
print()
eve_means, mor_means = [], []
for subj, sessions in trials_cache.items():
    ev = [t["score"] for t in sessions.get("eve", [])
          if not t["rejected"] and t["score"] is not None and not math.isnan(t["score"])]
    mo = [t["score"] for t in sessions.get("mor", [])
          if not t["rejected"] and t["score"] is not None and not math.isnan(t["score"])]
    if ev and mo:
        eve_means.append(np.mean(ev))
        mor_means.append(np.mean(mo))

if len(eve_means) >= 3:
    try:
        _, p_em = stats.wilcoxon(eve_means, mor_means)
        test_em = "Wilcoxon"
    except Exception:
        _, p_em = stats.ttest_rel(eve_means, mor_means)
        test_em = "Paired-t"
    sig = "***" if p_em < 0.001 else ("**" if p_em < 0.01 else ("*" if p_em < 0.05 else "ns"))
    direction = "Eve > Mor ✓" if np.mean(eve_means) > np.mean(mor_means) else "Mor > Eve ✗"
    print(f"Eve vs Mor:  Eve={np.mean(eve_means):.3f}  Mor={np.mean(mor_means):.3f}  "
          f"{test_em} p={p_em:.4f} {sig}  [{direction}]")
else:
    print("Eve vs Mor:  not enough subjects")

# Neg vs Neu overall
neg_means, neu_means = [], []
for subj, sessions in trials_cache.items():
    ng = [t["score"] for sk in ["eve","mor"] for t in sessions.get(sk, [])
          if not t["rejected"] and t["label"] == 1
          and t["score"] is not None and not math.isnan(t["score"])]
    nu = [t["score"] for sk in ["eve","mor"] for t in sessions.get(sk, [])
          if not t["rejected"] and t["label"] == 2
          and t["score"] is not None and not math.isnan(t["score"])]
    if ng and nu:
        neg_means.append(np.mean(ng))
        neu_means.append(np.mean(nu))

if len(neg_means) >= 3:
    try:
        _, p_nn = stats.wilcoxon(neg_means, neu_means)
        test_nn = "Wilcoxon"
    except Exception:
        _, p_nn = stats.ttest_rel(neg_means, neu_means)
        test_nn = "Paired-t"
    sig_nn = "***" if p_nn < 0.001 else ("**" if p_nn < 0.01 else ("*" if p_nn < 0.05 else "ns"))
    print(f"Neg vs Neu:  Neg={np.mean(neg_means):.3f}  Neu={np.mean(neu_means):.3f}  "
          f"{test_nn} p={p_nn:.4f} {sig_nn}")

# Session × condition breakdown
print()
em_neg = subj_means[("eve",1)]; em_neu = subj_means[("eve",2)]
mo_neg = subj_means[("mor",1)]; mo_neu = subj_means[("mor",2)]
if all(len(x) >= 3 for x in [em_neg, em_neu, mo_neg, mo_neu]):
    valid = [s for s in trials_cache
             if all((sk,c) in {k: v for k,v in [
                 (("eve",1), subj_means[("eve",1)]),
                 (("eve",2), subj_means[("eve",2)]),
                 (("mor",1), subj_means[("mor",1)]),
                 (("mor",2), subj_means[("mor",2)])]}.items() for sk,c in [("eve",1),("eve",2),("mor",1),("mor",2)])]
    # Simple: compare paired means where all 4 conditions available
    n = min(len(em_neg), len(em_neu), len(mo_neg), len(mo_neu))
    eve_all = [(a+b)/2 for a,b in zip(em_neg[:n], em_neu[:n])]
    mor_all = [(a+b)/2 for a,b in zip(mo_neg[:n], mo_neu[:n])]
    delta = np.mean(eve_all) - np.mean(mor_all)
    print(f"Eve-Mor delta (paired, all conditions): {delta:+.3f}  "
          f"({'Eve higher ✓' if delta > 0 else 'Mor higher ✗'})")
