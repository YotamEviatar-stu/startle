#!/usr/bin/env python
"""
Smoke test + cache inspector for the HR pipeline.

Usage (run from /Users/yotameviatar/startle-1):
  .venv/bin/python HR/.claude/skills/run-hr/smoke.py [--full]

Without --full: verifies cache health + analysis params (~300 ms, no disk writes).
With    --full: also runs the complete pipeline and saves all plots (~2 min).
"""

import os, sys, time, pickle

REPO  = os.path.expanduser("~/startle-1")
CACHE = os.path.expanduser("~/Desktop/spo2_output/_cache/spo2_cache.pkl")

sys.path.insert(0, REPO)
from HR import hr_config as config, hr_processor as processor

# ── 1. Cache health ───────────────────────────────────────────────────────────

print("=== HR cache health ===")
assert os.path.exists(CACHE), f"Cache not found: {CACHE}"

t0 = time.time()
with open(CACHE, "rb") as f:
    data = pickle.load(f)
load_ms = (time.time() - t0) * 1000

trials = data.get("trials", {})
traces = data.get("traces", {})

assert trials, "Cache has no 'trials' key — may need FORCE_RELOAD"
assert traces, "Cache has no 'traces' key — may need FORCE_RELOAD"

n_subjs  = len(trials)
n_trials = sum(len(tl) for s in trials.values() for tl in s.values())
cache_mb = os.path.getsize(CACHE) / 1e6

print(f"  Loaded in {load_ms:.0f} ms  ({cache_mb:.1f} MB)")
print(f"  Subjects: {n_subjs}  ({sorted(trials.keys())})")
print(f"  Trials:   {n_trials}")

sample_subj  = sorted(trials.keys())[0]
sample_sess  = sorted(trials[sample_subj].keys())[0]
t0_trial     = trials[sample_subj][sample_sess][0]
assert "epoch_wide" in t0_trial, "Missing epoch_wide in cached trial"
assert "times_wide" in t0_trial, "Missing times_wide in cached trial"
assert "sfreq"      in t0_trial, "Missing sfreq in cached trial"
ep_sfreq = t0_trial["sfreq"]
ep_dur   = len(t0_trial["epoch_wide"]) / ep_sfreq
print(f"  Epoch:    {t0_trial['epoch_wide'].shape}  @{ep_sfreq:.0f} Hz  ({ep_dur:.1f} s)")

sample_trace = traces[sample_subj][sample_sess]
tr_sfreq = sample_trace["sfreq"]
tr_dur   = len(sample_trace["signal"]) / tr_sfreq / 60
print(f"  Trace:    {len(sample_trace['signal'])} samps  @{tr_sfreq:.0f} Hz  ({tr_dur:.1f} min)")

# ── 2. apply_analysis_params ─────────────────────────────────────────────────

print("\n=== apply_analysis_params ===")
t0 = time.time()
processor.apply_analysis_params(trials, config)
proc_ms = (time.time() - t0) * 1000
print(f"  Ran in {proc_ms:.0f} ms")

n_acc = sum(1 for s in trials.values() for tl in s.values() for t in tl if not t["rejected"])
n_rej = n_trials - n_acc
print(f"  Accepted: {n_acc}/{n_trials}  ({n_rej} rejected = {100*n_rej/n_trials:.1f}%)")

t_check = trials[sample_subj][sample_sess][1]
for key in ("baseline_mean", "score", "rejected", "epoch_anal", "times_anal"):
    assert t_check[key] is not None or t_check["rejected"], \
        f"Trial key '{key}' is None on an accepted trial"
bl  = t_check["baseline_mean"]
sc  = t_check["score"] if not t_check["rejected"] else float("nan")
print(f"  {sample_subj} {sample_sess} trial-1: baseline={bl:.1f} bpm  score={sc:.2f}%  rejected={t_check['rejected']}")

score_max = getattr(config, "SCORE_MAX_PCT", None)
if score_max is not None:
    import math
    outliers = [t for s in trials.values() for tl in s.values()
                for t in tl if not t["rejected"] and not math.isnan(t["score"])
                and abs(t["score"]) > score_max]
    assert not outliers, f"{len(outliers)} accepted trials exceed SCORE_MAX_PCT={score_max}"
    print(f"  SCORE_MAX_PCT={score_max}%  ✓")

print("\n=== Smoke PASSED ===")

# ── 3. Optional full pipeline run ─────────────────────────────────────────────

if "--full" in sys.argv:
    print("\n=== Full pipeline run ===")
    import importlib
    from HR import hr_main
    importlib.reload(hr_main)
    hr_main.main()
    print("=== Full run DONE ===")
