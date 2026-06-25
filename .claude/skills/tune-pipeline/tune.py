#!/usr/bin/env python
"""
Autonomous parameter tuner for HR or Airflow pipeline.

Both pipelines now share the same two-layer cache architecture:
  Layer 1 cache key: "sessions"  — raw signal + trigger samps + trial metadata
  Layer 2:           apply_analysis_params(sessions_cache, cfg) → (trials_data, traces)

No MFF reloading — each evaluation is fast (epoch cutting + analysis only).
Coordinate-descent finds the parameter set where Evening > Morning startle
reactivity is clearest, writes the final config to disk, and prints a change log.

Usage (run from /Users/yotameviatar/vs_code):
  .venv/bin/python .claude/skills/tune-pipeline/tune.py hr
  .venv/bin/python .claude/skills/tune-pipeline/tune.py airflow
"""

import sys, os, re, math, pickle
import numpy as np
from scipy import stats

REPO = os.path.expanduser("~/vs_code")
sys.path.insert(0, REPO)

pipeline = sys.argv[1].lower() if len(sys.argv) > 1 else "hr"

# ── Load pipeline ─────────────────────────────────────────────────────────────

if pipeline == "hr":
    import HR.hr_config as cfg
    from HR import hr_processor as processor
    cache_path  = os.path.join(cfg.OUTPUT_DIR, "_cache", "hr_cache.pkl")
    config_path = os.path.join(REPO, "HR", "hr_config.py")

    # Parameters searched in order of expected impact.
    # Each entry: (param_name, candidates, physiological reason)
    SEARCH = [
        ("SCORE_TMAX",
         [2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
         "HR startle peaks at 2–4s; longer windows capture post-response drift"),
        ("SCORE_TMIN",
         [0.0, 0.5, 1.0, 1.5, 2.0],
         "HR signal updates every ~2s; delay avoids the pre-response flat step"),
        ("BASELINE_TMIN",
         [-2.0, -3.0, -4.0, -5.0, -6.0],
         "Baseline window length; too long = slow drift contamination"),
        ("BASELINE_METHOD",
         ["median", "mean"],
         "Median is more robust to BPM spikes; mean may capture more signal"),
        # SCORE_METHOD is fixed at max_minus_baseline (canonical, matches EMG reference)
        # Mean would dilute the startle peak by averaging rise + plateau + recovery.
        ("WIDE_TMAX",
         [12.0, 15.0, 20.0],
         "Wider epoch gives more context; too wide increases edge artifact risk"),
        ("WIDE_TMIN",
         [-8.0, -10.0, -12.0],
         "Longer pre-stimulus window needed for stable baseline estimation"),
        ("SCORE_MAX_PCT",
         [30.0, 40.0, 50.0, 60.0, 80.0],
         "Outlier rejection on % change score; lower = stricter"),
        ("HR_Z_SCORE_THRESHOLD",
         [2.0, 2.5, 3.0, 3.5, 4.0],
         "Baseline variability gate; lower = stricter probe-stability requirement"),
        ("FLAT_MIN_SEC",
         [6.0, 7.0, 8.0, 10.0, 12.0],
         "Minimum frozen-BPM duration for probe dropout; normal blocks ≤7s"),
        ("SPIKE_THRESH_BPM",
         [15.0, 20.0, 25.0, 30.0],
         "BPM deviation from local block median flagged as spike; SpO2 steps are 1–4 BPM"),
        ("FLAT_BASELINE_CONTIGUOUS_MAX",
         [0.30, 0.40, 0.50, 0.60],
         "Max single artifact gap in baseline window before rejection"),
        ("FLAT_SCORE_CONTIGUOUS_MAX",
         [0.30, 0.40, 0.50, 0.60],
         "Max single artifact gap in score window before rejection"),
        ("FLAT_BASELINE_TOTAL_MAX",
         [0.30, 0.40, 0.50, 0.60],
         "Max total artifact fraction in baseline window before rejection"),
        ("FLAT_SCORE_TOTAL_MAX",
         [0.30, 0.40, 0.50, 0.60],
         "Max total artifact fraction in score window before rejection"),
    ]

elif pipeline == "airflow":
    import Airflow.airflow_config as cfg
    from Airflow import airflow_processor as processor
    cache_path  = os.path.join(cfg.OUTPUT_DIR, "_cache", "airflow_cache.pkl")
    config_path = os.path.join(REPO, "Airflow", "airflow_config.py")

    SEARCH = [
        ("RESPONSE_TMAX",
         [1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
         "Airflow gasp resolves within 2–4s; longer window captures post-gasp normalization"),
        ("RESPONSE_TMIN",
         [0.0, 0.3, 0.5, 1.0],
         "Gasp begins immediately post-startle; shift if onset is delayed"),
        ("BASELINE_TMIN",
         [-2.0, -3.0, -4.0, -5.0, -6.0],
         "Need ≥1 full breath cycle (~4s at 0.25 Hz) for stable baseline amplitude"),
        ("WIDE_TMAX",
         [12.0, 15.0, 20.0],
         "Wider epoch gives more post-startle context"),
        ("WIDE_TMIN",
         [-6.0, -8.0, -10.0],
         "Longer pre-stimulus window needed for stable baseline"),
        ("AIRFLOW_SCORE_MAX",
         [5.0, 8.0, 10.0, 15.0, 20.0],
         "Gasp ratio outlier gate; implausibly large ratios = signal artifact"),
        ("AIRFLOW_Z_SCORE_THRESHOLD",
         [2.0, 2.5, 3.0, 3.5, 4.0],
         "Baseline variability gate for breathing stability"),
        ("AIRFLOW_AMPLITUDE_Z_THRESHOLD",
         [3.0, 4.0, 5.0, 6.0],
         "Robust z-score gate on max epoch amplitude; catches movement artifacts"),
        # RSP_CLEAN_METHOD is expensive (re-runs NK2 on all sessions) — search last
        ("RSP_CLEAN_METHOD",
         ["khodadad2018", "biosppy"],
         "NK2 cleaning method; khodadad2018=Butterworth 0.05–3Hz; biosppy=alternate"),
    ]

else:
    print(f"Unknown pipeline '{pipeline}'. Use 'hr' or 'airflow'.")
    sys.exit(1)

# ── Load cache ────────────────────────────────────────────────────────────────

assert os.path.exists(cache_path), (
    f"Cache not found: {cache_path}\n"
    f"Run the pipeline first: .venv/bin/python -m "
    f"{'HR.hr_main' if pipeline == 'hr' else 'Airflow.airflow_main'}"
)
with open(cache_path, "rb") as f:
    data = pickle.load(f)

if "sessions" in data:
    sessions_cache = data["sessions"]
    cache_fmt = "sessions"
    n_total = sum(len(sess.get("trials_meta", []))
                  for subj in sessions_cache.values()
                  for sess in subj.values())
elif "trials" in data:
    sessions_cache = data["trials"]   # old format — processor handles transparently
    cache_fmt = "trials (legacy)"
    n_total = sum(len(tl) for subj in sessions_cache.values()
                  for tl in subj.values())
    # WIDE_TMIN/TMAX cannot be searched without raw signal — drop from HR search
    if pipeline == "hr":
        SEARCH = [s for s in SEARCH if s[0] not in ("WIDE_TMIN", "WIDE_TMAX")]
else:
    raise RuntimeError(f"Unknown cache format: keys={list(data.keys())}")

print(f"{'='*60}")
print(f"Pipeline: {pipeline.upper()}   Cache ({cache_fmt}): {n_total} trials across {len(sessions_cache)} subjects")
print(f"{'='*60}\n")


# ── Objective function ────────────────────────────────────────────────────────

def compute_metrics(sessions_cache, cfg_obj):
    """Apply current cfg_obj params via Layer 2 and return metrics dict."""
    trials_data, _ = processor.apply_analysis_params(sessions_cache, cfg_obj)

    n_rej = sum(1 for sess in trials_data.values()
                for tl in sess.values() for t in tl if t["rejected"])
    n_all = sum(len(tl) for sess in trials_data.values() for tl in sess.values())
    rej_rate = n_rej / n_all if n_all else 0.0

    eve_means, mor_means = [], []
    for sessions in trials_data.values():
        ev = [t["score"] for t in sessions.get("eve", [])
              if not t["rejected"] and t["score"] is not None and not math.isnan(t["score"])]
        mo = [t["score"] for t in sessions.get("mor", [])
              if not t["rejected"] and t["score"] is not None and not math.isnan(t["score"])]
        if ev and mo:
            eve_means.append(np.mean(ev))
            mor_means.append(np.mean(mo))

    n_pairs   = len(eve_means)
    eve_mean  = np.mean(eve_means) if eve_means else np.nan
    mor_mean  = np.mean(mor_means) if mor_means else np.nan
    direction = "Eve>Mor" if (not math.isnan(eve_mean) and eve_mean > mor_mean) else "Mor>Eve"
    p, test   = np.nan, "—"

    if n_pairs >= 3:
        try:
            _, p = stats.wilcoxon(eve_means, mor_means)
            test = "Wilcoxon"
        except Exception:
            _, p = stats.ttest_rel(eve_means, mor_means)
            test = "Paired-t"

    # Composite score: direction is required; then -log10(p); penalise extreme rejection
    if direction == "Mor>Eve" or n_pairs < 3:
        score = -50.0
    else:
        score = -math.log10(max(p if not math.isnan(p) else 0.5, 1e-10))
        if rej_rate > 0.30:
            score -= (rej_rate - 0.30) * 15
        if rej_rate < 0.02:
            score -= (0.02 - rej_rate) * 8

    sig = ("***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))) \
          if not math.isnan(p) else "—"

    return {
        "score":     score,
        "p":         p,
        "sig":       sig,
        "test":      test,
        "direction": direction,
        "eve_mean":  eve_mean,
        "mor_mean":  mor_mean,
        "rej_rate":  rej_rate,
        "n_pairs":   n_pairs,
    }


def fmt(m):
    p_str = f"p={m['p']:.4f} {m['sig']}" if not math.isnan(m['p']) else "p=—"
    return (f"{m['direction']}  {p_str}  "
            f"rej={m['rej_rate']*100:.1f}%  n={m['n_pairs']}  "
            f"(Eve={m['eve_mean']:.3f} Mor={m['mor_mean']:.3f})")


# ── Coordinate descent ────────────────────────────────────────────────────────

current = compute_metrics(sessions_cache, cfg)
print(f"Initial:  {fmt(current)}")
print(f"Params:   {', '.join(f'{p}={getattr(cfg, p, None)}' for p, *_ in SEARCH)}\n")

change_log = []

for param, candidates, range_reason in SEARCH:
    orig_val     = getattr(cfg, param, None)
    best_val     = orig_val
    best_score   = current["score"]
    best_metrics = current

    for val in candidates:
        if val == orig_val:
            continue
        setattr(cfg, param, val)
        m = compute_metrics(sessions_cache, cfg)
        if m["score"] > best_score:
            best_score   = m["score"]
            best_val     = val
            best_metrics = m

    setattr(cfg, param, best_val)  # lock in best (or restore original)

    if best_val != orig_val:
        if isinstance(orig_val, (int, float)) and isinstance(best_val, (int, float)):
            direction_word = "narrowed" if abs(best_val) < abs(orig_val) else "widened"
            reason = f"{range_reason} → {direction_word} {orig_val}→{best_val}"
        else:
            reason = f"{range_reason} → changed {orig_val!r}→{best_val!r}"
        change_log.append({
            "param":  param,
            "from":   orig_val,
            "to":     best_val,
            "reason": reason,
            "before": fmt(current),
            "after":  fmt(best_metrics),
        })
        current = best_metrics
        print(f"  ✓ {param}: {orig_val!r} → {best_val!r}")
        print(f"    Reason: {reason}")
        print(f"    Before: {change_log[-1]['before']}")
        print(f"    After:  {change_log[-1]['after']}\n")
    else:
        print(f"  — {param}: no improvement (kept {orig_val!r})")


# ── Write final config to disk ────────────────────────────────────────────────

def _write_param(content, param, val_str):
    """Write a single parameter value into config source, handling standalone
    and tuple assignments (e.g. WIDE_TMIN, WIDE_TMAX = -8.0, 15.0)."""
    esc = re.escape(param)
    # 1. Standalone:  PARAM = VALUE
    new = re.sub(rf'^({esc}\s*=\s*)\S+', rf'\g<1>{val_str}',
                 content, flags=re.MULTILINE)
    if new != content:
        return new
    # 2. Tuple, first position:  PARAM, OTHER = NEW_VAL, old_val
    new = re.sub(rf'^({esc}\s*,\s*\w+\s*=\s*)\S+', rf'\g<1>{val_str}',
                 content, flags=re.MULTILINE)
    if new != content:
        return new
    # 3. Tuple, second position:  OTHER, PARAM = old_val, NEW_VAL
    #    Capture everything up to and including the comma before the second value.
    new = re.sub(rf'^(\w+\s*,\s*{esc}\s*=\s*\S+\s*,\s*)\S+', rf'\g<1>{val_str}',
                 content, flags=re.MULTILINE)
    return new


if change_log:
    with open(config_path, "r") as f:
        content = f.read()
    for entry in change_log:
        param   = entry["param"]
        new_val = getattr(cfg, param)
        if isinstance(new_val, str):
            val_str = f'"{new_val}"'
        elif new_val is None:
            val_str = "None"
        else:
            val_str = str(new_val)
        content = _write_param(content, param, val_str)
    with open(config_path, "w") as f:
        f.write(content)
    print(f"Config written → {config_path}")


# ── Final report ──────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("TUNING COMPLETE")
print(f"{'='*60}")
print(f"Final:    {fmt(current)}\n")

if change_log:
    print(f"Changes ({len(change_log)} total):")
    for i, c in enumerate(change_log, 1):
        print(f"  {i}. {c['param']}: {c['from']!r} → {c['to']!r}")
        print(f"     {c['reason']}")
        print(f"     {c['before']}")
        print(f"     → {c['after']}")
        print()
else:
    print("No changes — initial parameters were already optimal for this dataset.")

p_final   = current["p"]
sig_final = current["sig"]
if not math.isnan(p_final) and p_final < 0.05 and current["direction"] == "Eve>Mor":
    print(f"✓ Target achieved: Eve > Mor  {p_final:.4f} {sig_final}")
elif current["direction"] == "Eve>Mor":
    print(f"~ Eve > Mor direction correct but p={p_final:.4f} — review timecourse plots.")
else:
    print(f"✗ Eve > Mor not achieved. The signal may not show a robust session effect.")

print(f"\nNext: regenerate plots with the new parameters:")
print(f"  .venv/bin/python -m {'HR.hr_main' if pipeline == 'hr' else 'Airflow.airflow_main'}")
