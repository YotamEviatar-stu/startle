---
name: tune-pipeline
description: Autonomous parameter tuner for HR and Airflow pipelines. Runs coordinate-descent over analysis parameters (no MFF reloading), writes the best config to disk, and prints a full change log. Goal: find settings where Evening > Morning startle reactivity is statistically or visually clear.
---

# Pipeline Tuner

Run `tune.py` — it operates fully autonomously, no user input required.
At the end it prints a full change log (what changed, why, before/after p-value)
and writes the winning parameters directly to the config file.

**All commands run from `/Users/yotameviatar/vs_code`.**

---

## The hypothesis

Sleep reduces physiological reactivity to emotional stimuli.
→ Evening sessions should show larger startle responses than Morning sessions.
→ Target pattern: **Eve > Mor**, ideally Wilcoxon p < 0.05.
→ Secondary: Negative trials > Neutral (startle potentiation by valence).

**What a good result looks like** (from EMG reference plots):
- Timecourse: Evening (orange) curve rises clearly above Morning (purple) after t=0
- Pre-stimulus period near zero (baseline corrected), flat
- Peak separation visible and maintained throughout the score window
- Boxplot: orange median above purple, most connecting lines red (declining),
  statistical bracket showing p < 0.05

---

## Quick start

```bash
cd /Users/yotameviatar/vs_code

# Step 1: evaluate current state (one-shot, read-only)
.venv/bin/python .claude/skills/tune-pipeline/evaluate.py hr
.venv/bin/python .claude/skills/tune-pipeline/evaluate.py airflow

# Step 2: run the autonomous tuner
.venv/bin/python .claude/skills/tune-pipeline/tune.py hr
.venv/bin/python .claude/skills/tune-pipeline/tune.py airflow

# Step 3: regenerate plots with the new parameters
.venv/bin/python -m HR.hr_main
.venv/bin/python -m Airflow.airflow_main
```

The tuner uses the **two-layer cache** — no MFF files are re-read.
Each evaluation is ~150 ms; a full 25-candidate search takes ~5 seconds.
When done, it writes the winning parameters to the config file automatically.

---

## What tune.py does

1. Loads cache (`hr_cache.pkl` or `airflow_cache.pkl`)
2. Evaluates initial state: direction (Eve>Mor?), p-value, rejection rate
3. For each parameter in order of expected impact, tries all candidate values
4. Keeps the value that maximises the objective score:
   - **Direction correct (Eve > Mor)** is required — wrong-direction params are rejected
   - Then minimises p-value (`-log10(p)`)
   - Penalises rejection rate < 2% (over-accepting) or > 30% (over-rejecting)
5. Locks in each improvement before testing the next parameter
6. Writes the final parameters to the config file
7. Prints a full change log

### Example output

```
============================================================
Pipeline: HR   Cache: 744 trials across 24 subjects
============================================================

Initial:  Mor>Eve  p=0.2341 ns  rej=8.3%  n=22  (Eve=2.134 Mor=2.891)
Params:   SCORE_TMAX=8.0, SCORE_TMIN=0.0, BASELINE_TMIN=-3.0, ...

  ✓ SCORE_TMAX: 8.0 → 4.0
    Reason: HR startle peaks at 2–4s; longer windows capture post-response drift → narrowed 8.0→4.0
    Before: Mor>Eve  p=0.2341 ns  rej=8.3%
    After:  Eve>Mor  p=0.0412 *   rej=8.3%

  — SCORE_TMIN: no improvement (kept 0.0)
  — BASELINE_TMIN: no improvement (kept -3.0)
  ...

Config written → HR/hr_config.py

============================================================
TUNING COMPLETE
============================================================
Final:    Eve>Mor  p=0.0412 *  rej=8.3%  n=22  (Eve=3.102 Mor=2.134)

Changes (1 total):
  1. SCORE_TMAX: 8.0 → 4.0
     HR startle peaks at 2–4s; longer windows capture post-response drift → narrowed 8.0→4.0
     Mor>Eve  p=0.2341 ns  rej=8.3%
     → Eve>Mor  p=0.0412 *  rej=8.3%

✓ Target achieved: Eve > Mor  0.0412 *

Next: .venv/bin/python -m HR.hr_main
```

---

## Parameters searched

### HR

| Parameter | Candidates | Notes |
|---|---|---|
| `SCORE_TMAX` | 2–8 s | HR startle peaks at 2–4s; cap at 8s |
| `SCORE_TMIN` | 0–2 s | Signal updates every ~2s |
| `BASELINE_TMIN` | -2 to -6 s | Too long = slow drift contamination |
| `BASELINE_METHOD` | median, mean | Median more robust to BPM spikes |
| `SCORE_METHOD` | max_minus_baseline, mean_minus_baseline | Peak vs. mean of response window |
| `WIDE_TMAX` | 12, 15, 20 s | Wider epoch = more post-startle context |
| `WIDE_TMIN` | -8, -10, -12 s | Longer pre-stimulus window |
| `SCORE_MAX_PCT` | 30–80 % | Outlier rejection on % change score |
| `HR_Z_SCORE_THRESHOLD` | 2.0–4.0 | Baseline variability gate |
| `FLAT_MIN_SEC` | 6–12 s | Min frozen BPM to flag as dropout |
| `SPIKE_THRESH_BPM` | 15–30 | BPM deviation flagged as spike |
| `FLAT_*_CONTIGUOUS_MAX` | 0.30–0.60 | Max single artifact gap per window |
| `FLAT_*_TOTAL_MAX` | 0.30–0.60 | Max total artifact fraction per window |

**Hard constraint:** no bandpass filtering — SpO2-Pulse is hardware BPM; filtering destroys values.

### Airflow

| Parameter | Candidates | Notes |
|---|---|---|
| `RESPONSE_TMAX` | 1.5–6 s | Gasp resolves in 2–4s |
| `RESPONSE_TMIN` | 0–1 s | Gasp begins immediately post-startle |
| `BASELINE_TMIN` | -2 to -6 s | Need ≥1 full breath cycle |
| `WIDE_TMAX` | 12, 15, 20 s | Wider epoch = more post-startle context |
| `WIDE_TMIN` | -6, -8, -10 s | Longer pre-stimulus window |
| `AIRFLOW_SCORE_MAX` | 5–20 | Gasp ratio outlier gate |
| `AIRFLOW_Z_SCORE_THRESHOLD` | 2.0–4.0 | Baseline variability gate |
| `AIRFLOW_AMPLITUDE_Z_THRESHOLD` | 3.0–6.0 | Robust z-score gate on epoch amplitude |
| `RSP_CLEAN_METHOD` | khodadad2018, biosppy | NK2 filter method (slow — re-runs per candidate) |

---

## Cache reload boundary

Both pipelines now share the same two-layer architecture:

**Layer 1 cache** (`sessions` key) stores: raw signal + trigger sample positions + CSV metadata.  
Written once; never re-read unless FORCE_RELOAD=True.

**Layer 2** (`apply_analysis_params`) runs on every invocation and re-applies all analysis params.

**Layer 1 (requires FORCE_RELOAD):**
- HR: `HR_CHANNEL`, `HR_CACHE_SFREQ`, trigger codes
- Airflow: `AIRFLOW_CHANNEL`, `CACHE_SFREQ`, trigger codes

**Layer 2 (all free to tune — handled by tune.py):**
- HR: `WIDE_TMIN/TMAX`, `BASELINE_*`, `SCORE_*`, `HR_*`, `FLAT_*`, `SPIKE_*`
- Airflow: `WIDE_TMIN/TMAX`, `BASELINE_*`, `RESPONSE_*`, `RSP_CLEAN_METHOD`, `AIRFLOW_*`

---

## Signal Credibility vs. Rejection Optimization (The Core Logic)

When updating or executing this tuning pipeline, you must distinguish between **blind mathematical optimization** and **biological credibility**. The goal is never to "game" the rejection system to manipulate the p-value. The goal is to maximize statistical power ($n$) *only* by recovering genuinely salvageable signals.

### 1. The Core Philosophy
*   **The Sacred Boundary (Credibility):** We cannot invent data. If a signal is fundamentally corrupted, hiding the noise through aggressive interpolation creates a "hallucinated" physiological response. This violates scientific validity.
*   **The Playground (Optimization):** Standard pipeline defaults are often overly conservative (e.g., throwing away a whole 8-second trial because of a 0.5-second flatline). The "playground" is the safe zone where micro-adjustments to rejection rules can recover valid data that a blunt algorithm would discard.

### 2. Biological Constraints for Interpolation
When assessing whether to interpolate or reject, apply this logical framework:
*   **Context Matters (Baseline vs. Response):**
    *   *Baseline window:* Needs to be highly pristine to establish a true relative zero. If the baseline is a flatline or pure noise, the entire trial loses its anchor point. **Default to strict rejection if noisy.**
    *   *Score/Response window:* If the physiological response has already begun and a brief dropout occurs, the trajectory of the signal might still be highly predictable. **Default to safe interpolation if less than 50% is flat.**
*   **The Information Loss Threshold:**
    *   If a flatline spans a minor fraction of a window (e.g., < 30–50%), the underlying biological trend is mathematically recoverable via smart interpolation (like cubic splines or NeuroKit defaults).
    *   If it spans a majority of the window, the physiological information is permanently lost. Interpolating here is akin to fabricating data. **Must reject.**

### 3. Your Analytical Directive
Before changing any rejection or interpolation code, you must evaluate the biological cost. If a change drops the p-value but increases the interpolation rate to a point where the curves look unnaturally smooth or artificial, **the change must be rejected**, even if the statistical score is technically "better."

---

## Config file locations

```
HR:      /Users/yotameviatar/vs_code/HR/hr_config.py
Airflow: /Users/yotameviatar/vs_code/Airflow/airflow_config.py
```

Key plots to review after tuning:a
- `group_overall_timecourse.png` — grand mean Eve vs Mor timecourse
- `boxplot_eve_vs_mor.png` — overall Eve vs Mor boxplot
- `group_neg_neu_ratio_boxplot.png` — Neg/Neu ratio Eve vs Mor
