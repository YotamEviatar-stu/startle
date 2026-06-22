---
name: tune-pipeline
description: Active parameter tuning skill for HR and Airflow pipelines. Evaluates current results, iterates config parameters using signal-specific physiological constraints, and reports each change with a one-sentence justification. Goal: find settings where Evening > Morning startle reactivity is statistically or visually clear.
---

# Pipeline Tuner

Active optimization loop. Evaluate → decide → change config → explain → re-evaluate.
Every change is shown to the user with a one-line reason before the next run.

**All commands run from `/Users/yotameviatar/vs_code`.**

---

## The hypothesis

Sleep reduces physiological reactivity to emotional stimuli.
→ Evening sessions should show larger startle responses than Morning sessions.
→ The target pattern: **Eve > Mor** across both Negative and Neutral conditions.
→ Secondary: Negative trials should show larger responses than Neutral (startle potentiation).

"Sweet spot" = parameter set where group boxplots and average timecourses show clear,
interpretable Eve > Mor separation, ideally with Wilcoxon p < 0.05.

---

## Step 1 — Evaluate current state

```bash
cd /Users/yotameviatar/vs_code
.venv/bin/python .claude/skills/tune-pipeline/evaluate.py hr
.venv/bin/python .claude/skills/tune-pipeline/evaluate.py airflow
```

Read the output. Key numbers to check:
- **Rejection rate**: ideally 5–20%. If > 30%, criteria are too strict.
- **Eve vs Mor direction**: Eve > Mor is the expected direction. If reversed, the window is capturing the wrong part of the signal.
- **p-value**: aim for < 0.05. If > 0.15, the window or baseline is likely misaligned.
- **Per-condition means**: all four should be in a physiologically plausible range.

---

## Step 2 — Decide what to change

Read the metrics, then apply the signal-specific rules below to decide one parameter
to change. Change one parameter at a time. Re-evaluate. Explain the decision.

**Report format for each change:**
```
Pipeline: HR
Changed:  SCORE_TMAX  6.0 → 4.0
Reason:   HR responses peak within 2–4 s of startle; the 6 s window was including
          post-response HR recovery which dilutes the peak signal.
Result:   Eve vs Mor p improved 0.12 → 0.04 *
```

---

## HR signal — what it is and valid ranges

**What it is**: SpO2-Pulse is a hardware-computed BPM output from the pulse oximeter.
It is NOT a raw PPG waveform. It updates approximately every 2 seconds (staircase signal).
No filtering should be applied — it would destroy the BPM values.

**Physiological response to startle**:
- Startle triggers a brief sympathetic surge → HR increases within 2–5 s.
- The response is transient — HR typically returns toward baseline within 8–10 s.
- Morning sessions: smaller and shorter HR increase expected (less arousal).

**Valid parameter ranges for HR:**

| Parameter | Valid range | Notes |
|---|---|---|
| `SCORE_TMIN` | 0.0 – 2.0 s | Start of response window. Don't start at t=0 — signal takes ~1–2 s to update. |
| `SCORE_TMAX` | 3.0 – 8.0 s | End of response window. >8 s captures post-response drift, not the startle peak. |
| `BASELINE_TMIN` | -6.0 – -2.0 s | Need at least 2–3 BPM readings (signal updates every 2 s). |
| `BASELINE_TMAX` | 0.0 (fixed) | Always end at trigger. |
| `HR_MIN_BPM` | 30–50 | Lower = keep more trials. Raise if seeing physiologically impossible baselines. |
| `HR_MAX_BPM` | 150–200 | Upper. 200 is already generous. |
| `HR_Z_SCORE_THRESHOLD` | 2.0 – 4.0 | Z-score on baseline variability. Tighten (lower) to reject unstable baselines. |
| `SCORE_MAX_PCT` | 20 – 80 | Outlier rejection. Start at 50. Tighten if large outliers dominate the boxplots. |

**What to try first if Eve vs Mor is not significant:**
1. Narrow `SCORE_TMAX` toward 4 s — most HR startle responses peak at 2–4 s
2. Widen baseline: try `BASELINE_TMIN = -4.0` (steady BPM window, not too long)
3. If too many rejections: loosen `SCORE_MAX_PCT` to 80 or `HR_Z_SCORE_THRESHOLD` to 4.0

**What NOT to do with HR:**
- Do not set `SCORE_TMAX > 10 s` — this captures HR drift unrelated to startle
- Do not set baseline window > 8 s — slow HR drift contaminates baseline mean
- Do not apply any filter — the signal is already BPM

---

## Airflow signal — what it is and valid ranges

**What it is**: Respiratory airflow/belt signal, cleaned by NeuroKit2 (`RSP_Clean`).
Peaks and troughs are detected by NeuroKit2 and stored per epoch.
Score = gasp ratio = abs(response amplitude / baseline amplitude).
Gasp ratio > 1 means the post-startle breath was larger than the baseline breath.

**Physiological response to startle**:
- Startle typically causes a brief respiratory pause or gasp.
- The gasp (if present) occurs within 0.5–3 s of the probe.
- Post-gasp: breathing normalizes within 5–8 s.
- Morning: smaller/shorter gasp expected.

**Valid parameter ranges for Airflow:**

| Parameter | Valid range | Notes |
|---|---|---|
| `RESPONSE_TMIN` | 0.0 – 1.0 s | Start of gasp window. Gasp begins immediately after startle. |
| `RESPONSE_TMAX` | 2.0 – 6.0 s | End of gasp window. >6 s captures post-gasp normalization, not the gasp. |
| `BASELINE_TMIN` | -6.0 – -2.0 s | Need at least 1–2 full breath cycles (~4–6 s at 0.2–0.3 Hz). |
| `BASELINE_TMAX` | 0.0 (fixed) | Always end at trigger. |
| `AIRFLOW_Z_SCORE_THRESHOLD` | 2.0 – 4.0 | Reject if baseline breathing variability is extreme. |
| `AIRFLOW_SCORE_MAX` | 5.0 – 20.0 | Reject if gasp ratio is implausibly large (signal artifact). |

**What to try first if Eve vs Mor is not significant:**
1. Narrow `RESPONSE_TMAX` toward 3 s — most gasps resolve within 3 s
2. Widen baseline toward -5 s — need stable baseline amplitude over full breaths
3. If rejection rate high: check `AIRFLOW_SCORE_MAX` (default 10); loosen if needed

**What NOT to do with Airflow:**
- Do not set `RESPONSE_TMAX > 8 s` — post-gasp normalization is not the signal of interest
- Do not set baseline < 2 s — won't capture a full breath cycle
- NeuroKit2 already handles cleaning — do not add additional filtering

---

## Step 3 — Apply the change

Edit the relevant config file:
- HR: `HR/hr_config.py`
- Airflow: `Airflow/airflow_config.py`

No cache reload needed for these parameters (both are Layer 2 — applied on every run):
- `SCORE_TMIN`, `SCORE_TMAX`, `RESPONSE_TMIN`, `RESPONSE_TMAX`
- `BASELINE_TMIN`, `BASELINE_TMAX`
- `HR_MIN/MAX_BPM`, `HR_Z_SCORE_THRESHOLD`, `SCORE_MAX_PCT`
- `AIRFLOW_Z_SCORE_THRESHOLD`, `AIRFLOW_SCORE_MAX`

Re-run `evaluate.py` immediately after and report the before/after numbers.

---

## Step 4 — Stopping criteria

Stop tuning when any of:
- Eve vs Mor Wilcoxon p < 0.05 and direction is Eve > Mor
- After 5 iterations with no improvement in p-value or direction
- Rejection rate drops below 5% (over-accepting) or rises above 35% (over-rejecting)

If after 5 iterations no improvement: report the best parameter set found and note
that the signal may not show a robust Eve > Mor effect in this dataset.

---

## Full pipeline run (to generate plots after finding good params)

```bash
cd /Users/yotameviatar/vs_code
.venv/bin/python -m HR.hr_main
.venv/bin/python -m Airflow.airflow_main
```

Key plots to check after tuning:
- `group_overall_timecourse.png` — grand mean Eve vs Mor timecourse
- `boxplot_eve_vs_mor.png` — overall Eve vs Mor boxplot
- `group_neg_neu_ratio_boxplot.png` — Neg/Neu ratio Eve vs Mor

---

## Config file locations

```
HR:      /Users/yotameviatar/vs_code/HR/hr_config.py
Airflow: /Users/yotameviatar/vs_code/Airflow/airflow_config.py
```
