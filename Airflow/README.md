# Airflow — Gasp-Ratio Pipeline

Scores startle-evoked respiratory responses from the Airflow channel using the **gasp ratio**: the ratio of post-startle breath amplitude to pre-startle baseline amplitude, detected via NeuroKit2 peak/trough analysis.

This is a standalone pipeline — it has no dependency on the RESP package.

## Files

| File | Role |
|------|------|
| `airflow_config.py` | All tunable parameters — paths, windows, triggers, scoring flags. |
| `airflow_processor.py` | File discovery, event extraction, signal processing, and gasp ratio scoring. |
| `airflow_main.py` | Orchestration: iterates subjects/sessions, saves CSV, generates review plots. |

---

## How to run

From the **repo root**:

```bash
.venv/bin/python -m Airflow.airflow_main
```

Output lands in `~/Desktop/airflow_output/`.

---

## Key parameters (`airflow_config.py`)

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `SUBJECT_FILTER` | `[]` | Subjects to process; `[]` = all |
| `AIRFLOW_CHANNEL` | `"Airflow"` | EEG channel name |
| `BASELINE_TMIN/TMAX` | `-5.0, 0.0 s` | Pre-startle window for baseline amplitude |
| `RESPONSE_TMIN/TMAX` | `0.5, 5.0 s` | Post-startle window for gasp detection |
| `ANAL_TMIN/TMAX` | `-5.0, 10.0 s` | Trimmed epoch stored per trial |
| `USE_SUBJECTIVE_TRIAL_TYPE` | `True` | Neg/Neu classification method |

---

## Output structure

```
~/Desktop/airflow_output/
  airflow_trial_scores.csv          ← per-trial scores (subject, session, label, score, rejected)
  review_trials/
    <SubjectID>/<session>/
      trial_001_OK.png
      trial_002_REJ.png
      ...
```

---

## Scoring method

For each startle trial:

1. NeuroKit2 (`nk.rsp_process`) cleans the full-session Airflow signal and detects breath peaks and troughs.
2. **Baseline amplitude** = mean(peaks) − mean(troughs) in `[BASELINE_TMIN, BASELINE_TMAX]`. Falls back to max−min if no peaks/troughs detected.
3. **Response amplitude** = first peak − first trough in `[RESPONSE_TMIN, RESPONSE_TMAX]`. Same fallback.
4. **Gasp ratio** = `|response amplitude / baseline amplitude|`.

A trial is rejected if the baseline mean is NaN (epoch out of bounds).
