# startle-airflow

Breath-by-breath analysis of the nasal-pressure (Airflow) channel recorded during the Startle
experiment. Each breath is detected with a Python port of BreathMetrics (Noto et al. 2018), and
each trial's response breaths are scored as % change from the session's rest breaths.

## Layout

| Path | What it is |
|---|---|
| `breathmetrics_py/final_pipeline.py` | The pipeline. Raw `.mff` → breaths → trial verdicts → change scores. |
| `breathmetrics_py/pipeline_config.py` | Data paths and the frozen session review: exclusions, rest-span overrides, manual trial rejects/accepts, bad spans. |
| `breathmetrics_py/*.py` (others) | BreathMetrics port: `load_mff`, `pressure` (pressure → flow), `breathmetrics` + `extrema`, `onsets_pauses`, `offsets`, `durations`, `volumes`, `features`, `erp`, `fft_smooth`. |
| `breathmetrics_py/PIPELINE_WALKTHROUGH.md` | Step-by-step description of what the code does. |
| `breathmetrics_py/SESSION_REVIEW_FINAL.md` | The by-eye review behind every value in `pipeline_config.py`. |
| `extras/` | Shared Startle helpers: file discovery (`emg_raw_potentiation.py`) and ratings-CSV loading (`trial_epochs.py`). |
| `figures/` | Final result figures and per-session overview PDFs. |
| `papers/` | Reference papers and notes. |

## Requirements

Python 3.12 and the packages in `requirements.txt`:

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Two inputs live outside the repo; their paths are set in `pipeline_config.py`:

- `RAW_DATA_DIR`: one folder per subject, each holding `EEG/` (recordings named `*_eve_*.mff` /
  `*_mor_*.mff`) and `startle output/` with the ratings CSVs (`*_1.csv` = evening,
  `*_2.csv` = morning).
- `TRIGGER_CACHE`: `airflow_cache.pkl`, the per-trial trigger times (D105, picture code, D110)
  and D102 rest spans. The code that built it is no longer in the repo.

## Run

```
.venv/bin/python breathmetrics_py/final_pipeline.py            # every subject
.venv/bin/python breathmetrics_py/final_pipeline.py AB22 MS18  # selected subjects
```

The first run reads every `.mff` and caches the breaths in `breathmetrics_py/_cache/`; later runs
reuse that cache unless the signal settings at the top of `final_pipeline.py` change. Outputs go
to `breathmetrics_py/_out_final/`:

| File | Rows |
|---|---|
| `trials_final.csv` | one per kept trial × anchor (`code` = picture, `sound` = D110), with `_resp`, `_fix`, `_delta`, `_pct`, `_log` per metric |
| `subject_medians.csv` | one per anchor × subject × session × valence (Negative / Neutral / All) |
| `trial_verdicts.csv` | one per trial × anchor, with rejection reason |
| `breaths.csv` | one per breath × anchor, with its window label |
| `rest_reference.csv` | one per session × anchor: rest medians (pooled, pre, post) |
| `session_qc.csv`, `skipped.csv` | per-session counts and spans; sessions skipped and why |

Metrics: `ti_s` (inhale time), `ttot_s` (breath length), `vi_over_ti` (mean inspiratory flow
proxy), `mv_au_min` (minute-ventilation proxy).
