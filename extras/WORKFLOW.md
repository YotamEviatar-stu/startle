# Pipeline Workflow Reference

## Goal
Demonstrate that **Evening startle reactivity > Morning** across two independent physiological channels (HR and Airflow), using Wilcoxon signed-rank test on per-subject condition means.

---

## Active Pipelines

| Pipeline | Signal | Output dir | Cache |
|---|---|---|---|
| HR | SpO2-Pulse (BPM) | `~/Desktop/spo2_output/` | `spo2_output/_cache/hr_cache.pkl` |
| Airflow | Airflow belt | `~/Desktop/airflow_output/` | `airflow_output/_cache/airflow_cache.pkl` |

Canonical reference for all design decisions: `scripts/emg_raw_potentiation.py`

---

## Key Files

```
HR/
  hr_config.py          ← all tunable parameters (edit here only)
  hr_processor.py       ← processing logic (two-layer cache)
  hr_main.py            ← run to rebuild plots
  hr_explore.ipynb      ← interactive analysis notebook
  hr_raw_show.ipynb     ← raw signal viewer per subject (NEW)

Airflow/
  airflow_config.py
  airflow_amp_processor.py
  airflow_main.py
  airflow_explore.ipynb
  airflow_raw_show.ipynb  ← raw signal viewer per subject (NEW)

.claude/skills/tune-pipeline/tune.py   ← coordinate-descent parameter tuner
```

---

## Run Order

### 1. Full pipeline run (HR)
```bash
.venv/bin/python -m HR.hr_main
```
Rebuilds cache (if FORCE_RELOAD=True) and regenerates all plots.

### 2. Full pipeline run (Airflow)
```bash
.venv/bin/python -m Airflow.airflow_main
```

### 3. Raw signal inspection (per subject)
Open `HR/hr_raw_show.ipynb` or `Airflow/airflow_raw_show.ipynb` in VS Code.  
Set `SUBJECT = 'AB22'` (and optionally `SESSION = 'eve'`) in cell 1, then run all.

- **View 1** — individual trial grid with pass/reject annotations
- **View 2** — full session signal with startle onset markers

For HR: `APPLY_FILTER = True/False` toggles a scipy Butterworth overlay (does NOT affect scoring).

### 4. Parameter tuning
```bash
.venv/bin/python .claude/skills/tune-pipeline/tune.py hr
.venv/bin/python .claude/skills/tune-pipeline/tune.py airflow
```
Coordinate-descent over config params. Writes best values back to config file automatically.

### 5. Interactive analysis
Open `HR/hr_explore.ipynb` — override parameters in cell 1 without touching config.

---

## Current HR Parameters (`hr_config.py`)

| Parameter | Value | Notes |
|---|---|---|
| `WIDE_TMIN / TMAX` | -10 / +20 s | Epoch cut window |
| `BASELINE_TMIN / TMAX` | -6 / -1 s | Pre-stimulus baseline |
| `BASELINE_METHOD` | `mean` | |
| `SCORE_METHOD` | `max_minus_baseline` | Peak HR − baseline (canonical, matches EMG) |
| `SCORE_TMIN / TMAX` | 1.5 / 8.0 s | Post-startle response window |
| `HR_Z_SCORE_THRESHOLD` | 3.0 | Noisy baseline gate |
| `SCORE_MAX_PCT` | 30% | Outlier score gate |
| `FLAT_MIN_SEC` | 12.0 s | Probe dropout threshold |
| `SPIKE_THRESH_BPM` | 20.0 | Impulse artifact gate |
| `SCORE_FLAT_DOMINANT_MAX` | 0.85 | NEW: reject if one BPM block covers >85% of score window |
| `SIGNAL_LOWPASS_HZ` | None | No filtering (SpO2 is already BPM) |

---

## Rejection Gates (HR) — in order applied

1. **bpm_range** — baseline mean outside 30–200 BPM (probe off)
2. **noisy_baseline** — baseline std > 3σ above session median (unstable probe)
3. **flat_baseline** — >50% of baseline window is interpolated artifact
4. **flat_score_window** — >50% of score window is interpolated artifact
5. **flat_score_dominant** *(new)* — single BPM block covers >85% of score window (no update = no measurable response)
6. **score_too_large** — |score| > 30% change (physiologically implausible)

No rejection by direction (`SCORE_MIN_PCT = None`).

---

## Current HR Results

| Metric | Value |
|---|---|
| N subjects (paired) | 20 |
| Rejection rate | ~24% |
| Direction | **Eve > Mor** ✓ |
| Wilcoxon p | ~0.114 |
| Eve mean score | ~0.74% |
| Mor mean score | ~0.54% |

---

## Two-Layer Cache Design

```
Layer 1  (requires FORCE_RELOAD = True to rebuild)
  → loads MFF, crops to D101–D124, extracts SpO2-Pulse channel
  → stores: raw_signal, sfreq, startle_samps, trials_meta
  → only changes when: HR_CHANNEL, HR_CACHE_SFREQ, trigger codes change

Layer 2  (runs every time from cache — free to tune)
  → cuts epochs at WIDE_TMIN/TMAX, cleans artifacts, computes baseline,
     applies rejection gates, scores trials
  → everything in hr_config.py except channel/sfreq/triggers
```

---

## Known Data Issues (do not fix in code)

| Subject | Issue |
|---|---|
| ES29 eve | Corrupt MFF — skipped |
| MG14 mor | Corrupt MFF — skipped |
| ML28 eve | macOS hidden file picked up instead of MFF — skipped |
| AH19 eve | SpO2 probe off entire session — all 30 trials rejected |
| LO21 | Loads MH20's MFF (files are in LO21's folder on drive) |

---

## Skills (invoke in Claude Code chat)

| Skill | When to use |
|---|---|
| `/startle-experiment` | Start of new session — loads full experiment context |
| `/cross-pipeline-audit` | After modifying HR or Airflow code |
| `/tune-pipeline` | After changing rejection logic — re-optimise params |
