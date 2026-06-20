# RESP — Respiration Analysis Pipeline

Mirrors the `emg_raw_potentiation.py` architecture for respiration channels
(`Airflow`, `Chest_belt`, `Abdomen_belt`, `Pleth`).

## Files

| File | Role |
|------|------|
| `resp_config.py` | **All tunable parameters.** Edit this for filters, windows, rejection, per-subject overrides. |
| `resp_processor.py` | MFF loading + raw epoch extraction (cached); filter/score application (live). |
| `resp_main.py` | Orchestration, plotting, execution toggles. |
| `view_respiration.py` | Standalone interactive viewer (separate from the pipeline). |

---

## How to run

From the **repo root**:

```bash
.venv/bin/python -m RESP.resp_main
```

Output lands in `~/Desktop/startle_output/RESP/`.

---

## Two-layer cache — what needs a reload and what doesn't

The pipeline separates slow MFF work (Layer 1, cached) from fast analysis (Layer 2, always live).

### Layer 1 — MFF cache (`resp_signals_cache.pkl`)

Built once from raw MFF files. Stores unfiltered wide epochs for every trial.

**Triggers a rebuild** (set `FORCE_PREPROCESSING = True` in `resp_main.py`):
- `RESP_CHANNELS` changed (new channel added)
- `WIDE_TMIN` or `WIDE_TMAX` changed

**After rebuilding**, set `FORCE_PREPROCESSING = False` again.

### Layer 2 — live on every run (no MFF access, takes ~seconds)

Everything in `DEFAULT_PARAMS` and `SUBJECT_PARAMS` in `resp_config.py`:

| Parameter | What it controls |
|-----------|-----------------|
| `hp_cutoff`, `lp_cutoff`, `filter_order` | Butterworth bandpass applied to each epoch |
| `baseline_tmin`, `baseline_tmax` | Pre-stimulus window for baseline mean (default −4 s → 0 s, ≈ 1 breath cycle) |
| `anal_tmin`, `anal_tmax` | Trimmed epoch stored per trial for plotting |
| `score_tmin`, `score_tmax` | Window for scoring: `mean(signal) − baseline_mean` |
| `z_score_threshold` | Rejection: outlier baseline std relative to session |
| `absolute_max` | Hard amplitude rejection limit (None = disabled) |

---

## Normal workflow

### 1. First run (builds the cache)

```bash
.venv/bin/python -m RESP.resp_main
```

MFF files are loaded once, epochs extracted, cache saved. Subsequent runs skip this step.

### 2. Change filter / baseline / scoring parameters

Edit `resp_config.py` — no `FORCE_PREPROCESSING` needed:

```python
DEFAULT_PARAMS = {
    "lp_cutoff":     0.5,    # tighten the low-pass
    "baseline_tmin": -5.0,   # longer baseline window
    "score_tmax":    3.0,    # shorter score window
    ...
}
```

Re-run:

```bash
.venv/bin/python -m RESP.resp_main
```

### 3. Tune parameters per subject

Add overrides in `resp_config.py` under `SUBJECT_PARAMS`. Any key from `DEFAULT_PARAMS` can be overridden:

```python
SUBJECT_PARAMS = {
    "DA01": {"lp_cutoff": 0.5, "baseline_tmin": -5.0},
    "NB03": {"z_score_threshold": 2.0},
}
```

Re-run. Cache unchanged, overrides applied live.

### 4. Run only specific plots

Toggle the booleans at the top of `main()` in `resp_main.py`:

```python
FORCE_PREPROCESSING               = False
PLOT_INDIVIDUAL_TRIALS            = False   # slow — skip unless debugging
PLOT_MULTICHANNEL_GROUP_TIMECOURSE = True   # key RESP diagnostic plot
PLOT_GROUP_FOUR_CONDITIONS        = True
...
```

### 5. Run a subject subset

Set `SUBJECT_FILTER` in `resp_config.py`:

```python
SUBJECT_FILTER = ["DA01", "AB22"]   # [] = all subjects
```

---

## Timing reference

Based on the actual startle CSV data:

| Measure | Value |
|---------|-------|
| Picture duration | ~5–7 s |
| Startle probe (D110) within picture | ~2–4 s after onset |
| Time from D110 to next picture onset | min 7 s, mean 9 s |
| Inter-D110 interval (consecutive sound trials) | min 14 s, mean 20 s |
| Respiratory period (15 breaths/min) | ~4 s |

Current defaults: `WIDE_TMAX = 6.0 s` (stays within every trial's fixation window); `baseline_tmin = −4.0 s` (covers ≈ 1 full breath cycle for a stable, phase-independent baseline).

---

## Output structure

```
~/Desktop/startle_output/RESP/
  resp_signals_cache.pkl                  ← Layer 1 cache
  group_four_conditions_boxplot.png
  group_neg_neu_ratio_boxplot.png
  group_session_comparison_boxplot.png
  group_eve_mor_ratio_boxplot.png
  group_overall_eve_vs_mor.png
  group_overall_ratio.png
  group_average_timecourse.png
  group_overall_timecourse.png
  multichannel_group_timecourse.png       ← all 4 channels × sessions grid
  stai_correlations.png
  subjective_negative_percentage_boxplot.png
  trial_score_scatter/
    <SubjectID>_trial_scores.png
  subject_average_timecourses/
    <SubjectID>_average_timecourse.png
  subject_multichannel_timecourses/
    <SubjectID>_multichannel_timecourse.png
  individual_trials/                      ← only if PLOT_INDIVIDUAL_TRIALS = True
    <SubjectID>/<session>/trial_NNN_NEG.png
```

---

## Interactive viewer (separate tool)

```bash
.venv/bin/python RESP/view_respiration.py DA01 eve
```

Scroll with `← →`, zoom with `+ −`. Does not use the pipeline cache.
