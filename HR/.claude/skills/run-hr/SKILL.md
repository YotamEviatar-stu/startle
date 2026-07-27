---
name: run-hr
description: Run, debug, and interactively explore the HR (SpO2-Pulse) startle pipeline. Use this to run the pipeline, check cache health, change analysis configurations, produce group plots, and launch the Jupyter notebook for interactive analysis.
---

# HR Pipeline Skill

Python pipeline that extracts per-trial SpO2-Pulse (BPM) epochs from MFF recordings,
applies moving-baseline correction, scores peak % change, rejects outliers, and
produces group comparison plots. Driven via `smoke.py` (cache check + optional full run)
or the Jupyter notebook for interactive config iteration.

**All commands run from `/Users/yotameviatar/startle-1`.**

## Prerequisites

No extra installs — the `.venv` already has everything:
```
mne 1.8, numpy 2.0, pandas 2.3, scipy 1.13, matplotlib 3.9, jupyter
```

## Cache

The pickle cache lives at `~/Desktop/spo2_output/_cache/spo2_cache.pkl`.
It stores:
- `trials[subj][sess_key]` — raw epoch_wide per trial + CSV metadata
- `traces[subj][sess_key]` — continuous session signal for raw-session plots

**Cache boundary — what requires `FORCE_RELOAD = True` in `hr_config.py`:**
| Needs reload | Does NOT need reload |
|---|---|
| `WIDE_TMIN / WIDE_TMAX` | `ANAL_TMIN / ANAL_TMAX` |
| `HR_CHANNEL` | `BASELINE_TMIN/TMAX/METHOD` |
| Adding subjects with no prior MFF data | `SCORE_TMIN/TMAX/METHOD` |
| | `HR_MIN/MAX_BPM`, `HR_Z_SCORE_THRESHOLD`, `SCORE_MAX_PCT` |
| | All plot flags, colors, smoothing |

After enabling `FORCE_RELOAD`, set it back to `False` immediately after the first run.

## Run: smoke test (fast, no plots)

Verifies cache integrity and `apply_analysis_params` in ~150 ms. No files written.

```bash
cd /Users/yotameviatar/startle-1
.venv/bin/python HR/.claude/skills/run-hr/smoke.py
```

Expected output:
```
=== HR cache health ===
  Loaded in ~30 ms
  Subjects: 24
  Trials:   1373
  ...
=== apply_analysis_params ===
  Accepted: 1302/1373  (5.2% rejected)
  SCORE_MAX_PCT=50.0%  ✓
=== Smoke PASSED ===
```

## Run: full pipeline (all plots, ~2 min)

```bash
cd /Users/yotameviatar/startle-1
.venv/bin/python -m HR.hr_main
```

All plots saved to `~/Desktop/spo2_output/`. Subjects in cache print "using cache"; new
subjects load from drive (slow, one-time). ES29 eve, MG14 mor, ML28 eve are corrupt MFFs
— they skip gracefully every run.

## Interactive config iteration (no files written)

```bash
cd /Users/yotameviatar/startle-1
.venv/bin/jupyter notebook HR/hr_explore.ipynb
# or in VS Code: open HR/hr_explore.ipynb, select ".venv" kernel
```

Workflow:
1. Edit **Cell 2** (config overrides — e.g. change `SCORE_TMAX`, `HR_Z_SCORE_THRESHOLD`)
2. Re-run **Cell 3** (loads cache + applies new params, ~150 ms, no drive needed)
3. Re-run any plot cell — renders inline, no files written

## Changing a configuration

Edit `HR/hr_config.py`, then re-run the full pipeline or the notebook.
Key parameters:

```python
SUBJECT_FILTER      = []           # [] = all subjects; ['AG05'] = one subject
ANAL_TMIN / ANAL_TMAX              # analysis window shown in epoch plots
BASELINE_TMIN / BASELINE_TMAX      # pre-startle window used as each trial's baseline
SCORE_TMIN / SCORE_TMAX            # window in which peak BPM change is scored
HR_MIN_BPM / HR_MAX_BPM            # reject if baseline outside range
HR_Z_SCORE_THRESHOLD               # reject if baseline variability z-score > threshold
SCORE_MAX_PCT = 50.0               # reject trial if |% change| > this
```

## Debug: inspect a specific subject / session

```python
import pickle, os
from HR import hr_config as config, hr_processor as processor

with open(os.path.expanduser("~/Desktop/spo2_output/_cache/spo2_cache.pkl"), "rb") as f:
    data = pickle.load(f)
trials = data["trials"]
processor.apply_analysis_params(trials, config)

for i, t in enumerate(trials["AG05"]["mor"]):
    print(f"  Trial {i+1:02d}: bl={t['baseline_mean']:.1f}  score={t['score']:.2f}%  rej={t['rejected']}")
```

## Gotchas

- **Smoothing sigma**: SpO2-Pulse steps every 2 s at the source. Display smoothing uses
  `gaussian_filter1d` with σ=4 s (epochs) and σ=6 s (session traces). σ < 2 s leaves
  visible staircase steps.

- **AH19 eve: 30/30 rejected** — baseline HR outside 30–200 bpm range. Probe was off.
  Excluded from all group plots automatically.

- **LO21 loads MH20 files** — MH20's MFF is inside LO21's folder on the drive.

- **ES29 eve / MG14 mor / ML28 eve** — corrupt MFFs; skipped every run.

- **`apply_analysis_params` signature**: takes the full `{subj: {sess: [trials]}}` dict,
  not a list of trials.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: spo2_cache.pkl` | Run full pipeline once with the drive attached |
| `AttributeError: 'list' has no attribute 'items'` | Passing a list to `apply_analysis_params`; pass the full dict |
| Staircase steps still visible | Increase `sigma_sec` in `_smooth()` in `hr_main.py` |
| Full run takes > 5 min | Check cache size: `ls -lh ~/Desktop/spo2_output/_cache/spo2_cache.pkl` |
