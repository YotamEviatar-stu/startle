# HR Pipeline

Event-aligned heart rate (HR) analysis for the startle emotional-image paradigm.
Same subjects, same EGI MFF recordings, same DIN triggers and CSV rating files
as the EMG and ECG pipelines — but reading the `SpO2-Pulse` channel (hardware HR output) from the pulse oximeter.

## What it does

For each subject × session (evening / morning):

1. Loads the MFF file and crops to task boundaries (D101 → D124).
2. Reads `SpO2-Pulse` — the oximeter's processed pulse rate output in BPM.
   **No filtering is applied.** The signal is already processed hardware-side;
   filtering would destroy the BPM values.
3. Cuts a wide epoch around each D110 startle trigger.
4. Computes a **moving per-trial baseline** from the window immediately before
   each trigger. Because each trial uses its own local baseline, slow HR drift
   across the session is automatically corrected without a separate detrending
   step (trial 1's baseline may differ from trial 30's — and that's fine).
5. Computes a **score** (peak HR in the post-startle window minus baseline) and
   flags rejected trials.
6. Produces per-trial review plots, session-timecourse overlays, condition-mean
   timecourses, a group boxplot, and a trial-level CSV.

## Why it differs from the EMG pipeline

| Concept | EMG | HR (SpO2-Pulse channel) |
|---|---|---|
| Signal | Raw muscle potential (µV) | HR in BPM via SpO2-Pulse channel (hardware-processed) |
| Filtering | HP 28 Hz → rectify → LP 30 Hz | None |
| Epoch window | ±250 ms | −10 to +15 s (autonomic response unfolds over seconds) |
| Baseline window | −50 to 0 ms | −5 to 0 s, per-trial (moving) |
| Baseline method | mean of window | configurable: mean or median |
| Score | Peak amplitude in 20–100 ms | Peak HR in 0–6 s minus baseline |
| Rejection | Absolute µV threshold + z-score | Baseline outside physiological HR range |

## Package layout

```
SpO2/
  spo2_config.py    ← all tunable parameters — edit here first
  spo2_processor.py ← stateless: load, epoch, baseline, score, rejection
  spo2_main.py      ← orchestration, cache, plots, CSV export
```

## Run

```bash
# from repo root
.venv/bin/python -m SpO2.spo2_main
# or:
.venv/bin/python SpO2/spo2_main.py
```

## All config parameters (spo2_config.py)

```python
# Paths
RAW_DATA_DIR       # path to raw data (MFF + CSV files)
OUTPUT_DIR         # where outputs are written

# Run scope
SUBJECT_FILTER     # list of subject IDs; [] = run all

# Signal
SPO2_CHANNEL       # default "SpO2-Pulse"

# Epoch windows (seconds, relative to D110 trigger)
WIDE_TMIN / WIDE_TMAX          # full cut window stored in cache (default -10, +15)
ANAL_TMIN / ANAL_TMAX          # analysis / plot window, change freely (default -5, +10)

# Baseline — moving, per-trial
BASELINE_TMIN / BASELINE_TMAX  # pre-trigger baseline window (default -5, 0)
BASELINE_METHOD                # "mean" (default) or "median"

# Scoring
SCORE_METHOD                   # "max_minus_baseline" (default)
SCORE_TMIN / SCORE_TMAX        # post-startle peak-HR window (default 0, +6 s)

# Rejection
HR_MIN_BPM                     # floor — below this = artifact (default 30)
HR_MAX_BPM                     # ceiling — above this = artifact (default 200)

# Classification
USE_SUBJECTIVE_TRIAL_TYPE      # True = ratings; False = image_type column

# Cache
FORCE_RELOAD                   # True = ignore pickle, reload all MFF files

# Output
VERBOSE / PLOT_DPI
```

## Cache design

`process_session()` (MFF loading + epoch cutting) writes to a pickle.
`apply_analysis_params()` (baseline, score, rejection) runs every time from
the cached epochs, so any config parameter can be changed without reloading MFF.
