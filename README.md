# ECG Startle Pipeline

Heart-rate response analysis for an emotional-image startle paradigm.
Processes EGI MFF recordings, extracts R-peaks, computes per-trial HR change scores,
and generates group-level plots and STAI-T correlations.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
# Must run from repo root — ECG is imported as a package
.venv/bin/python -m ECG.ecg_main
```

## Repository layout

```
ECG/                        # Main pipeline package
  ecg_config.py             # All tunable parameters — edit here first
  ecg_processor.py          # Stateless signal processing functions
  ecg_main.py               # Orchestration, plotting, exports

scripts/
  emg_raw_potentiation.py   # Standalone EMG startle analysis (independent)

agents/                     # Claude Code agent configs (see agents/README.md)
```

## Data layout

```
RAW_DATA_DIR/
  <SubjectID>/
    EEG/                    # *_eve_*.mff  and  *_mor_*.mff
    startle output/         # *_1.csv (evening)  and  *_2.csv (morning)
```

## Key config points

| Parameter | File | Effect |
|-----------|------|--------|
| `ECG_CHANNEL` / `SUBJECT_CHANNEL_OVERRIDES` | ecg_config.py | Channel name per subject |
| `TRIGGER_*` | ecg_config.py | DIN codes (101=start, 124=end, 110=startle) |
| `FORCE_RELOAD_ECG` | ecg_config.py | Set `True` after changing filter or R-peak params |
| `USE_SUBJECTIVE_TRIAL_TYPE` | ecg_config.py | Label trials by ratings vs image_type column |

## Dependencies

See `requirements.txt`. Pinned versions: mne 1.8, numpy 2.0, pandas 2.3, scipy 1.13, matplotlib 3.9.
