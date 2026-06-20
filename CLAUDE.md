# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the pipeline

Run from the repo root (required because `ecg_main.py` imports `ECG.ecg_config` and `ECG.ecg_processor` as a package):

```bash
.venv/bin/python -m ECG.ecg_main
# or equivalently:
.venv/bin/python ECG/ecg_main.py
```

The virtualenv (`.venv/`) contains: `mne 1.8`, `numpy 2.0`, `pandas 2.3`, `scipy 1.13`, `matplotlib 3.9`.

## Architecture

**ECG pipeline** (`ECG/`): three-file separation of concerns.

- `ecg_config.py` — all tunable parameters: data paths, channel names, trigger codes, filter settings, event windows, color/marker mappings, output flags. **This is the primary file to edit when adapting to new data.**
- `ecg_processor.py` — stateless processing functions: MFF file discovery, DIN event extraction, CSV loading, per-session signal processing (crop to task boundaries, Butterworth HP+LP filtering, event windowing). Returns a list of per-trial dicts containing the ECG snippet, full session trace, and the entire CSV metadata row.
- `ecg_main.py` — orchestration only: iterates subjects/sessions, calls `ecg_processor`, then generates plots and exports. Plotting functions live here (`plot_session_timecourse`, `plot_event_windows`, `export_events_csv`).

**Data layout expected on disk:**
```
RAW_DATA_DIR/
  <SubjectID>/
    EEG/         ← contains *_eve_*.mff and *_mor_*.mff files
    startle output/   ← or CSVs directly in subject folder
      *_1.csv    ← evening ratings
      *_2.csv    ← morning ratings
```

**Event alignment:** DIN trigger channels (prefix `D`) are detected by amplitude threshold. Trigger 101 = session start, 124 = session end, 110 = startle. Events are matched to CSV rows by sequential order (first startle event → first CSV row).

**EMG script** (`scripts/emg_raw_potentiation.py`): original, standalone startle EMG analysis. Same data directory conventions and EGI MFF format, but a single-file script with inline config. Independent of the ECG pipeline.

**Agent workflow** (`agents/`): three Claude Code sessions — Coder, Reviewer (this session), Debugger. See `agents/README.md`. Open via `Cmd+Shift+P → Tasks: Run Task → Open All Agents`.

## Key configuration points

- `ECG_CHANNEL` / `SUBJECT_CHANNEL_OVERRIDES` — channel name differs between recordings; override per subject as needed.
- `TRIGGER_*` constants — DIN codes; change if the task software used different trigger values.
- `COLOR_BY_COLUMN` — which CSV column drives event marker color; must match a column name in the ratings CSV.
- `FORCE_RELOAD_ECG` — set `True` to bypass any cached state; useful when re-running after data corrections.
