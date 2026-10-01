# CLAUDE.md

Project guidance for the **HR** repo: heart rate / HRV in the startle experiment, from an ECG-like signal recovered from the EEG. General communication style, global skills and memory conventions live in `~/.claude/CLAUDE.md` and apply here unchanged (that file doesn't exist on this Windows machine yet).

## What we are building

A cardiac processing pipeline, rebuilt from scratch. The previous implementation (`hr/ica_cardiac.py` and its plotting scripts) is ignored — do not read it, reuse it, or port logic from it.

Stages, built strictly one at a time. A stage starts only after the user has approved the previous one:

1. **SASICA / CARACAS implementation** — extract the ECG-like component from the EEG, following the provided documentation.
2. **Signals for all sessions** — run stage 1 over every session and produce the cardiac signal per session.
3. **Trial / window alignment** — align the signal to trials and rest windows.
4. **Scoring** — HR and HRV measures, per trial and per session.

Within a stage, work autonomously: implement, run, check. At the end of a stage, stop and review with the user what was done. The one exception inside a stage is a method choice not covered by the documentation — that follows "How to propose a method" below.

Final measures and statistical tests are not fixed yet; they are decided as we go. Don't anticipate them in earlier stages.

## Sources of truth

- **ECG-from-EEG know-how** (how to extract the cardiac signal): the user-provided documentation and tools. Follow them; don't substitute your own approach.
- **`papers/`** — the reference library: PDFs of the papers behind every method used, plus `papers/INDEX.md` (one line per paper: citation, what it's used for, implementation repo URL). Check it first for any methodology question; when a new method is adopted, add its paper and index line.
- **Everything else** (beat detection, cleaning, rejection, alignment, HR/HRV): the user decides, method by method.
- **"Validated" means sourced:** every method is a system or function from a published paper with its public implementation (e.g. the authors' git repo). Never invent logic, heuristics or thresholds, only if the user explicitly says so.

## How to propose a method

When a step needs a method, or the documentation is silent/ambiguous (e.g. a procedure with no threshold value), don't decide — bring a sourced recommendation and wait for the user's decision. Each proposal states:

1. **Current signal** — what the data looks like at this point.
2. **Suggestion** — the method, with paper + implementation link.
3. **Is it common** — how standard it is in the literature.
4. **Final output** — what this step produces.

## Blinding

Condition labels (session key `eve`/`mor`, sound code, picture/valence code) enter **only at the scoring stage**. No extraction, cleaning, rejection or alignment step may read them.

## Scope — cardiac only

**Out of scope — do not search, read, run or discuss:** breathing, airflow, nasal pressure, BreathMetrics, the Airflow GLM, EMG startle scoring. The breathing project lives in a separate repo (`~/airflow`) and is deliberately invisible here; never open it, grep it, or import from it. If a task seems to need something from it, stop and ask the user.

## Data and experiment

- **All data lives in `C:\startle_data`** — the only storage. `F:\startle_raw` (raw MFFs) is temporary and will be removed; nothing may depend on it. Recordings are uniform-rate; don't add defensive handling for non-uniform rates.
- 250 Hz data: `C:\startle_data\<SUBJ>\<SUBJ>_<eve|mor>_raw.fif` — all eve/mor sessions, built by `hr/downsample.py` (no ICA or other processing). Channels: 257 EEG, `SpO2-Pulse`, DIN stim channels; exact trigger times are in the annotations (from the 1000 Hz data). DA01 eve comes from its `task` file, AK12 eve from `eve2` (`config.SESSION_FILE_ALIASES`). Per-session source file and failures: `C:\startle_data\downsample_log.csv`. Work from these files, not the raw MFFs.
- Behavioral files: `C:\startle_data\<SUBJ>\startle output\` (CSV `_1` = eve, `_2` = mor, plus `.log`/`.psydat`), copied from F. `subjects.xlsx` (STAI-T) was not on F — ask the user if it's needed.
- Machine: Windows 11, repo at `D:\user\Desktop\startle-repo`. No full MATLAB (only MATLAB Runtime v95), no FieldTrip, empty `SASICA\CARACAS\heart_functions` submodule — SASICA/CARACAS can't run until these are installed. `hr/config.py` MATLAB/FieldTrip/heart_functions paths and `.mcp.json` still point at the old Mac. Details: `/cardiac-ica` → `references/environment.md`.
- `hr/config.py` — data paths, session file aliases, `SUBJECTS_EXCLUDE`. HR-only: no breathing/airflow settings and no manual span edits — rest pre/post come from the D102 markers and the `/hr-experiment` timeline (≈60.22 s each). Don't change exclusions from automated checks.
- `/hr-experiment` skill — experiment design, data layout, DIN triggers, rest-block timeline, known bad files. Invoke before touching data loading or trial/rest spans.
- Cache only derived/downsampled data, never full-rate traces.
- Run with the repo venv by absolute path: `D:\user\Desktop\startle-repo\.venv\Scripts\python.exe` (from the repo root).

## Plotting

- Show raw, cropped signal unless asked otherwise; decimate with a peak-preserving method (min/max envelope), never naive striding.
- "Timecourse" means per-trial / per-beat values over time, not an averaged epoch.
- Every plot serves a diagnostic or statistical purpose. Show PNGs inline via Read rather than handing back a path.

## Code comments (WIP phase)

No explanatory comments or docstrings in source unless asked; justify choices in conversation instead.
