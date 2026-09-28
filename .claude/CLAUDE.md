# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository (the Startle experiment project). General communication style, cross-project skills, and memory conventions live in the global `~/.claude/CLAUDE.md` — this file covers only what's specific to Startle's domain.

## Research Context

This is **academic research**. The deliverable is a pipeline whose processing can be defended — not a result. Every analytical choice (filtering cutoffs, cycle detection, baseline, epoch timing, rejection rules) must be justifiable from the signal itself and from published practice, and applied identically to every session.

**What the pipeline is evaluated on:** does it detect, clean, and score the physiological signal the way that signal should be processed? A pipeline is validated when each stage demonstrably does what it claims on the real recordings — artifacts are caught, genuine physiology survives, and every rejection carries a named reason traceable to a measured property of the signal. No stage is judged by whether it produces a group difference.

**Condition labels are blind to the pipeline.** No cleaning, rejection, or scoring function may read the session key (`eve`/`mor`), the valence label, or `has_sound`. Evening vs Morning survives in exactly one role: an **after-the-fact balance report** — how much signal each condition lost, seconds-weighted — which exists to catch a condition-biased gate. It is never an objective, never a gate, and never a reason to move a threshold.

**Unit of judgement.** The **breath** is the unit: one verdict per breath, made once, from that breath's own measured properties, feeding downstream stages that never re-derive validity from window statistics. Invoke `/breathmetrics` before touching detection or rejection — BreathMetrics has no automatic rejection of its own (its only rejection is a human clicking in `bmGui.m`, and it reaches nothing but the session-summary averages), so every gate in this pipeline is ours to justify from the signal.

**Active line of work:** the BreathMetrics port — `breathmetrics_py/` (Python), with the MATLAB `breathmetrics/` tree as ground truth.

**Superseded — the old method.** The Airflow GLM path (`Airflow/`, `glm_deconvolution`, `RA_norm`/`RFR_norm`, `beta`/CRF, `CYCLE_*`/`TRIAL_*` constants) is historical background, not current state; its code has been removed. Do not read `Airflow/CLAUDE.md`, and do not carry GLM or `RA_norm` framing into a session, unless the user names Airflow. The live settings (`SUBJECTS_EXCLUDE`, manual span/trial lists, data paths) moved to `breathmetrics_py/pipeline_config.py` on 2026-09-28; `Airflow/` no longer exists.

**Canonical reference:** EMG raw (`extras/emg_raw_potentiation.py`) defines the event logic, trial structure, DIN trigger handling, and baseline approach for generic helpers only (file discovery, event extraction) — epoch/trial extraction itself now lives in `extras/trial_epochs.py` (see `/startle-experiment`). All pipelines must be consistent with it.

## Pipeline Conventions

When adapting or modifying a pipeline, read the EMG reference first and mirror its structure exactly — do not introduce custom approaches unless explicitly asked.

Key technical dimensions that must be justified rigorously and consistently:
- **Filtering** — bandpass / highpass / lowpass cutoffs appropriate to the signal (EMG vs. nasal-pressure airflow have different frequency content); document the choice.
- **Baseline** — pre-stimulus baseline window length and reference method (mean subtraction, z-score) must match across conditions and sessions.
- **Epoch timing** — onset offset relative to DIN trigger, epoch length, and any pre/post padding must be principled and matched to the EMG reference.
- **Rejection** — one decision point per pipeline, made on the signal's own measured properties, with the reason recorded per unit. Never re-derive validity downstream; never reject on the score being scored. Where a reference statistic is untrustworthy, the gate **abstains and says so** rather than silently passing. Per-pipeline thresholds live in each pipeline's own `CLAUDE.md`.
- **Thresholds** — set from physiological plausibility, robust-statistics convention, or a cohort percentile frozen before any condition comparison. A threshold chosen because it improved an outcome is not a threshold.
- **Aggregation** — report per-trial values and condition means; do not collapse across conditions unless explicitly asked.

## Per-Pipeline State

The active line's entry point is `breathmetrics_py/final_pipeline.py`, configured by `breathmetrics_py/pipeline_config.py` and described in `breathmetrics_py/PIPELINE_WALKTHROUGH.md`; the MATLAB spec, porting traps and sanctioned deviations live in `.claude/skills/breathmetrics/references/`. The MATLAB `breathmetrics/` tree was moved out of the repo on 2026-09-28 (to `~/startle-1_removed_20260928/breathmetrics/`).

## Skills

Invoke proactively on the active line: `/breathmetrics` (whenever BreathMetrics or `breathmetrics_py/` is involved at all — porting, debugging, parameter provenance, MATLAB-vs-Python parity), `/nasal-pressure-flow-volume` (pressure→flow/volume questions and what this sensor can support), `/startle-research` (domain literature — checks `papers/` first), `/startle-experiment` (data layout, CSV structure, DIN trigger scheme).

There is deliberately **no parameter-search skill**. Coordinate descent over analysis parameters against an outcome was removed on 2026-08-12; a threshold is chosen from the signal and frozen, not searched.

## Plotting

- Show RAW, segmented/cropped signal without high-pass filtering unless explicitly asked.
- Never use naive stride-based downsampling — use decimation that preserves peaks.
- Avoid over-zoomed Y-axes and limit sample counts so MNE does not freeze the machine.
- When the user asks for a "timecourse", plot the per-trial value over time — NOT an epoch-averaged or mean value across trials.
- Every plot must serve a statistical or diagnostic purpose; decorative plots are not the goal.

## Code Comments & Docstrings (WIP phase)

The codebase is currently in active, iterative development — not yet finalized. Do NOT add explanatory comments, docstrings, or per-entry justification strings to code unless explicitly asked to; a bare list/dict is enough. The user will add documentation themselves once a piece of work is settled — if reasoning should be kept somewhere, say so in the response, not in the file.

This overrides the general instinct to document non-obvious choices while code is still churning. It does NOT override the requirement to justify choices *in conversation* — only what gets written INTO source files.

## Communication Style

General communication/explanation style lives in `~/.claude/CLAUDE.md` and applies here too — including the **answer-length ceiling (3 paragraphs / ~20 sentences max, max 3 findings per audit report — aim well below it)**, which binds skill output in this repo as well as ordinary replies; plus the numeric-example format for code changes, layered concept teaching, orienting-in-codebase requirements, test-statistic/p-value reporting, and not flagging theoretical issues on uniform data. Startle-specific addition:

- When explaining a paper's method (e.g. BreathMetrics), stay at the level of what's specific to that method: sliding-window extrema, onset/pause detection thresholds, per-breath volume integration, smoothing windows and their parameters at this sample rate. Don't derive generic signal-processing or statistics mechanics from scratch.

## Data & Caching

Recordings are uniform; assume a consistent sample rate and don't add defensive handling for non-uniform rates unless told otherwise. Cache only downsampled traces, never full 1000 Hz traces, to avoid multi-minute load times.

## Data Layout

See `/startle-experiment` for the full directory layout, CSV structure, and DIN trigger scheme (triggers 101=start/105=fixation/110=startle/124=end; validate each trial's `D{trigger_num}` against the CSV directly, not sequential order).
