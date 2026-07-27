# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository (the Startle experiment project). General communication style, cross-project skills, and memory conventions live in the global `~/.claude/CLAUDE.md` — this file covers only what's specific to Startle's domain.

## Research Context

This is **academic research**. Results must be statistically defensible, not merely visually appealing. Every analytical choice — filtering cutoffs, baseline windows, epoch timing, artifact rejection thresholds — must be justifiable and consistent across signals and sessions.

**Central hypothesis:** Morning physiological reactivity is calmer than Evening. Sleep reduces autonomic and respiratory responses to startling stimuli. The analysis must demonstrate this with real effect sizes and statistical tests (Wilcoxon signed-rank preferred given small N), not just trends in plots.

**Active pipelines:** HR (`HR/`) and Airflow (`Airflow/`). Both must be analysed to converge on the same conclusion, strengthening the claim across independent physiological channels.

**Canonical reference:** EMG raw (`extras/emg_raw_potentiation.py`) defines the event logic, trial structure, DIN trigger handling, and baseline approach for generic helpers only (file discovery, event extraction) — epoch/trial extraction itself now lives in `extras/trial_epochs.py` (see `/startle-experiment`). All pipelines must be consistent with it.

## Pipeline Conventions

When adapting or modifying a pipeline, read the EMG reference first and mirror its structure exactly — do not introduce custom approaches unless explicitly asked. For deeper cross-pipeline consistency checks, invoke `/cross-pipeline-audit`.

Key technical dimensions that must be tuned rigorously and consistently:
- **Filtering** — bandpass / highpass / lowpass cutoffs appropriate to the signal (HR vs. Airflow have different frequency content); document the choice.
- **Baseline** — pre-stimulus baseline window length and reference method (mean subtraction, z-score) must match across conditions and sessions.
- **Epoch timing** — onset offset relative to DIN trigger, epoch length, and any pre/post padding must be principled and matched to the EMG reference.
- **Artifact handling** — flag or exclude trials with implausible values (e.g. HR outside 30–200 BPM, `HR_MIN_BPM`/`HR_MAX_BPM` in `HR/hr_config.py`); do not silently average over bad data.
- **Aggregation** — report per-trial values and condition means; do not collapse across conditions unless explicitly asked.

## Airflow Pipeline — Current State

`glm_deconvolution` (GLM) is the **primary, active line of work** — `SCORING_METHOD` in
`airflow_config.py` defaults to it, and `GLM_PRIMARY_METRIC` is the confirmatory endpoint.
`peak_excursion_normalized` (BxB, "amp") is kept only as a secondary/legacy comparison
path, not a co-equal method. Both share the same Layer 1 cache
(`airflow_glm.process_session`: native-rate 70 Hz anti-alias lowpass only — no highpass;
`signal_raw` deliberately keeps DC content, mirroring PsPM's unfiltered `resp` — →
resample to `CACHE_SFREQ=25 Hz` → ground-truth trial windows from `extras/trial_epochs.py`).
Rejection gates mostly share config-name-level parity, but two verified asymmetries remain
between the paths (`SUBJECTS_EXCLUDE`'s per-session `"SUBJ/sess"` form and `QC_ROBUST_GATES`'
statistic are GLM-only — see `Airflow/README.md`'s Rejection Gates section) — new work,
tuning, and audits should default to the GLM path unless told
otherwise.

- **`peak_excursion_normalized` (BxB, `airflow_amp_processor.py`, secondary/legacy)** —
  breath-by-breath peak-picking, normalised by session-median NeuroKit2 `RSP_Amplitude`.
  Project-specific; PsPM does not cover this method.
- **`glm_deconvolution` (GLM, `airflow_glm.py`, primary)** — Bach et al. (2016) linear
  respiration model, modeled on PsPM's `pspm_resp_pp.m`/`pspm_glm.m` (verified
  line-for-line against PsPM source). Full cascade-filter → cycle-detection →
  per-modality sensitivity-filter → CRF-GLM pipeline is in
  `Airflow/GLM_METHOD_FOUNDATIONS.md` and the `pspm-respiration-audit` skill — read
  those before touching this path. Score = β₁, the CRF regressor's weight, either
  **per-trial** or **pooled session-wide** (`GLM_ESTIMATION`).

`GLM_PRIMARY_METRIC="RA"` is the single pre-registered confirmatory endpoint
(multiplicity discipline — many Wilcoxons get computed, only one is reported as
confirmatory); RP/RFR/composite stay secondary/exploratory. Estimation path and primary
metric are fixed a priori and never chosen by watching the Eve-vs-Mor p-value.

Rejection gates (shared across both paths, a-priori and symmetric across Evening/Morning):
`no_cycles_found`, `noisy_baseline`, `flat_signal`, `flat_response`, `rate_artifact`,
`atypical_shape` — plus GLM-only `cycle_gap` and an amplitude-spike gate
(`GLM_ARTIFACT_METHOD`). See `Airflow/README.md` for thresholds and intent.

`SUBJECTS_EXCLUDE` in `airflow_config.py` is an **unresolved placeholder**, not an
established rejection rule — chosen by eyeballing one outlier scan and applied
inconsistently across similarly-contaminated sessions (documented as such in the config
file itself). Treat any Airflow result as provisional until this is replaced with a
principled, consistently-applied criterion.

Full mechanical write-up lives in `Airflow/README.md`; GLM-specific assumptions live in
`Airflow/GLM_METHOD_FOUNDATIONS.md` and `DISCUSSION.md`. Their reported RA p-values are
**stale** versus the current `SUBJECTS_EXCLUDE`/`GLM_ARTIFACT_METHOD` — see memory
(`project_pipeline_status`) for current status; rerun before citing.

## Skills

Project skills should be invoked proactively (see each skill's own description for what it does): `/startle-experiment` (start of any new session, before touching HR or Airflow code), `/cross-pipeline-audit` (whenever HR or Airflow code is written or reviewed), `/tune-pipeline` (autonomous parameter tuning), `/startle-research` (domain literature/methodology research — checks `papers/` and existing method docs before searching externally).

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

General communication/explanation style (numeric-example format for code changes, layered concept teaching, orienting-in-codebase requirements, test-statistic/p-value reporting, not flagging theoretical issues on uniform data) lives in `~/.claude/CLAUDE.md` — applies here too. Startle-specific addition:

- When explaining a paper's method (e.g. PsPM's GLM), stay at the level of what's specific to Startle/PsPM: canonical basis functions and their parameters, orthogonalization, event-train construction from the experimental design, per-modality filter choices, session-wide vs. per-trial architecture. Don't derive generic OLS/regression mechanics from scratch.

## Data & Caching

Recordings are uniform; assume a consistent sample rate and don't add defensive handling for non-uniform rates unless told otherwise. Cache only downsampled traces, never full 1000 Hz traces, to avoid multi-minute load times.

## Data Layout

See `/startle-experiment` for the full directory layout, CSV structure, and DIN trigger scheme (triggers 101=start/105=fixation/110=startle/124=end; validate each trial's `D{trigger_num}` against the CSV directly, not sequential order).
