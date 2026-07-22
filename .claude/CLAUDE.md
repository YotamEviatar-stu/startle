# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository (the Startle experiment project). General communication style, cross-project skills, and memory conventions live in the global `~/.claude/CLAUDE.md` — this file covers only what's specific to Startle's domain.

## Research Context

This is **academic research**. Results must be statistically defensible, not merely visually appealing. Every analytical choice — filtering cutoffs, baseline windows, epoch timing, artifact rejection thresholds — must be justifiable and consistent across signals and sessions.

**Central hypothesis:** Morning physiological reactivity is calmer than Evening. Sleep reduces autonomic and respiratory responses to startling stimuli. The analysis must demonstrate this with real effect sizes and statistical tests (Wilcoxon signed-rank preferred given small N), not just trends in plots.

**Active pipelines:** HR (`Startle/HR/`) and Airflow (`Startle/Airflow/`). Both must be analysed to converge on the same conclusion, strengthening the claim across independent physiological channels.

**Canonical reference:** EMG raw (`Startle/emg_raw_potentiation.py`) defines the event logic, trial structure, DIN trigger handling, and baseline approach for generic helpers only (file discovery, event extraction) — epoch/trial extraction itself now lives in `Startle/trial_epochs.py` (see `/startle-experiment`). All pipelines must be consistent with it.

## Pipeline Conventions

When adapting or modifying a pipeline, read the EMG reference first and mirror its structure exactly — do not introduce custom approaches unless explicitly asked. For deeper cross-pipeline consistency checks, invoke `/cross-pipeline-audit`.

Key technical dimensions that must be tuned rigorously and consistently:
- **Filtering** — bandpass / highpass / lowpass cutoffs appropriate to the signal (HR vs. Airflow have different frequency content); document the choice.
- **Baseline** — pre-stimulus baseline window length and reference method (mean subtraction, z-score) must match across conditions and sessions.
- **Epoch timing** — onset offset relative to DIN trigger, epoch length, and any pre/post padding must be principled and matched to the EMG reference.
- **Artifact handling** — flag or exclude trials with implausible values (e.g. HR outside 40–180 BPM); do not silently average over bad data.
- **Aggregation** — report per-trial values and condition means; do not collapse across conditions unless explicitly asked.

## Airflow Pipeline — Current State

Two parallel scoring paths share the same Layer 1 cache
(`airflow_amp_processor.process_session`: native-rate 0.01–70 Hz wideband bandpass →
resample to `CACHE_SFREQ=25 Hz` → ground-truth trial windows from `trial_epochs.py`).
`SCORING_METHOD` in `airflow_config.py` selects which Layer 2 path runs; each has its own
config block and rejection gates kept at parity so switching methods doesn't silently
change strictness.

- **`peak_excursion_normalized` (BxB, `airflow_amp_processor.py`)** — breath-by-breath
  peak-picking. Score = max absolute excursion from baseline in the response window,
  normalised by the session-median NeuroKit2 `RSP_Amplitude` (i.e. multiples of a
  "typical breath" for that subject/session). A project-specific design choice — PsPM does
  not document or cover this method.
- **`glm_deconvolution` (GLM, `airflow_glm.py`)** — implements Bach et al. (2016), "A
  linear model for event-related respiration responses," modeled directly on PsPM's
  `pspm_resp_pp.m` / `pspm_glm.m` (verified line-for-line against PsPM source; see the
  `pspm-respiration-audit` skill):
  1. Mean-center + two **cascaded 1st-order** Butterworth filters (0.6 Hz lowpass, then
     0.01 Hz highpass, each bidirectional `filtfilt`) + downsample to 10 Hz + robust
     (median/MAD) z-score → a detection-only trace (`raw_z`).
  2. Detect cycle onsets on `raw_z` (negative zero-crossings — the bellows-style rule,
     correct for a flow transducer like Airflow). Measure RA/RFR on the **native raw
     signal** (`signal_raw`), never on the filtered/z-scored copy — PsPM's `resp` vs
     `newresp` separation. This matters more for airflow than for PsPM's bellows/chest-
     strap signal: flow ≈ d(volume)/dt, so differentiation pushes real inspiratory-peak
     energy into frequencies the 0.6 Hz detection filter would otherwise clip.
  3. Interpolate cycles to a continuous 10 Hz series per metric, then apply a
     **per-modality** "sensitivity" high-pass (RP 0.01 Hz; RA/RFR 0.001 Hz — PsPM does
     NOT share one cutoff across the three metrics) + a shared 1 Hz low-pass, applied
     unidirectionally (causal, `lfilter` not `filtfilt`).
  4. Fit a canonical-response-function GLM — Gaussian bump, published `(tau, sigma)`
     matched to PsPM to the decimal (RP 4.20/1.65, RA 8.07/3.74, RFR 6.00/3.23),
     orthogonalized derivative regressor for RA/RFR only (matches PsPM's `bf_type`
     defaults) — either **per-trial** (`GLM_ESTIMATION="per_trial"`, default: one
     independent regression per trial) or **pooled session-wide**
     (`"pooled_session"`: PsPM's actual `pspm_glm.m` architecture — one design matrix per
     session, one beta per condition). Score = β₁, the CRF regressor's weight.

`GLM_PRIMARY_METRIC="RA"` is the single pre-registered confirmatory endpoint
(multiplicity discipline — many Wilcoxons get computed, only one is reported as
confirmatory); RP/RFR/composite stay secondary/exploratory. Estimation path and primary
metric are fixed a priori and never chosen by watching the Eve-vs-Mor p-value.

Rejection gates (shared names/thresholds/intent across both paths): `no_cycles_found`,
`noisy_baseline`, `flat_signal`, `flat_response`, `rate_artifact`, `atypical_shape`
(shared waveform shape-centroid gate) — plus GLM-only `cycle_gap` (absolute minimum-rate
floor: catches a real cycle at each window edge with a long dead stretch in between,
which the std-based gates miss) and an amplitude-spike gate (Hampel upper-bound outlier
on cycle RA; `GLM_ARTIFACT_METHOD` controls whether flagged cycles are dropped or their
whole trial is rejected). All gates are a-priori and symmetric across Evening/Morning —
none inspects the Eve-vs-Mor result.

`SUBJECTS_EXCLUDE` in `airflow_config.py` is an **unresolved placeholder**, not an
established rejection rule — chosen by eyeballing one outlier scan and applied
inconsistently across similarly-contaminated sessions (documented as such in the config
file itself). Treat any Airflow result as provisional until this is replaced with a
principled, consistently-applied criterion.

Full mechanical write-up lives in `Startle/Airflow/README.md` (Layer 1 caching, shared by
both paths); GLM-specific assumptions and the current confirmatory-test state live in
`Startle/Airflow/GLM_METHOD_FOUNDATIONS.md` and `DISCUSSION.md`.

## Skills

Project skills should be invoked proactively (see each skill's own description for what it does): `/startle-experiment` (start of any new session, before touching HR or Airflow code), `/cross-pipeline-audit` (whenever HR or Airflow code is written or reviewed), `/tune-pipeline` (autonomous parameter tuning), `/startle-research` (domain literature/methodology research — checks `Startle/papers/` and existing method docs before searching externally).

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
