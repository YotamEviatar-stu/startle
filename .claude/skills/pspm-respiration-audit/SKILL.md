---
name: pspm-respiration-audit
description: Verifies the Airflow/respiration pipeline (Startle/Airflow/, esp. airflow_glm.py, airflow_config.py) against PsPM (bachlab/PsPM), the reference MATLAB toolbox for respiration/SCR/pupil/cardiac analysis. Use whenever Airflow/respiration code is written, edited, or reviewed — filtering cutoffs, cycle/breath detection, RP/RA/RFR extraction, GLM basis functions (tau/sigma), epoch/window definitions, artifact rejection. Also use for "does this match PsPM", "is this how PsPM does it", or references to pspm_resp_pp, pspm_glm, PsPM basis functions by name. Ground-truth reference only — does not run MATLAB/PsPM itself, holds its source-level defaults for line-by-line Python comparison.
---

# PsPM Respiration Audit

PsPM (Psychophysiological Modelling, bachlab/PsPM) is the reference MATLAB
toolbox this project's Airflow pipeline is modeled on. It is not vague
"industry practice" — it is a specific, versioned, published codebase with
exact filter cutoffs, exact cycle-detection logic, and exact basis-function
parameters, backed by peer-reviewed papers (Bach et al. 2016 for the linear
respiration model; Castegnetti et al. 2017 for the fear-conditioning
variant). The job of this skill is to hold that source-level ground truth
so that when reviewing `Startle/Airflow/airflow_glm.py`,
`airflow_config.py`, or `airflow_amp_processor.py`, deviations get named
precisely — "this differs from `pspm_resp_pp.m` line X" rather than "this
seems off."

Two reference files hold the actual transcribed source:

- `references/pspm-preprocessing.md` — `pspm_resp_pp.m`: mean-centering,
  the two-stage Butterworth filter, median filter, bellows-vs-cushion
  cycle detection, the 1-second minimum-IBI floor, and how RP/RA/RFR are
  computed and interpolated. Read this before auditing anything in
  `airflow_config.py`'s "Phase 1/2/3" preprocessing section or any custom
  cycle-detection code.
- `references/pspm-glm-respiration.md` — the `defaults.glm` table from
  `pspm_init.m` (per-modality filter settings for `ra_e`/`rp_e`/`rfr_e`)
  and the three basis function files (`pspm_bf_rprf_e.m`,
  `pspm_bf_rarf_e.m`, `pspm_bf_rfrrf_e.m` — the Gaussian tau/sigma values
  and derivative-regressor defaults). Read this before auditing
  `GLM_RF_PARAMS`, `GLM_FINAL_HP`/`GLM_FINAL_LP`, or any GLM design-matrix
  logic in `airflow_glm.py`.

Both files are direct transcriptions of the live bachlab/PsPM `develop`
branch (fetched and verified when this skill was built), not paraphrase —
quote them directly when reporting a finding.

## Scope: GLM path only

This project runs two scoring paths in parallel: a peak-picking scorer
(`airflow_amp_processor.py`, `SCORING_METHOD = "peak_excursion_normalized"`)
and a GLM/deconvolution scorer (`airflow_glm.py`,
`SCORING_METHOD = "glm_deconvolution"`). **PsPM only documents the GLM
approach** — `pspm_resp_pp.m` → `pspm_glm.m` is the published, peer-reviewed
method. The peak-excursion scorer is a legitimate project-specific design
choice, but PsPM has no opinion on it and this skill should not invent one.
When auditing, scope findings to the GLM path (`airflow_glm.py`,
`airflow_config.py`'s `GLM_*` constants) unless the user explicitly asks
about the peak-picking scorer too.

## How to audit

1. **Read the current Airflow GLM code first** — `airflow_config.py`'s
   `GLM_*` section and `airflow_glm.py` — before opening the reference
   files, so you know what you're comparing against, not what you expect
   to find.
2. **Match each processing stage to its PsPM counterpart**, in order:
   anti-alias/downsample → cycle-detection bandpass → median filter →
   zero-crossing detection → per-cycle RP/RA/RFR computation →
   interpolation → GLM basis-function convolution → per-modality
   sensitivity filter. A mismatch in *ordering* (e.g. z-scoring before vs
   after cycle detection) can matter as much as a mismatch in a cutoff
   value, because PsPM computes RA/RFR from the **native-rate raw signal**
   at each detected cycle window, not from the filtered/z-scored trace —
   check which signal a computation actually reads from, not just what
   filter was nominally applied upstream. **This is a higher-priority check
   for this project specifically**: the Airflow channel is a flow
   transducer, and a flow signal carries more high-frequency content than
   the bellows/chest-strap signal PsPM's 0.6 Hz detection filter was tuned
   against (flow ≈ d(volume)/dt, and differentiation pushes energy toward
   higher frequencies — peak inspiratory flow is a sharp, early-breath
   feature, not a slow rounded peak). That means measuring RA/RFR on the
   filtered/detection copy risks clipping *more* of the true peak height
   for this project's sensor than it would for a classic bellows trace. See
   "Why the raw-vs-filtered separation matters more for airflow" in
   `references/pspm-preprocessing.md` before deciding this finding is
   low-priority.
3. **Distinguish real deviations from defensible project-specific
   choices.** Not every difference from PsPM is a bug:
   - PsPM's `pspm_resp_pp` only enforces a 1-second *minimum* IBI; an
     upper bound / minimum-rate floor (like this project's
     `GLM_MIN_RATE_THRESHOLD`) is an addition, not a contradiction — flag
     it as "beyond PsPM" for the user's awareness, not as an error.
     Per `CLAUDE.md`, artifact rejection should not silently average over
     bad data, so additional gates are generally welcome — the goal is
     accurate labeling, not forcing parity with PsPM's exact gate set.
   - `ra_fc` (fear-conditioning) and `ra_e` (evoked) are different,
     equally valid PsPM configurations. Compare Airflow against whichever
     one matches its actual experimental design (this project uses
     evoked/event-related responses to startle onsets, so `_e` variants
     are the right comparison), not whichever is closest numerically.
   - Bellows vs. cushion cycle detection is a hardware choice, not a
     quality difference. Airflow sensors are flow-based, so the bellows
     (direct zero-crossing) rule is the correct PsPM analogue — don't flag
     a bellows-style detector as "missing" the cushion derivative logic.
4. **Report findings the way this project reports technical changes**
   (see `CLAUDE.md`): name the file and line, quote the PsPM value it's
   being compared against, and show a concrete numerical example of the
   consequence — not just "this uses a different filter." For example:

   > `airflow_config.py:118` sets `GLM_CYCLE_BANDPASS` with a code comment
   > claiming "order=2" for the cycle-detection bandpass. `pspm_resp_pp.m`
   > (see `references/pspm-preprocessing.md`, Stage 1) does not design a
   > single 2nd-order bandpass — it runs two *separate* 1st-order
   > Butterworth filters (`lporder=1` at 0.6 Hz, `hporder=1` at 0.01 Hz),
   > each applied bidirectionally via `filtfilt`. A single
   > `scipy.signal.butter(N=2, [0.01, 0.6], btype='band')` has a different
   > roll-off and phase response than two cascaded 1st-order filtfilt
   > passes, even though both are nominally "order 2." At the passband
   > edges (e.g. 0.55-0.6 Hz, near typical fast-breathing rates ~35 bpm)
   > the two designs will disagree on attenuation, which can shift where a
   > borderline-fast breath's zero-crossing lands by a sample or two —
   > exactly the kind of cycle-timing jitter that changes RP/RA readings.
   > **Whether this actually matters here depends on measuring the two
   > filters' impulse/frequency response directly — don't assert the
   > magnitude without checking, just flag the design difference.**

   Then, per `CLAUDE.md`, always say whether the fix is free to tune or
   requires `FORCE_RELOAD` (bandpass/cycle-detection constants in the GLM
   path are Layer 2 — no reload needed, since they run off the cached
   `signal_raw` at `CACHE_SFREQ`).
5. **Cross-check basis function parameters exactly.** `GLM_RF_PARAMS` in
   `airflow_config.py` should match the `(mu, sigma)` table in
   `references/pspm-glm-respiration.md` to the decimal — these are
   published constants (8.07/3.74 for RA, 4.2/1.65 for RP, 6.0/3.23 for
   RFR), not tunable hyperparameters. If they've drifted, that's worth
   flagging regardless of whether it changes the conclusion, since it
   breaks the traceability back to the cited papers.
6. **Check the per-modality filter split**, the single most likely place
   for silent drift: PsPM gives RP a different high-pass (0.01 Hz) than RA
   and RFR (0.001 Hz) in the `_e` (evoked) defaults. A pipeline using one
   shared `GLM_FINAL_HP` constant across all three channels has collapsed
   this distinction — confirm whether that's intentional (documented,
   e.g. "we use one shared filter because X") or accidental convergence.

## What this skill does not do

- It does not run MATLAB, install PsPM, or execute any PsPM function —
  it holds transcribed source as ground truth for comparison.
- It does not adjudicate whether the peak-excursion (`airflow_amp_processor.py`)
  scoring method is statistically sound — that's outside what PsPM
  documents. Use `/cross-pipeline-audit` or `/red-team-review` for that.
- It does not re-derive statistical conclusions (p-values, effect sizes)
  — it only checks whether the *signal-processing methodology* matches
  the cited reference implementation.
