# Airflow Pipeline

Two scoring methods share one cache. **`glm_deconvolution`** (GLM, `airflow_glm.py`)
implements the linear model from Bach et al. 2016, *"A linear model for
event-related respiration responses"* (`Airflow/GLM_METHOD_FOUNDATIONS.md` has the
method's assumptions; this file is the mechanical step-by-step). It is the
**primary, active** path (`SCORING_METHOD` in `airflow_config.py`). **`peak_excursion_normalized`**
(BxB, `airflow_amp_processor.py`) is a breath-by-breath peak-picking scorer kept as a
**secondary/legacy** comparison — a project-specific design PsPM does not cover.

Both methods run on top of the same **Layer 1** cache (session read + filter +
resample + ground-truth trial windows). Only Layer 2 (the method-specific
scoring) differs, and it re-runs every invocation regardless of cache state.

---

## Run

```bash
.venv/bin/python Airflow/airflow_main.py
```

- Method: set `SCORING_METHOD = "glm_deconvolution"` (default) or
  `"peak_excursion_normalized"` in `airflow_config.py`.
- Output: `<OUTPUT_DIR>/airflow_trial_scores.csv` (`OUTPUT_DIR` in
  `airflow_config.py`) — columns `score_rp`, `score_ra`, `score_rfr` for GLM, or
  `baseline_mean`, `score`, `score_ventilation` for BxB, plus `rejected` /
  `rejection_reason` either way. Per-trial review PNGs go to
  `<OUTPUT_DIR>/review_trials_glm/` or `<OUTPUT_DIR>/review_trials/`.
- Cache: `<OUTPUT_DIR>/_cache/airflow_cache.pkl`. Only rebuilt when
  `FORCE_RELOAD=True`; otherwise every run re-applies Layer 2 to the existing
  cache. If `RAW_DATA_DIR` (the external drive) isn't mounted and a cache
  already exists, `airflow_main.py` skips MFF loading and scores off the cache
  as-is.

---

## Layer 1 — Cache Build (shared, `airflow_glm.py: process_session`)

Per session, only on a cache miss:

1. Read the raw EGI `.mff` at full native rate, full channel set
   (`mne.io.read_raw_egi`).
2. `trial_epochs.build_trial_epochs` (`extras/trial_epochs.py`) on that
   untouched raw — derives ground-truth per-trial windows from the
   D105 (fixation) → D{code} (picture identity) → D110 (startle probe,
   sound trials only) → next D105 trigger cycle. Runs **before** channel-pick
   or resample: DIN triggers must not be resampled or have other channels
   dropped before detection. Raises `TriggerAlignmentError` on any D105/CSV
   count mismatch or code-value mismatch; callers let this propagate rather
   than silently continuing with untrustworthy alignment.
3. Pick the Airflow channel (`AIRFLOW_CHANNEL`).
4. **Anti-alias lowpass only** — 2nd-order Butterworth, `filtfilt`, cutoff
   `AIRFLOW_LOWPASS`=70 Hz, at native rate. `AIRFLOW_HIGHPASS`=0.01 Hz (also in
   `airflow_config.py`) is **not applied anywhere in the current code** — the
   only filter function actually called in `process_session` is
   `_butter_lowpass_filter`; `_butter_bandpass_filter`/`_butter_bandpass` are
   defined but never called. `signal_raw` therefore keeps its DC / very-low-frequency
   content deliberately — it stands in for PsPM's unfiltered `resp`
   variable (`pspm_resp_pp.m` measures RA/RFR on the untouched trace), so it
   must not carry a highpass PsPM itself keeps out of that variable.
5. Resample to `CACHE_SFREQ`=25 Hz (`raw.resample`, `npad="auto"`). The 70 Hz
   anti-alias sits well above the resampled Nyquist (12.5 Hz), so this cutoff
   can't be realized after resampling — it must happen at native rate, before
   step 5.
6. Cache payload per session: `signal_raw` (25 Hz), `times_raw`, and
   `trials_meta` — one dict per CSV row with `code_time`, `d105_time`,
   `d110_time`, `baseline_range_sec`=`(d105_time, code_time)`,
   `response_range_sec`=`(code_time, next trial's code_time or D124)`, plus
   labels/ratings. Trigger sample indices are converted to seconds at
   **native** rate, so trial timing is independent of `CACHE_SFREQ`.

---

## Layer 2 — GLM (`airflow_glm.py`, primary)

Re-applied every run to the Layer 1 cache; the code's own stage comments are
followed here.

### Stage 0 — Session QC pre-filter (`airflow_qc.py`, gates on `QC_ENABLED`, default `True`)

Runs on `signal_raw` before anything else:

- `session_breath_size`: robust-SD (median/MAD) of `signal_raw` bandpassed to
  `QC_BREATH_BAND`=(0.01, 0.6) Hz — the session's own typical-breath amplitude
  scale, call it `A`.
- `flag_excursions`: rolling-median (`QC_EXCURSION_WIN_SEC`=1.0s) deviation
  test, flags samples where `|x - rolling_median| > QC_EXCURSION_K × A`
  (`QC_EXCURSION_K`=5.0); flagged runs closer than `QC_MERGE_GAP_SEC`=0.2s are
  merged. Runs no longer
  than `QC_MAX_RUN_SEC`=1.0s are linearly interpolated over in a cleaned copy
  and their span recorded as `masked_spans`; longer runs (sustained
  contamination) are left alone — that's `SUBJECTS_EXCLUDE`'s job, not this
  gate's.
- `flag_deep_breaths` (`QC_MAX_BREATH_RATIO`=5.0, needs ≥3 cycles): after
  Stage 2 cycle detection, any cycle whose RA exceeds 5× the session's median
  RA is dropped and its `[onset, assign)` span added to the missing-span list
  — applied unconditionally, regardless of `GLM_ARTIFACT_METHOD`.
- **Important interaction:** when `QC_ENABLED=True` (the default), the Stage 1
  despike step below is skipped — `run_glm_scoring` passes
  `despike=(not qc_enabled) and GLM_DESPIKE_ENABLED`. QC's excursion masking
  supersedes despiking under current defaults; `GLM_DESPIKE_*` only takes
  effect if QC is turned off.

`masked_spans` (QC excursions) and dropped-cycle spans (deep breaths /
Hampel, see Stage 3) are collected into one `missing_spans` list, NaN-blanked
in the Stage 4 continuous series **after** filtering (see Stage 4) — comment
in code cites PsPM's `model.missing`.

### Stage 1 — Signal conditioning (`phase1_filter_downsample`)

Produces `raw_z`, used **only** for cycle detection (never for RA/RFR
amplitude, see Stage 2):

- Despike (conditional on Stage 0, see above): interpolate over native
  samples whose robust z-score exceeds `GLM_DESPIKE_K`=50 for a run no longer
  than `GLM_DESPIKE_MAX_RUN_SEC`=1.0s.
- Mean-center.
- Two **cascaded 1st-order** Butterworth filters, each bidirectional
  (`filtfilt`): lowpass at `GLM_CYCLE_BANDPASS[1]`=0.6 Hz, then highpass at
  `GLM_CYCLE_BANDPASS[0]`=0.01 Hz — matches `pspm_resp_pp.m` Stage 1 exactly
  (two separate passes, not one joint 2nd-order bandpass design).
- Downsample to `GLM_TARGET_SFREQ`=10 Hz (`scipy.signal.resample`), anti-aliased
  by the 0.6 Hz lowpass already applied.
- Robust z-score (median/MAD) if `GLM_ZSCORE_RAW=True` — preserves cycle
  timing while rescaling amplitude into session-relative SD units without an
  outlier spike inflating the noise floor.

### Stage 2 — Cycle detection (`detect_cycles`)

- 1s median filter (`GLM_MEDIAN_WIN_SEC`) on `raw_z`, then negative
  zero-crossings → inspiration onsets.
- Refractory period `GLM_REFRACTORY_SEC`=1.0s: a candidate onset is dropped if
  its gap to the *preceding* candidate is under this, single pass over the
  original candidate list (matches `pspm_resp_pp.m`'s `ibi<1` rule exactly,
  including its edge case where two close-together candidates in a row can
  both be dropped).
- Per cycle: **RP** = onset-to-onset duration (s); **RA** = peak-to-trough of
  the cycle window measured on `signal_for_pipeline` (the Stage-0-cleaned,
  *unfiltered* native cache — never on `raw_z`), mirroring `pspm_resp_pp.m`'s
  `resp`/`newresp` separation; **RFR** = RA/RP.

### Stage 3 — Artifact layer, cycle-level (`GLM_ARTIFACT_METHOD`, default `"manual_exclude"`)

Three interchangeable handling modes for amplitude-spike cycles (Hampel
identifier: `RA > median(RA) + GLM_ARTIFACT_K × MAD(RA) × 1.4826`
(`GLM_ARTIFACT_K`=3.5), upper bound only, needs ≥3 cycles):

- `"manual_exclude"` **(current default)** — no cycle-level action beyond
  `SUBJECTS_EXCLUDE` (see Rejection); cycles pass through unchanged.
- `"hampel_drop_cycles"` — flagged cycles are removed before Stage 4
  interpolation and their span added to `missing_spans`.
- `"hampel_reject_trials"` — cycles are kept, but any trial whose window
  contains a flagged cycle's onset is rejected downstream (`amplitude_artifact`,
  see Rejection).

`SUBJECTS_EXCLUDE` is always honoured regardless of this setting (whole
subject `"SUBJ"` or single session `"SUBJ/sess"` keys) — it parks hand-flagged,
session-wide contamination the session-relative Hampel bound structurally
can't catch (a uniformly-corrupted session has a huge median too).

### Stage 4 — Continuous series (`build_continuous_series`)

- Interpolate each metric's cycle values (assigned to the *following*
  inspiration onset) to the full session at 10 Hz.
- Per-metric causal (`lfilter`, one-pass) Butterworth bandpass:
  high-pass `GLM_FINAL_HP` = {RP: 0.01, RA: 0.001, RFR: 0.001} Hz, low-pass
  `GLM_FINAL_LP`=1.0 Hz shared. PsPM does not share one high-pass across the
  three channels.
- `missing_spans` (Stage 0 QC excursions + Stage 3 dropped cycles) are
  NaN-blanked **after** filtering, not before — NaN-ing pre-filter would poison
  every sample downstream via the causal filter's own feedback; the
  interpolated line is needed only to keep that recursion well-behaved. GLM
  fitting already masks NaNs out of its regression.

### Stage 4.5 — Per-subject z-scoring (`zscore_subject_series`)

Removes between-subject anatomy/calibration variance from RA and RFR before
fitting. `signal_raw` is in each recording's own native units — this
project's airflow channel comes through as raw EGI/MNE Volts (session
peak-to-trough range ~0.015 in one checked example), never rescaled — so
absolute RA/RFR magnitude differs by rib-cage-to-lung-volume mapping and
sensor contact, not just reactivity (Bach et al. 2016, §3.4/Discussion:
*"RA and RFR measures contain between-subject variance of no interest, due
to individual anatomy"*).

- Controlled by `GLM_ZSCORE_METRIC` = `{RP: False, RA: True, RFR: True}`. RP
  is left raw — it's already an absolute physical unit (seconds), not a
  device-calibration-dependent one.
- For each subject, pools every finite sample of `series['RA']` (and
  `series['RFR']`) across **all of that subject's sessions** (Eve + Mor
  together), computes one `(mean, std)` pair per subject, then applies
  `z = (x - mean) / std` to each session's series individually. Pooling
  across sessions — never per-session — is required: z-scoring each session
  on its own would force every session to mean 0, erasing the Eve-vs-Mor
  contrast the pipeline exists to measure.
- Runs after Stage 4 (continuous series) and before Stage 5 (GLM fit): the
  fit sees z-scored RA/RFR input, so `score_ra`/`score_rfr` come out in
  z-units (dimensionless, comparable across subjects) instead of native
  Volts. `score_rp` stays in seconds throughout.
- Pure function — takes `{session_key: series_dict}` for one subject,
  returns a new dict, never mutates its input. The QC notebook
  (`airflow_qc_show.ipynb`) memoizes the *raw* (pre-z-score) series per
  session; every cell that refits the GLM calls this helper fresh each time
  (`get_subject_zscored_series`) rather than caching a z-scored copy, so
  repeated calls can't double-apply the transform.

### Stage 4.6 — Amplitude ceiling (`apply_amplitude_ceiling`)

`GLM_MAX_ABS_Z` (default `3.0`; `None` disables). Any sample whose `|RA(z)|`
or `|RFR(z)|` exceeds the ceiling is set NaN across **all** metrics of that
session's series, and cleared in Array A's `valid` mask so the trial's
reported `valid_fraction` reflects it.

Must run here, not inside Array A: Array A operates on `raw_z` and cycles,
before the series exists and before Stage 4.5 pools Eve+Mor — RA(z) is not
defined until after z-scoring.

Reference distribution (all 20 subjects, finite RA(z) samples entering the
GLM): 99th percentile is 3.27 (Eve) / 3.18 (Mor); 99.9th is 6.65 / 6.26. A
ceiling of 3.0 therefore removes ~1.3% of Evening and ~1.2% of Morning
samples — 0.050% vs 0.020% at a ceiling of 8, 1.289% vs 1.189% at 3.

**This is a score-dependent gate and that is the problem with it.** It removes
data on the basis of the very quantity being measured, so the surviving
distribution is shaped by the threshold rather than by the physiology, and it
truncates the top of the amplitude range in whichever session genuinely
reaches higher (per-session max RA(z): Eve mean 6.99, Mor mean 5.61). Set from
the cohort distribution a priori; never moved to change an outcome. Whether
this stage should exist at all is an open question — a cycle-level verdict
made before the series is built is the design the redesign replaces it with
(`Airflow/_scratch/cycle_rejection_spec.md`).

### Stage 5 — GLM fit (`fit_trial_glm` / `fit_pooled_session_glm`)

Canonical response function, a Gaussian `h(t) = exp(-(t-τ)²/(2σ²))`
(`GLM_RF_PARAMS`):

| Metric | τ (s) | σ (s) | Derivative term? (`GLM_USE_DERIVATIVE`) |
|---|---|---|---|
| RP  | 4.20 | 1.65 | No  |
| RA  | 8.07 | 3.74 | Yes |
| RFR | 6.00 | 3.23 | Yes |

Basis columns `[CRF, dCRF/dt]` are orthogonalized (serial Gram-Schmidt,
`spm_orth.m`-equivalent) and unit-range normalized before the intercept is
added — matches every `pspm_bf_r*rf_e.m` basis file.

`GLM_ESTIMATION` (**current default: `"pooled_session"`**) selects the
architecture:

- `"per_trial"` — one independent regression per trial over its full
  `[baseline_range_sec[0], response_range_sec[1])` window:
  `y = β₀ + β₁·CRF(t) + β₂·dCRF/dt(t)`, solved by Moore-Penrose pseudoinverse
  (`np.linalg.pinv`). Score = β₁ (`score_rp`/`score_ra`/`score_rfr`).
- `"pooled_session"` — PsPM's `pspm_glm.m` architecture: one design matrix for
  the **entire session**, one event train per condition (grouped by
  `GLM_CONDITION_FIELD`, default `"label"` = Negative/Neutral valence) convolved
  with the same orthonormalized basis (fixed -10s..+30s support), the design
  columns causally high-passed per-metric and mean-centred, then
  re-orthogonalized per condition block before one session-wide solve. Each
  trial is written its condition's β_c as `score_<metric>` — one
  number per condition per session, not per trial.

Every trial with any valid signal contributes an event to the design. Bad data
is removed at **sample** level (NaN in `series`), never by dropping a trial's
onset column: dropping the column while its samples remain in `y` would leave a
real response with no regressor to explain it, inflating the residual and
biasing β. Only a trial with zero valid samples is skipped and keeps
`score_rp`/`score_ra`/`score_rfr` = NaN. `GLM_PRIMARY_METRIC="RA"` names the
metric the pipeline is validated against — the chain whose every stage has to
hold up (see `GLM_METHOD_FOUNDATIONS.md` §7); RP/RFR/composite are carried but
are not the reference.

### Stage 6 — Orchestration (`run_glm_scoring` / `_run_glm_analysis`)

- Two passes per subject, required by Stage 4.5: **Pass A** builds every
  session's cycles/series (Stages 0-4) without fitting anything; then Stage
  4.5 z-scores RA/RFR pooled across those sessions; then **Pass B** builds
  each session's trial list and runs the Stage 5 fit on the (now z-scored)
  series. Z-scoring can't be folded into the original single-session pass —
  it needs every session's series in hand first.
- Session crop for the pooled design:
  `[min(d105_time)-GLM_PRE_FIXATION_SEC(15s), max(code_time)+GLM_POST_CODE_SEC(15s)]`;
  everything outside is masked missing before the pooled solve.
- PASS 1 sets labels/condition and a display-only `baseline_mean`. PASS 2
  records each trial's `valid_fraction` (Array A coverage within
  `[onset, onset+QC_TRIAL_WINDOW_SEC)`, diagnostic only) and marks
  `rejected` **solely** when that fraction is zero, then either the per-trial
  fit runs inline or the pooled fit runs once afterwards.

---

## Layer 2 — BxB / peak-excursion (`airflow_amp_processor.py`, secondary/legacy)

Runs NK2 (`nk.rsp_process`, `RSP_CLEAN_METHOD`/`RSP_PEAK_METHOD_CLEANING` both
`"khodadad2018"`) on the full-session Layer 1 signal, then cuts one
ground-truth epoch per trial (`[d105_time, response_range_sec[1])`, t=0 at
`code_time`).

Per-trial breath-by-breath (BxB) extraction:
- **Baseline breath** = last NK2 peak at or before t=0.
- **Response breath** = first NK2 peak after the first NK2 trough following
  t=0 (the first *complete* post-probe breath).
- `session_breath_amp` = median of each trial's baseline NK2 `RSP_Amplitude`
  across the session (only positive, finite values).
- Score: `(response_amp - baseline_amp) / session_breath_amp`.
  `score_ventilation`:
  `(response_amp·response_rate - baseline_amp·baseline_rate) / session_breath_amp`.

Shares the shape-centroid rejection gate implementation with the GLM path
(same `AIRFLOW_SHAPE_*` config names/values, applied to `epoch_clean_wide`
instead of `raw_z`).

**`SUBJECTS_EXCLUDE` asymmetry (verified in code):** `apply_analysis_params`
only checks `subj in subjects_exclude` — a bare subject-ID match. It does
**not** check the `"SUBJ/sess"` per-session form the GLM path checks
(`f"{subj}/{sess_key}" in subjects_exclude`). Given the current
`SUBJECTS_EXCLUDE` list (`"ES29/eve"`, `"MG14/mor"`, `"MS13/eve"` are
per-session entries), those three sessions are excluded from the GLM path but
**not** from the BxB path — a real parity break between the two scoring
methods, not just a documentation gap.

---

## Rejection

> **⚠ This section is stale (checked 2026-08-12).** It describes a per-SAMPLE
> "Array A" scheme that is **not the code on disk**. What actually runs today is a
> per-TRIAL gate set — eleven booleans OR-ed together in
> `airflow_glm._run_glm_analysis` — plus `airflow_qc.flag_deep_breaths`,
> `airflow_qc.flag_excursions` and `GLM_ARTIFACT_METHOD`. See
> `Airflow/_scratch/STATUS.md` §A for what runs, and
> `Airflow/_scratch/cycle_rejection_spec.md` for the cycle-based design that
> replaces both. Read those two before trusting anything below.
>
> The principle that survives all three schemes: **one verdict per unit of signal,
> from that unit's own measured properties, blind to the session key and the trial
> label — consumed downstream, never re-derived.**

**The GLM path rejects SIGNAL, not trials.** Validity is a per-sample property
of the continuous session signal; bad samples become NaN and drop out of the
fit. There is no threshold that discards a whole trial. This replaced the old
per-trial gate set (`no_cycles_found`, `noisy_baseline`, `flat_signal`,
`flat_response`, `rate_artifact`, `atypical_shape`, `cycle_gap`,
`low_information`, `low_coverage`) — those were computed over
`[baseline_start, response_end)`, a window that bleeds into the next trial's
fixation period (true min trial→trial gap 12.49 s vs. a ~30 s response
window), and imposed a baseline-vs-response split that has no meaning on one
continuous signal deconvolved as a single GLM. `noisy_baseline` and the
baseline half of `flat_signal` are discarded as *concepts*, not just gates:
there is no baseline window left to be noisy or flat about.

**Array A** (`compute_sample_validity`) — `bool[n_samples]`, whole session,
from local signal properties only, no trial context:

| Component | Config | Logic |
|---|---|---|
| cycle coverage | — | Samples outside any detected cycle are invalid (`no_cycle_coverage`). |
| amplitude spike | `GLM_ARTIFACT_K`=3.5 | Per-cycle Hampel on RA vs. the session median/MAD, **upper bound only** (`amplitude_artifact`). |
| rate plausibility | `GLM_MIN_RATE_THRESHOLD`=5, `GLM_RATE_ARTIFACT_THRESHOLD`=40 bpm | `60/RP` outside that range (`rate_implausible`). |
| flat / noisy | `GLM_MIN_STD_RATIO`=0.1, `GLM_Z_SCORE_THRESHOLD`=3.0 | Rolling local std of `raw_z` (window = session median RP) against a robust median/MAD reference (`flat_or_noisy`). |

Invalid runs are converted to spans (`_invalid_mask_to_spans`) and fed through
the existing `missing_spans` → NaN pathway in `build_continuous_series`. Both
`zscore_subject_series` and the GLM solve already gate on `np.isfinite`, so no
separate masking is needed downstream. **Stage 4.6's amplitude ceiling
(`GLM_MAX_ABS_Z`) clears `valid` too**, after z-scoring.

**Per-trial bookkeeping** — each trial records `valid_fraction` (Array A
coverage within `[onset, onset+QC_TRIAL_WINDOW_SEC)`, post-onset only,
**diagnostic**). `rejected` is set **only** when that fraction is zero
(`rejection_reason="no_valid_signal"`) — a trial with no data can't be handed a
pooled β it contributed nothing to. It is not a gate. Current rates: Evening
721/730 kept (98.8%), Morning 743/771 (96.4%).

`score_ceiling` (`GLM_SCORE_MAX`, default `None` = **disabled**) is the one
remaining trial-level rejection and is deliberately off: rejecting a trial
because of the score it produced is circular.

The **BxB path** is unchanged and still uses the old per-trial gate set with
its own `AIRFLOW_*` config names (`AIRFLOW_Z_SCORE_THRESHOLD`,
`AIRFLOW_MIN_STD_RATIO`, `AIRFLOW_POST_MIN_STD_RATIO`,
`RSP_RATE_ARTIFACT_THRESHOLD`, `AIRFLOW_SHAPE_*`, `AIRFLOW_SCORE_MAX`=3.0
enabled). **The two paths are no longer comparable on rejection** — this is now
a structural difference, not a naming one.

Not a per-trial gate, but shapes what data reaches Array A: Stage 0's
QC excursion masking and `flag_deep_breaths` (see Layer 2 — GLM) blank spans
of signal before scoring, independent of `GLM_ARTIFACT_METHOD`.

`AIRFLOW_AMPLITUDE_Z_THRESHOLD` in `airflow_config.py` is set but not read
anywhere in the current code — a dead/unused config value, not an active
gate.

`SUBJECTS_EXCLUDE` is a hand-maintained, unresolved placeholder (see
`CLAUDE.md`), honoured asymmetrically between the two paths (see BxB section
above).

---

## Files

| File | Role |
|---|---|
| `airflow_config.py` | All tunable parameters — Layer 1 cache constants, GLM Stage 1–6 constants, QC constants, rejection thresholds, run/output paths. |
| `airflow_glm.py` | Layer 1 cache build (`process_session`) + GLM Layer 2 scoring (Stages 1–6, primary path). |
| `airflow_amp_processor.py` | BxB/legacy Layer 2 scoring only (`apply_analysis_params`, secondary path). No longer builds the cache itself. |
| `airflow_qc.py` | Session-level QC helpers: excursion masking, robust stats, deep-breath flagging, cohort breath-size reference / flat-session report. |
| `airflow_main.py` | Orchestration: cache load/save, subject/session discovery, dispatch to GLM or BxB scorer, CSV + per-trial review-plot output. |
| `airflow_mff_show.py` | Diagnostic-only: shows one raw `.mff` straight off the drive, no Layer 1/2 processing — for sessions that failed to make it into the cache. |
| `airflow_glm_show.ipynb` | Per-subject/session GLM diagnostic plots. |
| `airflow_amp_show.ipynb` | Per-subject/session BxB diagnostic plots. |
| `airflow_qc_show.ipynb` | QC diagnostics (excursion masking, breath size, flat-session report) per subject/session. |
| `GLM_METHOD_FOUNDATIONS.md` | What the GLM score means, CRF/basis assumptions, per-trial vs. pooled-session estimation, and what has to be demonstrable before a β is worth reading — read before citing `score_ra`. |
| `DISCUSSION.md` | Open items and audit trail for the GLM method, tracked against PsPM source. |
| `_scratch/STATUS.md` | **Start here** — what is actually on disk, the open decisions, what is unimplemented. |
| `_scratch/cycle_rejection_spec.md` | The cycle-based rejection design: call chain, signatures, invariants, what gets deleted. Nothing in it is implemented. |
| `_scratch/verify_spec_numbers.py` | Regenerates every number in the spec straight from the cache; prints PASS/FAIL per line. |
| `AMPLITUDE_NORMALIZATION_REVIEW.md` | Whether GLM's raw-unit β₁ should be rescaled for subject dynamic range, checked against PsPM + normalization literature. |
| `GLM_SIGNAL_REVIEW.md` | Signal-processing-only summary of the GLM chain, referenced against PsPM v7.0.0 MATLAB source line numbers. |
