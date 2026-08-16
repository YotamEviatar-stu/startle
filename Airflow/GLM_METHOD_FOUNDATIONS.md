# GLM Method — Foundations

What anyone using `score_rp`/`score_ra`/`score_rfr` needs to understand
before trusting or writing up those numbers. This is specific to how
`airflow_glm.py` actually implements the method (Bach et al. 2016, "A
linear model for event-related respiration responses") — not a generic
GLM/convolution tutorial. See `README.md` for the mechanical step-by-step
pipeline; this file is about the assumptions underneath it.

---

## 1. What the score actually measures

For each trial, `score_ra` (and `score_rp`/`score_rfr`) is **a scaling
factor**: how much you'd need to stretch a fixed, known-in-advance response
shape to best match this trial's real breathing curve. It is:

- **Not** the biggest breath in the window.
- **Not** a before/after (baseline vs. response) difference.
- **A single number saying "how strongly did this trial's data match the
  expected shape of a real reaction,"** after first removing whatever flat
  level this person's breathing happened to sit at (see §3).

## 2. The canonical response function (CRF) — imported, not discovered

The "expected shape" (`generate_canonical_rf` in `airflow_glm.py`) is a
Gaussian bump: `h(t) = exp(-(t-τ)²/(2σ²))`, with `τ` (peak time) and `σ`
(width) fixed to published values (`GLM_RF_PARAMS` in `airflow_config.py`):
RP peaks at 4.20s, RFR at 6.00s, RA at 8.07s post-trigger.

**These numbers come from the literature, not from fitting this dataset.**
This matters for two reasons:
- If this population's true response timing differs meaningfully from the
  published values, every score is being scaled against a shape that
  doesn't quite match reality — a systematic bias, not random noise, and
  not something a p-value will reveal on its own.
- The same fixed shape is used for **every trial, every subject, every
  session** — this is the "Linear Time-Invariant" (LTI) assumption. The
  model has no way to represent "the response shape itself changed" (e.g.
  from habituation or fatigue across the session) — it can only report a
  lower score, indistinguishable from "the response was genuinely weaker."

## 3. Separating "how high up" from "what shape" (the baseline patch)

Before the observed curve is compared to the CRF, the model gives itself a
free flat-line adjustment (the intercept, β₀) to absorb whatever constant
level this person's breathing sits at — so a naturally deep breather and a
naturally shallow breather are compared on the same footing: only the
bump-shaped *deviation* from each person's own resting level gets scored,
never the absolute level itself.

## 4. The regression and its design matrix

Concretely, for one trial, the model solves for weights (β₀, β₁, β₂) in:

```
observed_curve(t) = β₀·(flat line) + β₁·CRF(t) + β₂·dCRF/dt(t)
```

- **β₀** — the flat-line weight (§3).
- **β₁** — `score_ra`/`score_rp`/`score_rfr`: the CRF's weight, i.e. the
  answer to "how strongly did this trial match the expected shape."
- **β₂** — only present for RA/RFR (`GLM_USE_DERIVATIVE` in
  `airflow_config.py`), a secondary adjustment for the CRF's slope, letting
  the fit account for small timing/shape mismatches without contaminating
  β₁.

**Two requirements for β₂ to not distort β₁:**
- The CRF and its derivative must be **orthogonalized** first (mirroring
  PsPM's `spm_orth`) — otherwise the two overlapping columns split credit
  for the same piece of data somewhat arbitrarily, and β₁ stops being a
  clean, comparable number across trials. See `orthogonalize_and_normalize_basis()`.
- There must be **enough real (non-NaN) time samples** in the trial window
  to solve for as many unknowns as there are columns — `fit_trial_glm`
  returns "no score" (`NaN`) rather than a manufactured number when a
  window is too short or too gappy.

## 5. Per-trial scoring vs. PsPM's session-wide design — know which one you're running

PsPM's own `pspm_glm.m` is described as *"similar to standard analysis of
fMRI data"*: it builds **one design matrix for the entire session**,
pooling every trial of a given condition into shared regressor columns,
and solves **one regression producing one beta per condition** (e.g. one
number for "all negative trials, this session"), not one per individual
trial.

`airflow_glm.py` supports **both** architectures, selected by
`GLM_ESTIMATION` in `airflow_config.py` (Layer 2, no reload):

- `"per_trial"` — an independent regression per trial
  (`fit_trial_glm`), each scoped to just that trial's own short window,
  producing one score *per trial*.
- `"pooled_session"` **(current default)** — PsPM's architecture (`fit_pooled_session_glm`): one
  design matrix for the whole session, one regressor set per condition
  (valence label, over that session's accepted trials), a single solve per
  metric → **one beta per condition**, written back onto each accepted
  trial. This is the faithful `pspm_glm.m` estimator.

**This is a real, deliberate architectural choice, not a bug** — state
which path produced any number you report:
- Per-trial scoring is what lets this project run per-trial diagnostics and
  the trial-level scatter/timecourse plots. It produces one value per trial;
  the pooled path produces one β per condition, written back onto each
  admitted trial, so a per-trial mean over a pooled session collapses to a
  trial-count-weighted mean of the Neg/Neu betas. The two are not
  interchangeable and must never be mixed in one summary.
- Per-trial is **not** what Bach et al. 2016 describes; a reader familiar
  with that paper would expect the pooled, per-condition estimates. Under
  both paths the response-function shapes, orthogonalized derivative and
  per-modality filters are faithfully reproduced; only `"pooled_session"`
  also reproduces the estimation architecture.
- The choice is fixed a priori from the method, never selected by comparing
  what each path does to a downstream contrast.

## 6. What this method deliberately does NOT do

- It does not adapt the expected response shape to this data — see §2.
- It does not model overlap between nearby trials' response windows —
  each trial is scored in isolation, so if a real physiological response
  genuinely bleeds past its own trial's window into the next one, this
  method has no mechanism to detect or correct for that.
- It does not distinguish "no response" from "a response, but not matching
  the fixed template's shape" — both produce a low score.

## 7. What has to be true before a `score_ra` is worth anything

`GLM_PRIMARY_METRIC` ("RA") names the metric this pipeline is **validated
against** — the one whose whole chain has to hold up, not a statistical
endpoint. Before any β is worth reading, each link below must be
demonstrable on the real recordings:

1. **The filter passes respiration and nothing else.** Check the PSD of the
   filtered trace against the raw one, per subject — not the group mean.
2. **One onset marks one real breath.** `detect_cycles` runs on `raw_z`;
   an onset that spans several breaths (baseline drift stopping the trace
   from crossing zero) or splits one is a detection failure, and it shows up
   as an implausible period, not as an error.
3. **`RP`/`RA`/`RFR` are measured on `signal_raw`, never on `raw_z`.** This
   is `pspm_resp_pp.m`'s `resp`/`newresp` split. `raw_z` is bandpassed and
   z-scored and exists only to *time* the onsets; measuring amplitude on it
   measures the filter's output. This has been wrong in this repo before.
4. **Rejection is one verdict per breath, with a named reason**, consumed by
   later stages rather than re-derived from window statistics — and blind to
   the session key, `label`, and `has_sound`.
5. **The series is blanked, not trimmed**, with NaN written *after* `lfilter`
   so the causal recursion does not propagate the hole, and
   `fit_pooled_session_glm` solving on finite rows only (it already does —
   `np.linalg.pinv(X[valid]) @ y[valid]`).
6. **The CRF actually resembles this cohort's response** — see §2. Fixed
   τ/σ imported from the literature means a systematic scaling bias if this
   population's timing differs, and no amount of statistics downstream will
   surface it.

`Airflow/_scratch/METHOD.md` §10 turns 3–5 into checkable
invariants; `Airflow/_scratch/verify_spec_numbers.py` regenerates the
supporting numbers straight from the cache.

**On earlier numbers in this file.** Previous versions of §7 reported a
pre-registered Evening-vs-Morning Wilcoxon on `score_ra` as the section's
subject. That framing is retired: the pipeline is evaluated on whether it
processes the signal correctly, not on what it does to a group contrast, and
every p-value once quoted here came from rejection code that is no longer on
disk (see `Airflow/CLAUDE.md` → *Historical record*). Do not reinstate a
result section here, and do not choose the estimation path, the metric, or a
threshold by watching a contrast.

## Related, ongoing tracking

See `DISCUSSION.md` for open items being actively tracked and resolved
against this method (orthogonalization status, per-modality filter
differences, cycle-detection design, etc.) and the `pspm-respiration-audit`
skill for the underlying PsPM source-level ground truth these foundations
are checked against.
