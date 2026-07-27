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

- `"per_trial"` **(default)** — an independent regression per trial
  (`fit_trial_glm`), each scoped to just that trial's own short window,
  producing one score *per trial*.
- `"pooled_session"` — PsPM's architecture (`fit_pooled_session_glm`): one
  design matrix for the whole session, one regressor set per condition
  (valence label, over that session's accepted trials), a single solve per
  metric → **one beta per condition**, written back onto each accepted
  trial. This is the faithful `pspm_glm.m` estimator.

**This is a real, deliberate architectural choice, not a bug** — state
which path produced a reported number in any write-up:
- Per-trial scoring is what lets this project run per-trial diagnostics and
  the trial-level scatter/timecourse plots; both paths still feed the same
  paired, within-subject Wilcoxon (the confirmatory cell averages
  `score_<metric>` over a session's accepted trials, which for the pooled
  path collapses to a trial-count-weighted mean of the Neg/Neu betas).
- Per-trial is **not** what Bach et al. 2016 describes; a reader familiar
  with that paper would expect the pooled, per-condition estimates. Under
  both paths the response-function shapes, orthogonalized derivative and
  per-modality filters are faithfully reproduced; only `"pooled_session"`
  also reproduces the estimation architecture.

## 6. What this method deliberately does NOT do

- It does not adapt the expected response shape to this data — see §2.
- It does not model overlap between nearby trials' response windows —
  each trial is scored in isolation, so if a real physiological response
  genuinely bleeds past its own trial's window into the next one, this
  method has no mechanism to detect or correct for that.
- It does not distinguish "no response" from "a response, but not matching
  the fixed template's shape" — both produce a low score.

## 7. The confirmatory result — pre-registered, and currently null on this path

The single pre-specified test (`GLM_PRIMARY_METRIC` in `airflow_config.py`, run
by the confirmatory cell in `airflow_glm_show.ipynb`): per-subject mean
`score_<metric>` over accepted trials, **Evening vs Morning**, paired Wilcoxon,
one-sided H1 Eve>Mor (direction fixed a priori in `CLAUDE.md`).

As of this analysis (24 subjects cached, N=17 with both sessions,
`SUBJECTS_EXCLUDE` always parking DA01/ER23/YL26 plus default
`GLM_ARTIFACT_METHOD="hampel_reject_trials"` layered on top for the rest),
the pre-registered test on the
**primary metric (RA)** is **null under both estimation paths** (§5):

| primary | path | Eve>Mor subjects | two-sided p | one-sided p |
|---|---|---|---|---|
| **RA** (primary) | per_trial       | 9/17  | 0.89 | 0.57 |
| **RA** (primary) | pooled_session  | 8/17  | 0.78 | **0.39** |
| RFR | per_trial       | 6/17  | 0.43 | 0.80 |
| RFR | pooled_session  | 7/17  | 0.82 | 0.61 |
| RP  | per_trial       | 10/17 | 0.35 | 0.18 |
| RP  | pooled_session  | 11/17 | 0.10 | 0.049 |
| composite | per_trial | 9/17  | 0.43 | 0.22 |

Reading this honestly:
- **The primary endpoint (RA) is still null.** The pooled session-wide GLM moves
  RA in the hypothesised direction and roughly halves the one-sided p (0.57 →
  0.39), consistent with it being the more powerful estimator §5 predicts — but
  it does not reach significance. That is the number to report for the Airflow
  GLM channel: **no significant Eve>Mor effect.**
- **RP under `pooled_session` crosses 0.05 one-sided (p=0.049, 11/17), but RP is
  a SECONDARY/exploratory metric, not the pre-specified primary**, and it does
  not survive two-sided (p=0.10). Quoting it as "the result" would be exactly the
  multiplicity tailoring the `GLM_PRIMARY_METRIC` discipline exists to prevent.
  Record it as an exploratory observation worth a pre-registered follow-up (RP =
  breath-period/deceleration), not a confirmatory finding.
- Per-trial RA β-weights sit essentially at zero (medians ~1e-4); the pooled
  betas are larger and better-conditioned (one solve over a condition's trials
  instead of averaging ~17 near-zero single-trial betas), which is why the pooled
  path shifts the p-values without changing which trials are accepted.

Recorded, not hidden. The methodology is PsPM-faithful and the pooled
session-wide GLM (§5) is now **implemented** (`GLM_ESTIMATION="pooled_session"`),
not just noted. Remaining legitimate, NON-p-hacking levers — pre-register before
running, never pick by watching the p:

- **The peak-excursion (BxB) scorer** (`airflow_amp_processor.py`) — a different,
  legitimate design PsPM does not cover; source of the earlier non-null numbers.
- **A pre-registered RP follow-up** — if the RP deceleration signal above is
  judged physiologically motivated a priori, promote it to primary in a fresh
  pre-registration; it cannot be claimed from this exploratory pass.

Tuning gates/thresholds — or picking the estimation path or metric by watching
the Eve-vs-Mor p — is NOT on that list.

## Related, ongoing tracking

See `DISCUSSION.md` for open items being actively tracked and resolved
against this method (orthogonalization status, per-modality filter
differences, cycle-detection design, etc.) and the `pspm-respiration-audit`
skill for the underlying PsPM source-level ground truth these foundations
are checked against.
