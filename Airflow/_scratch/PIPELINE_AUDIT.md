# Airflow GLM — audit, laid out along the diagram

2026-08-12. Same findings as before, re-organised so each one sits under the stage you can see in
`cycle_rejection_logic.html`. Measured on the current tree and the Layer-1 cache, 38 sessions
after `SUBJECTS_EXCLUDE`. Nothing here was decided from a p-value.

Legend: **CLEAN** = checked, works, don't re-audit · **OPEN** = needs a decision or a fix ·
**NOT BUILT** = design exists, code does not.

---

## Row 1 · `signal_raw` → onsets  (diagram row 1)

**CLEAN.** Cache integrity: 0 trials missing `code_time`, 0 missing `d105_time`, 0 missing
`subjective_label` (694 Negative / 769 Neutral across 1463 trials). No session has fewer than 2
detected cycles. Detection itself (`detect_cycles`, median filter + zero-crossing + 1 s
refractory) is unchanged and PsPM-faithful.

## Stage S1 · find the invalid values

**OPEN — the gate is blind.** `flag_excursions` compares each sample to a **1 s rolling median**,
which rises with any artifact longer than about a second. Over all of RP06/mor it fires
**0 times**, although that session contains a peak of 14.93 × the session breath size at
t = 1340.40 s. Decide: change the reference, lengthen the window, or drop sample-level detection
and rely on the cycle-level amplitude gate (which does catch it, at 7.10 × the session median).

## Stages S2–S4 · verdict per breath → gap-fill → recovery + knot

**NOT BUILT.** `classify_cycles`, `apply_neighbour_rules`, `admit_trials` and every `CYCLE_*`
threshold are absent from every `.py` file. The blanking geometry (drop the knot, blank from the
last surviving knot to the next surviving one) is settled and verified on MS18/eve — it is the
implementation that is missing, not the rule.

## Row 4 · `y`, the target

**CLEAN, mechanically.** Two things I expected to be wrong and are not:
- NaN is written **after** `lfilter` (`airflow_glm.py:449`), so a hole cannot propagate forward
  through the causal filter's feedback.
- `np.interp` extends the series with a constant outside the knot range (`left=knot_v[0],
  right=knot_v[-1]`, `:437`). Measured across all 38 sessions: **0.0 s** of that fabricated
  constant survives as finite `y` — the session crop and the `n_samples` truncation already
  cover both ends.

## ⚠ A stage the diagram does not show · per-subject z-scoring

Between row 4 and the fit, `zscore_subject_series` (`:456`) rescales RA and RFR using the pooled
mean and standard deviation of **both** of a subject's sessions. It is not in the diagram and it
should be — it is the only place one session's data touches the other's numbers.

**OPEN — it is not robust.** It uses `np.mean` / `np.std`, while the rest of the codebase uses
median/MAD (`airflow_qc.robust_sd`, `phase1_filter_downsample`). Measured `std / robust_sd` of the
pooled RA series: median **1.51**, p90 **2.34**, max **2.52** (ES29). Switching to median/MAD is a
two-line change, but it rescales each subject by a *different* factor (1.2–2.5×), so it changes
how much each subject weighs in any group comparison. Pooling across a subject's two sessions is
correct and must stay — per-session z-scoring would normalize away the Evening-vs-Morning
contrast itself.

## Row 5 · event train and trial admission

**OPEN — what a rejected trial blanks.** Not a question about epoch length: trial boundaries come
from the triggers (D105 → next D105) and are correctly variable. Measured over all 1463 trials:
**13.02 s min, 23.15 s median, 31.65 s max.** The question is only how much of `y` to NaN when a
trial loses its event, and the CRF decides it:

| metric | τ, σ | % of peak still standing at 12 s | drops under 10 % of peak at |
|---|---|---|---|
| **RA** | 8.07, 3.74 | **57.6 %** | **16.0 s** |
| RFR | 6.00, 3.23 | 17.8 % | 12.9 s |
| RP | 4.20, 1.65 | 0.0 % | 7.7 s |

So blanking only the 12 s admission window would leave **more than half** of the removed trial's
modelled RA response sitting in the data with no regressor — straight into the residual. Blanking
the full trial window instead removes 13–32 s, a different amount every trial, and reaches into
the *next* trial's regressor support (the CRF starts 10 s before onset).

Recommended: keep `TRIAL_WINDOW_SEC = 12 s` for **admission** (counting breaths near the
stimulus), and blank a separate, fixed **response-support** span of ~16 s for RA — where the CRF
falls under 10 % of peak. Two different quantities that the spec had conflated into one.

## Row 6 · design columns

**RESOLVED BY YOUR CALL — one β per session.** Today `GLM_CONDITION_FIELD = "label"` splits trials
into Negative/Neutral and fits one β per condition. You want a single score per session (two per
subject). That is already a supported value: `GLM_CONDITION_FIELD = "none"` sets every trial to
condition 1 (`airflow_glm.py:1015`), giving one event train, one CRF column, one β.

It also **dissolves the weighting problem** entirely. With two conditions, a session's value was
`(n_neg·β_neg + n_neu·β_neu) / (n_neg + n_neu)` — weighted by how many trials of each condition
*survived rejection*. The admitted-Negative fraction ran 0.21 → 0.74 across sessions, and the
resulting distortion reached 0.1072 on ML28/eve. With one condition every trial carries the same
β, the mean is that β, and rejection can no longer leak into the score through the trial mix.

Cost, stated plainly: the Negative-vs-Neutral contrast stops being a GLM output. If it is ever
wanted it becomes a separate model, not a column in this one.

## Row 7 · the solve

**CLEAN.** `fit_pooled_session_glm` computes `valid = isfinite(y) & all(isfinite(X))` (`:687`) and
solves `pinv(X[valid]) @ y[valid]` — holes are dropped, the grid is never trimmed, timestamps
never move. The forbidden `hampel_reject_trials` + `pooled_session` combination raises rather than
running silently (`:719`).

Latent, not reachable today: `onsets_by_cond.setdefault(t["condition"], …)` (`:657`) would build a
separate design column per trial if any condition were NaN, since `NaN != NaN`. No labels are
missing, so it cannot fire — and under one condition it disappears anyway.

---

## What is left, in order

1. Set `GLM_CONDITION_FIELD = "none"` — one β per session. Removes the weighting problem.
2. Split the two spans: admission 12 s, blanking ~16 s (RA's CRF support). Update spec §5.
3. Decide S1 (the blind sample gate), then the thresholds, then build S2–S4.
4. Robust z-scoring, and add the z-scoring stage to the diagram.
5. `/cross-pipeline-audit` and `/pspm-respiration-audit` at the end, against a stable pipeline.
