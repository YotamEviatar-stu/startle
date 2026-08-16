# CLAUDE.md (Airflow/)

Loaded only for sessions working under `Airflow/`. See the project root `.claude/CLAUDE.md` for general Startle conventions.

---

> ## ⚠ READ FIRST — rewritten 2026-08-16
>
> **1 · The code. The cycle-based scheme is IMPLEMENTED and live.** The previous banner said
> the opposite; it was written 2026-08-12 and is obsolete. `airflow_qc.classify_cycles`,
> `apply_neighbour_rules`, `GATE_ORDER`, `airflow_glm.admit_trials`, and the `CYCLE_*` /
> `TRIAL_*` constants are all on disk and run under `CYCLE_REJECTION_ENABLED = True`.
> **The rejection unit is the breath, not the trial.** The eleven per-trial booleans, the
> shape-centroid gate, `nk2_has_cycles`, the Hampel path and `score_ceiling` were DELETED on
> 2026-08-16, along with 15 config constants. `_run_glm_analysis` Pass 2 now reads
> `t["admitted"]` and nothing else.
>
> **2 · The framing.** The sections under *Historical record* judge the pipeline by an
> Evening-vs-Morning p-value and treat condition balance as a pass/fail **gate**. Both are
> superseded. The pipeline is judged on whether it detects, cleans, and scores the respiration
> signal correctly (root `.claude/CLAUDE.md`). Evening vs Morning is a seconds-weighted
> **balance report**: it exposes a condition-biased gate, it never passes or fails one, and it
> is never a tuning target.
>
> **3 · Two specific errors in the historical record below — do not repeat them:**
> - "any fixed `CYCLE_RP_MAX` removes Evening breaths preferentially" is **backwards**.
>   Morning breathes slower (median RP 3.60 s vs 3.30 s), so a fixed ceiling removes more
>   **Morning**: at `CYCLE_RP_MAX = 7.0 s`, Eve 5.52 % vs Mor 9.33 % of in-window breath
>   seconds (Mann-Whitney p = 0.21).
> - The confirmatory p-values (W=36.0 p=0.0569; W=28.0 p=0.0202) came from a scheme that is
>   not this one. **Do not quote them.** The endpoint is recomputed once, at the end, after
>   the rejection rules are frozen.
>
> **4 · Current settings** (`airflow_config.py`): `CYCLE_RP_MAX = 7.0` (no `CYCLE_RP_MIN`),
> `CYCLE_LOGRA_SCALE = 0.2378` / `CYCLE_LOGRA_K = 6.0` → `extreme` at 4.17x session median,
> `CYCLE_LOSTLOCK_MIN_PEAKS = 3`, `CYCLE_MIN_REF_CYCLES = 30`, `TRIAL_MIN_VALID_FRAC = 0.60`,
> `GLM_CONDITION_FIELD = "none"` (one beta per session), `GLM_ESTIMATION = "pooled_session"`.
>
> **5 · Scope.** Only **36.5 %** of detected cycles lie inside the analysis window (6218 of
> 17028; 6.56 h of 18.50 h). Any gate count or balance figure computed over the whole
> recording is ~3x inflated and answers a different question.
>
> **Authoritative instead of this file:** `airflow_glm.py`, `airflow_qc.py`,
> `airflow_config.py` for what runs; `Airflow/_scratch/STATUS.md` for status, open decisions
> and traps; `Airflow/_scratch/cycle_rejection_spec.md` for the design and its invariants.
> Verify with `.venv/bin/python Airflow/_scratch/verify_spec_numbers.py` (ALL CHECKS PASSED)
> and `.venv/bin/python Airflow/_scratch/debug_pipeline.py` (full-cohort invariant check).
>
> Keep the sections below as the historical record of an earlier design — do not read them as
> a description of the code.

---

## Airflow Pipeline — Current State

`glm_deconvolution` (GLM) is the **primary, active line of work** — `SCORING_METHOD` in
`airflow_config.py` defaults to it, and `GLM_PRIMARY_METRIC` names the metric the pipeline
is validated against.
`peak_excursion_normalized` (BxB, "amp") is kept only as a secondary/legacy comparison
path, not a co-equal method. Both share the same Layer 1 cache
(`airflow_glm.process_session`: native-rate 70 Hz anti-alias lowpass only — no highpass;
`signal_raw` deliberately keeps DC content, mirroring PsPM's unfiltered `resp` — →
resample to `CACHE_SFREQ=25 Hz` → ground-truth trial windows from `extras/trial_epochs.py`).
The two paths **no longer share a rejection scheme at all** — the GLM path rejects per
breath cycle, the BxB path still uses the old per-trial gate set under its own `AIRFLOW_*`
config names, which the GLM path never reads. Also asymmetric: `SUBJECTS_EXCLUDE`'s
per-session `"SUBJ/sess"` form is honoured by GLM only. New work, tuning, and audits should
default to the GLM path unless told otherwise.

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

`GLM_PRIMARY_METRIC="RA"` names the metric the pipeline is **validated against** — the one
whose cycle detection, per-breath measurement, series construction, and β estimation must be
defensible end to end. RP/RFR/composite are computed and carried but are not the reference.
The estimation path and the primary metric are fixed a priori from the method (Bach et al.
2016 / PsPM), never chosen by watching an outcome.

## What "validated" means here

The pipeline is finished when each stage can be shown to do what it claims on the real
recordings, in this order: the cascade filter passes respiration and nothing else; cycle
detection puts one onset on one real breath; per-breath `RP`/`RA`/`RFR` are measured on the
untouched `signal_raw` (never on `raw_z`, which is filtered and exists only to time onsets —
`pspm_resp_pp.m`'s `resp`/`newresp` split); rejection is one verdict per breath with a named
reason; the continuous series is blanked, not trimmed, with NaN written after `lfilter`; and
the GLM solves on finite rows only. `Airflow/_scratch/cycle_rejection_spec.md` §4 lists these
as checkable invariants — including **condition-blindness**: no function in this path may read
the session key, `label`, or `has_sound`.

## Communication

The answer-length cap in `~/.claude/CLAUDE.md` (2–3 paragraphs, ~20 sentences) binds
Airflow output too — including `/red-team-review`, `/cross-pipeline-audit` and
`/pspm-respiration-audit` reports, which report at most the 3 highest-impact findings,
one line each. This file is dense because it is a reference to read, not a template for
how long a reply should be. Long-form output goes to a `.md` file under
`Airflow/_scratch/` with the path in the reply.

## Documentation status

| file | status |
|---|---|
| `airflow_glm.py`, `airflow_qc.py`, `airflow_config.py` | **authoritative** — this is what runs |
| `Airflow/_scratch/cycle_rejection_spec.md` | the cycle-based design being proposed (signatures, call chain, invariants, what gets deleted); nothing in it is implemented |
| `Airflow/_scratch/STATUS.md` | the checklist — what is true now, the open decisions, what is unstarted |
| `Airflow/_scratch/verify_spec_numbers.py` | regenerates every number in the spec from the cache (`.venv/bin/python Airflow/_scratch/verify_spec_numbers.py`) |
| `Airflow/README.md`, `GLM_METHOD_FOUNDATIONS.md`, `GLM_SIGNAL_REVIEW.md`, `DISCUSSION.md` | **stale** on rejection — their Stage 1b/4b/6 sections and the Array A/B language describe superseded schemes |
| `Airflow/airflow_glm_input_show.ipynb` | cannot run — imports `airflow_qc.classify_cycles`, which does not exist |

---

# Historical record — superseded design and framing

Everything below is kept for provenance only. It describes (a) code that is not on disk and
(b) the earlier framing in which an Evening-vs-Morning p-value was the endpoint and condition
balance was a pass/fail gate. Neither is current. Do not read any of it as policy, and do not
quote its numbers.

---

## Rejection — the BREATH CYCLE is the single unit (rewritten 2026-08-05) — ⚠ DESCRIBES CODE THAT IS NOT ON DISK

One decision, made once, in `airflow_qc.classify_cycles`. The five overlapping layers that
preceded it (QC excursion interpolation, deep-breath ratio, Array A, `GLM_MAX_ABS_Z`
ceiling, Array B valid-fraction) are **gone**, along with ~20 config constants.

Per-cycle features (`RP`, `RA`, `RFR`, `std`, `kurt`) are all measured on the **untouched
`signal_raw`** — never on `raw_z`, which is filtered and exists only to time the onsets.
This is `pspm_resp_pp.m`'s `resp`/`newresp` split, and it was previously broken (RA read a
QC-interpolated copy). Five gates, in attribution order:

| gate | catches | rule | kind |
|---|---|---|---|
| `unmeasurable` | unusable breath | `RA` non-finite or ≤ 0 | absolute |
| `lost_lock` | span covering several breaths | ≥ `CYCLE_LOSTLOCK_MIN_PEAKS` (3) prominent inspiratory peaks inside one cycle | absolute |
| `rate_implausible` | not a breath | `RP` outside `CYCLE_RP_RANGE` = (2.0, **None**) | absolute |
| `shape` | cough / movement at normal amplitude | excess kurtosis over a **fixed** `CYCLE_KURT_WIN_SEC` (3.0 s) window > `CYCLE_KURT_MAX` = 5.55 | absolute |
| `flat` | dead signal / sensor off | `std` < `CYCLE_FLAT_RATIO` (0.10) × session median std — AASM apnea criterion | session-centred |
| `extreme` | cough / movement by amplitude | two-sided robust z on `log(RA)`, `|z|` > `CYCLE_LOGRA_Z` = 7.0 | session-centred |

`unmeasurable` has never fired (0/17457) and is unreachable: `RA = max−min` over a
non-empty finite segment cannot be non-finite or ≤ 0. It is kept as a guard, not a
working gate.

**`shape` is measured on a fixed 3.0 s window, not on the cycle's own span.** The
variable-length version tracked cycle duration at ρ=0.410 and fired on 0.37% of <3 s
cycles vs 48.5% of 10–20 s ones — it was largely a duration gate wearing a shape gate's
label (OM16/mor: a 94.5 s dead-flat span containing *zero* breaths scored kurtosis 53.05
and was attributed to `shape`). Fixed window: ρ=0.173, 0.71% vs 10.9%. `CYCLE_KURT_MAX`
= 5.55 is the cohort percentile (p97.39) the old 5.0 sat at, so the gate removes the same
*fraction* of cycles and only *which* cycles changed. Matched on the cohort before any
Eve-vs-Mor comparison.

**`lost_lock` is not an upper RP bound.** A genuine sigh-with-pause has RP ≫ median but
still one inspiratory excursion — 71% of cycles even at 15–25 s are single breaths, which
is why the 8.0 s bound was correctly removed. What `lost_lock` catches is the detector
failing to cross zero under baseline drift and emitting one span over several real
breaths (MG14/eve: 41.9 s holding 6; AK12/eve: 31.9 s holding 9). 26 cycles / 498 s
cohort-wide, 20 of which were previously admitted as valid breaths.

**`CYCLE_LOGRA_SCALE` = 0.300 is a frozen COHORT constant, not the session's own MAD.** With
a per-session MAD the same k is a different gate in every recording — measured on this
cohort it ranged from "reject >1.9× the median breath" (RP06/mor) to ">27.8×" (EV15/eve), a
15× spread. The frozen scale puts the cut at 8.2× the session median everywhere = cohort
p99.5, above the sigh band (p99 = 5.3×) and far below artifacts (p99.9 = 41.8×). Calibrated
over 39 sessions / 17828 cycles with leave-one-session-out deviation 2.0%, **before** any
Eve-vs-Mor comparison. Do not retune.

**There is no upper RP bound.** `pspm_resp_pp.m` enforces only `ibi >= 1 s` (the refractory
in `detect_cycles`); its 10 s marker is a plotting flag, not an exclusion. An 8.0 s bound
was tried and removed: cycles above 2× the session median RP carry RA 1.38× normal and RFR
0.49× — one deep breath followed by a hold, i.e. sigh-with-pause physiology, plausibly the
respiratory startle response itself. That bound failed condition balance at p<0.0001.

Session-relative gates are structurally blind to a uniformly corrupted session. The
`degenerate` flag reports that class from absolute-gate evidence only
(`SESSION_MAX_ABSOLUTE_FAIL` = 0.30, or a dead reference statistic). A gate whose session
reference is untrustworthy (< `CYCLE_MIN_REF_CYCLES` = 30 breaths, or zero MAD) **abstains
and reports doing so** — never silently.

Trial admission (`glm.admit_trials`) requires a **fraction** of the window's breaths to be
valid — `TRIAL_MIN_VALID_FRAC` (0.60) of the cycles overlapping `TRIAL_WINDOW_SEC` (12.0 s,
inside the true 12.49 s minimum trial gap). `TRIAL_MIN_VALID_CYCLES` is `None`; an absolute
count is **not** condition-neutral, because breathing rate differs between sessions (median
RP 3.30 s Eve vs 3.60 s Mor, Mann-Whitney p=2.4e-110) so a fixed "2 valid" demanded 55% of
Evening's breaths and 60% of Morning's. See the `airflow_config.py` block for the balance
table; do not add a count floor back on top of the fraction.

A rejected trial is dropped from the event train, scored NaN, **and its
`[onset, onset+TRIAL_WINDOW_SEC)` span is blanked to NaN in `series`** — under
`pooled_session` only (`rejected_trial_spans`, applied in Pass A before
`build_continuous_series`). Design and data must agree: leaving those samples in `y` with no
regressor puts them in the residual, which is the bias `fit_pooled_session_glm`'s own
docstring warns about. The blanking is gated on `GLM_ESTIMATION` because under `per_trial`
a 12 s blank overlaps the *next* trial's baseline window (6.74 s median lead, 85.5% of
pairs) — harmless for a pooled fit, which has no per-trial baseline, but not for per-trial.
Do not ungate it.

`TRIAL_REQUIRE_PEAK_CYCLE` has never fired: the RA `[τ−σ, τ+σ]` = [4.33, 11.81] covers 62%
of the 12 s window, so it cannot bind. It is inert, not a safeguard.

### Condition balance — recorded here as a hard gate; it is now a REPORT, not a gate

*Superseded framing.* Balance is measured and reported, never used to pass or fail a gate and
never used to pick a threshold. What survives from this section is the measurement rule below:
**weight by SECONDS, not by breaths.** Counting cycles books
a 332.9 s dead span (SH25/eve) and a 3 s breath as 2 equal units; the quantity that
matters is how much signal each condition loses. `classify_cycles`' report carries both
(`gate_counts`/`kept_frac` and `gate_seconds`/`kept_seconds_frac`), and F2 prints both.

Time-weighted (binding), Eve/Mor % of recorded time removed:
`lost_lock` 1.74/0.85 (p<1e-6, **FAIL**), `rate_implausible` 0.17/0.12 (p=0.14),
`shape` 3.29/4.66 (p<1e-6, **FAIL**), `flat` 2.03/0.12 (p<1e-6, **FAIL**),
`extreme` 2.23/0.44 (p<1e-6, **FAIL**), **overall 7.13/5.90 (p<1e-6, FAIL)**.

Cycle-weighted also fails now that `shape` is duration-corrected: overall 3.08/3.94
(p=0.0020). It passed at 3.13/3.19 (p=0.84) only because the duration confound was
mixing two populations — the old `shape` flagged Evening's long cycles, masking that the
underlying shape statistic fires more in Morning.

**Do not retune the gates to make this pass.** The failure is the instrument working. It
means either the gates need redesign or the imbalance is real physiology (Morning
breathing genuinely more peaked); which one is an open question, not a tuning target. The
cohort-sweep and F2 cells at the end of `airflow_qc_show.ipynb` re-run all of this; read
that table before any single-session figure.

Known accepted limitation: the MS18/eve cough at 633.1–638.4 s is **not** caught (kurtosis
3.15 vs threshold 5.0; amplitude 4.7× vs threshold 8.2×). Every kurtosis threshold low
enough to catch it fails condition balance (at 3.0: 4.29% Eve vs 5.20% Mor, p=0.0042), so
it stays uncaught deliberately.

`SUBJECTS_EXCLUDE` remains a hand-maintained list (`LG07/mor` added 2026-08-05 after it came
last in the cohort sweep at 71.7% breaths kept, next-worst 84.4%). It is **not** derived
from the gates and is out of scope of the cycle-level scheme.

## Current confirmatory result (2026-08-05, after the reset) — ⚠ NOT REPRODUCIBLE FROM THIS TREE

Pre-registered: per-subject mean `score_RA` over admitted trials, Evening vs Morning, paired
Wilcoxon, `pooled_session`. N=17 with both sessions, 1405/1463 trials admitted (96.0%).

**RA is null, and trends OPPOSITE to the hypothesis**: 7/17 subjects Eve>Mor, Eve mean
−0.3375 vs Mor −0.1457, two-sided **W=36.0 p=0.0569**, one-sided (H1 Eve>Mor) **p=0.9747**.
Exploratory: RP 11/17, two-sided p=0.109 / one-sided p=0.054; RFR 6/17, two-sided p=0.306.

## Current confirmatory result (2026-08-06, after the audit fixes) — ⚠ NOT REPRODUCIBLE FROM THIS TREE

Both blockers from the 08-06 audit are now fixed: rejected trials are blanked from `y`
(design/data coherence) and admission is a scale-free fraction. Gates: `lost_lock` added,
`shape` measured on a fixed window. Admitted 1424/1463 (97.3%) — `low_valid_fraction` 33,
`no_valid_cycles` 6.

**RA: 6/17 Eve>Mor, Eve −0.2579 vs Mor −0.0303, two-sided W=28.0 p=0.0202**, one-sided
(H1 Eve>Mor) p=0.9913. The effect is now *significant in the direction OPPOSITE the
hypothesis*. Exploratory: RP 9/17 p=0.927; RFR 6/17 p=0.244.

The p≈0.057 reported on 08-05 was an artifact of the incoherent design/data handling — both
coherent repairs converge on the same place (re-admitting the dropped trials gave p=0.0150,
blanking them gives p=0.0202), which is why 0.057 should not be quoted.

**RP's earlier one-sided p=0.054 is gone (now p=0.93) — do not cite it.** It was resting on
the same defect.

Report these numbers as they stand. The gates and the admission fraction are frozen and must
not be retuned against these p-values. `extreme` still fires low-side on 33 of 59 cycles,
i.e. mostly on shallow breaths rather than the "cough / movement" it is documented as
catching — open, and cosmetic relative to the above.

*(end of historical record)*
