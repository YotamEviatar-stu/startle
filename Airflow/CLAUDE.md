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
> `ANAL_PRE_BASELINE_SEC = 60.0`, `GLM_PRE_FIXATION_SEC = 15.0` / `GLM_POST_CODE_SEC = 15.0`,
> `GLM_CONDITION_FIELD = "none"` (one beta per session), `GLM_ESTIMATION = "pooled_session"`.
>
> **5 · Scope — rewritten 2026-08-17.** The **analysed window** is defined once per session at
> `airflow_glm.py:845-855`: from `min(first picture code, first d105 - ANAL_PRE_BASELINE_SEC)`
> to the last `response_range_sec` boundary (D124). **40.7 %** of detected cycles lie inside it
> (6960 of 17097). Every reference statistic and every reported count is now computed on that
> span alone — `classify_cycles(..., window_mask=)` and `_cycle_report(..., window_mask=)` in
> `airflow_qc.py`, `zscore_subject_series(..., window_masks=)` in `airflow_glm.py`;
> `admit_trials` always was. **Do not reintroduce a whole-recording statistic.**
>
> Two deliberate exceptions, both measured — see `METHOD.md` §6.0 and §9.6:
> - **Gates still run on every cycle**, so an out-of-window artifact is blanked out of the
>   filter's warm-up. Only the reference and the report are restricted.
> - **`build_continuous_series` still reads the whole pre-window signal.** RA's causal
>   high-pass is 0.001 Hz (tau = 159 s); cropping its input moves the in-window RA series by
>   0.112 robust-SD median / 0.507 worst, and a 30 s or 60 s pre-roll does not substitute.
>   That span is warm-up, not data, and its own output is already NaN before the window.
>
> **6 · Open, not resolved** (METHOD.md §9.6): ~11.7 s of rows at the start of each solve carry
> no regressor support (`GLM_PRE_FIXATION_SEC = 15` vs the CRF's -10 s support); varying that
> constant moves beta_RA by up to **1.04 between-session SD** in the worst session. And only
> 11/37 sessions have >= 3 tau of filter warm-up. Neither is fixed.
>
> **Authoritative instead of this file:** `airflow_glm.py`, `airflow_qc.py`,
> `airflow_config.py` for what runs; `Airflow/_scratch/METHOD.md` for the complete method
> description, its parameter provenance (§6), known divergences (§9) and claims (§10).
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
the GLM solves on finite rows only. `Airflow/_scratch/METHOD.md` §10 lists these
as checkable claims — including **condition-blindness**: no function in this path may read
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
| `Airflow/_scratch/METHOD.md` | **the method** — every stage, every parameter with its provenance, known divergences, claims |
| `Airflow/_scratch/verify_spec_numbers.py`, `debug_pipeline.py` | the two verifiers; both currently green |
| `Airflow/GLM_METHOD_FOUNDATIONS.md` | PsPM cascade/GLM background; **stale** on rejection |
| `Airflow/DISCUSSION.md` | open questions and reasoning log; **stale** on rejection |
| `Airflow/airflow_glm_input_show.ipynb` | marked BROKEN — do not run |

---

*(The historical record of the superseded per-trial design was removed 2026-08-16;
it is preserved in git history at commit 66dccb9.)*
