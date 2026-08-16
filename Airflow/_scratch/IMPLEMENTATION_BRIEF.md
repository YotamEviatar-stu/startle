# Implementation brief — Airflow cycle-rejection GLM

Written 2026-08-16, at the end of a decisions-and-cleanup session in which **no feature code
was written by request** beyond deleting dead paths. This file exists so the next session can
start cold and still be technically correct.

The launch prompt is a **separate file** — `Airflow/_scratch/PROMPT_implementation.md` — so
this brief contains no instructions a reading session could mistake for a new task. This file
is reference material: read it, don't execute it.

---

## 1 · What state the code is in

The cycle-based scheme is **implemented, live, and clean**. It is not a proposal.

- One verdict per breath in `airflow_qc.classify_cycles`, five gates in `GATE_ORDER`, then
  `apply_neighbour_rules` (`recovery`, `gap_fill`).
- `airflow_glm.admit_trials` turns breath verdicts into `t["admitted"]` over each trial's own
  `response_range_sec`; `_run_glm_analysis` Pass 2 reads that and **nothing else**.
- Every legacy path is gone: the eleven per-trial booleans, the shape centroid,
  `nk2_has_cycles`, `_hampel_flag_cycles`, `score_ceiling`, `flag_excursions`,
  `robust_cutoff`, `flat_session_report`, `cohort_breath_reference`,
  `_butter_bandpass_filter`, and 15 config constants.
- Full-cohort run is clean: **37 sessions, 1426 trials, 129 rejected (9.0 %)**, reasons
  `low_valid_fraction` 107 / `no_cycles` 22, **zero runtime warnings**, all spec §4
  invariants pass.

Two verification commands, both read-only, both currently green:

```
.venv/bin/python Airflow/_scratch/verify_spec_numbers.py   # ALL CHECKS PASSED
.venv/bin/python Airflow/_scratch/debug_pipeline.py        # ALL INVARIANTS PASSED
```

**Run both before you change anything and after every change.** If either goes red, that is
the change's fault until proven otherwise.

---

## 2 · Reading order (do not skip, do not reorder)

1. `Airflow/_scratch/STATUS.md` — status, statistics inventory, open decisions, traps.
2. `Airflow/_scratch/cycle_rejection_spec.md` — the design. §4 is the invariant list, §6
   opens with the scope warning, §8–§10 are this session's new material.
3. `Airflow/CLAUDE.md` — **banner only** (rewritten 2026-08-16). Everything below the banner
   is a historical record of a superseded design; do not read it as a description of the code.
4. `Airflow/airflow_glm.py`, `airflow_qc.py`, `airflow_config.py` — authoritative.

Do **not** trust: `Airflow/README.md`, `GLM_METHOD_FOUNDATIONS.md`, `DISCUSSION.md`,
`GLM_SIGNAL_REVIEW.md`, `AMPLITUDE_NORMALIZATION_REVIEW.md`,
`_scratch/cycle_pipeline_design.md`, or `_scratch/pipeline_now.html`. All predate the current
scheme.

---

## 3 · Hard rules

These are not style preferences. Breaking any of them invalidates the result.

1. **One decision point.** Validity is produced once, per breath, by `classify_cycles` +
   `apply_neighbour_rules`, and consumed by exactly two places
   (`build_continuous_series`, `admit_trials`). Nothing downstream may re-derive validity
   from window statistics, and nothing may reject on the score being scored.
2. **Condition-blind.** No function in this path may read the session key, `label`, or
   `has_sound`. Evening vs Morning exists only as an after-the-fact balance **report**, and
   never as a gate, an objective, or a reason to move a threshold.
3. **Never tune against the endpoint.** The Evening-vs-Morning Wilcoxon is not part of the
   acceptance test. Recompute it **once**, at the very end, after the rejection rules are
   frozen. A threshold chosen because it improved an outcome is not a threshold.
4. **Acceptance is by eye, on the trace**, on a couple of subjects, in the new
   `airflow_method_show.ipynb` (§6) — not on cohort summary tables.
5. **Measure on `signal_raw`.** `raw_z` is bandpassed and z-scored and exists only to time
   the onsets (`pspm_resp_pp.m`'s `resp` / `newresp` split). One known violation: `n_peaks`
   (see §5).
6. **Nothing is trimmed.** Rejection only writes NaN, always **after** `lfilter`, so the
   causal IIR is never poisoned. `len(series[m])` is invariant and no timestamp moves.
7. **Scope every number.** Only **36.5 %** of cycles (6218/17028) and 6.56 h of 18.50 h lie
   inside the analysis window. A whole-recording count is ~3× inflated and answers a
   different question. State the scope of every figure you report.
8. **No comments/docstrings in source unless asked** (project is in an active WIP phase);
   put reasoning in the reply or in `_scratch/`.
9. **Do not write implementation code until the user says to.** Settle each open matter
   conceptually first — measure, present the numbers, get the call.

---

## 4 · Settled — do not reopen without saying so

| decision | value | why |
|---|---|---|
| `CYCLE_LOGRA_SCALE` / `CYCLE_LOGRA_K` | 0.2378 / 6.0 → `extreme` at **4.17×** session median | frozen cohort constants, calibrated before any Eve/Mor comparison |
| `CYCLE_RP_MAX` | **7.0 s** | 1.4× the 5 s slowest normal breath (12–20 brpm, StatPearls via Islam et al. 2026 `papers/s13369-025-11052-6.pdf` §4.2); in-window p97 |
| `CYCLE_RP_MIN` | **does not exist** | PsPM's only floor is the 1 s refractory, already in `detect_cycles`; min observed RP is 1.20 s |
| `TRIAL_MIN_VALID_FRAC` | 0.60 | user's call; no PsPM equivalent, provenance on the record |
| trial epoch | `response_range_sec` = `[code_n, code_{n+1})` | it tiles; one span for both admission and blanking |
| repair | **none** | a flagged run marks its breath invalid; nothing is interpolated |
| sample-level gate | **deleted** | one amplitude decision point, at the breath |
| `GLM_CONDITION_FIELD` | `"none"` | one β per session; Eve/Mor is not a condition |

---

## 5 · Open decisions — settle BEFORE coding

**O1 · `zscore_subject_series`** *(explicitly parked by the user — leave it alone until asked)*.
`GLM_ZSCORE_METRIC` z-scores RA per subject using `np.mean`/`np.std` — the only mean/SD in
the pipeline; everything else is median/MAD. PsPM does not z-score `y` at all (it normalises
the *design*: basis to unit range, convolved columns mean-centred). Measured: SD/robust-SD is
1.487 median, 2.540 worst, after rejection. Options, best first: (a) divide by the subject's
**median RA over accepted breaths** — a physiological unit, resistant by construction, makes
β read as "in units of a typical breath"; `session_artifacts` already carries `cycles` and
`cycle_rejected` at the call site. (b) swap `np.std` → `airflow_qc.robust_sd` — fixes
resistance, keeps the circularity of scaling by a statistic of the scored series. (c) drop it
and normalise at group level, as PsPM does. Safe to decide on methodology alone: any
per-subject rescale hits that subject's Eve and Mor equally, so it cannot flip a sign.
Centring is provably irrelevant — there is an intercept column.

**O2 · `n_peaks` / NeuroKit2** (spec §9). `_nk2_peak_trough_times` runs NK2 `khodadad2018` on
**`raw_z`** and `attach_cycle_features` counts peaks per cycle; that count drives
`no_inspiration` (0 peaks) and `lost_lock` (≥3). So a second, independently-tuned detector
holds veto power over the first, on the wrong signal — the one violation of rule 5, and PsPM
has no equivalent. Measured (whole-recording scope — **re-run in-window**): `n_peaks` is 1 on
96.20 % of cycles; `lost_lock` at K=3 fires on **3 cycles cohort-wide** while 119 kept cycles
sit at `n_peaks == 2` with median RP 6.50 s (two breaths in one span); `no_inspiration`'s 414
cycles span RA 0.00004× to 4.95× the session median, with **26.7 % above 0.50×** — real
breaths NK2 missed, which are then deleted. Either qualify `no_inspiration` by amplitude, or
implement spec §3.2 as written (prominent peaks on `signal_raw`, one detector owning the
verdict). Judge on the trace first. Note `CYCLE_RP_MAX = 7.0` already absorbs about half the
`n_peaks == 2` population as `rate_implausible`.

**O3 · BxB path.** `peak_excursion_normalized` still uses the old `AIRFLOW_*` gates. Keep as
a comparison path or retire?

---

## 6 · What to build (nothing here exists yet)

**One new notebook: `Airflow/airflow_method_show.ipynb`.** Lean and purpose-built for visual
inspection of the method. Not an extension of `airflow_qc_show.ipynb` — reuse its
`compute_session_qc` and nothing else. Exactly two views, both of a subject's sessions side by
side (labelled A/B, see below), both restricted to the **analysed window** (the union of the trials'
`response_range_sec`, ~36.5 % of the recording).

1. **Main view — 3-layer pipeline.** Layer 1 `signal_raw` @ 25 Hz with rejected cycle spans
   shaded; Layer 2 `raw_z` @ 10 Hz with detected onsets; Layer 3 `series[METRIC]` with its
   NaN holes. Trial onsets marked, green admitted / red rejected.
2. **Tagging view — cycle labelling.** Paged 60 s rows over `signal_raw` across the analysed
   window, every cycle boxed and coloured by its verdict, the gate name printed on each
   rejected cycle. Cycle ids 1-based within the analysed window.

Nothing else for now — no statistics panel, no endpoint, no extra cells.

**Session labelling — condition-blind by construction.** The user wants to see both of a
subject's sessions side by side and does not care what they are called. So label the panels
**Session A** and **Session B** (left/right in a fixed order, e.g. sorted session key) and do
**not** print `eve`/`mor` or Evening/Morning anywhere in the figure. Put the mapping behind an
explicit `REVEAL_SESSION_LABELS = False` flag at the top of the notebook, off by default.
This satisfies rule 2 rather than overriding it: the viewer shows two sessions, and the
judgement cannot be steered by which condition is which.

### Deferred (not this session)

- Per-session walkthrough: seven aligned panels raw → verdicts → knots before/after → `y`
  with holes → event train → design columns → rows entering the solve. Static prototype in
  `_scratch/cycle_rejection_logic.html`.
- Descriptive per-session summary: % of time removed, per-gate counts and seconds, sessions
  where a gate abstained. Reported, never pass/fail.
- Rewrite or mark stale: `README.md`, `GLM_METHOD_FOUNDATIONS.md`, `DISCUSSION.md`.
  Re-publish the *One Verdict Per Breath* artifact from `_scratch/pipeline_now.html`
  (shows `RP_MAX = 10 s`, lists one-β-per-session as open — both outdated).
- `airflow_glm_input_show.ipynb` is marked **BROKEN — DO NOT RUN**; port what you need into
  the new notebook and delete it.

---

## 7 · The prompt

The launch prompt lives in its own file so this brief contains no instructions a reading
session could mistake for a new task:

    Airflow/_scratch/PROMPT_implementation.md

Start the session from the repo root `/Users/yotameviatar/startle-1` (memory is keyed to the
exact cwd), then paste that file's contents.

---

## 8 · If the next session wants to verify this brief rather than trust it

```bash
# the scheme is live, not proposed
grep -n "def classify_cycles\|def apply_neighbour_rules" Airflow/airflow_qc.py
grep -n "def admit_trials" Airflow/airflow_glm.py

# no legacy gate survives
grep -rn "no_cycles_found\|atypical_shape\|amplitude_artifact\|score_ceiling\|nk2_has_cycles" Airflow/*.py   # expect: nothing

# one beta per session
grep -n "GLM_CONDITION_FIELD" Airflow/airflow_config.py     # expect: "none"

# both verifiers green
.venv/bin/python Airflow/_scratch/verify_spec_numbers.py
.venv/bin/python Airflow/_scratch/debug_pipeline.py
```
