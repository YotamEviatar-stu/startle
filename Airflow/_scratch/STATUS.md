# Airflow GLM — status and implementation handoff

Rewritten 2026-08-16. **This file supersedes the 2026-08-12 version, whose §A claimed the
cycle scheme was unimplemented — that was false.** Companion to `cycle_rejection_spec.md`
(the design + this session's revisions) and `verify_spec_numbers.py` (the evidence,
currently **ALL CHECKS PASSED**).

Read this before writing any implementation code.

---

## A · What runs today

The **cycle-based scheme is implemented and live**. `CYCLE_REJECTION_ENABLED = True`.

```
run_glm_scoring                                   airflow_glm.py
  Pass A, per session
    qc_session                    → breath_size A only; NO masking, NO repair
    phase1_filter_downsample      → raw_z @ 10 Hz (robust z, MAD-scaled)
    detect_cycles(raw_z, …, signal_raw, …)        → RP / RA / RFR on signal_raw
    _nk2_peak_trough_times(raw_z) + attach_cycle_features  → n_peaks, std
    airflow_qc.classify_cycles    → one verdict per breath, 5 gates
    airflow_qc.apply_neighbour_rules → recovery, gap_fill
    admit_trials                  → t["admitted"] over response_range_sec
    build_continuous_series(…, rejected=)  → knots dropped, spans NaN'd after lfilter
  zscore_subject_series           → per subject, pooled over their sessions
  Pass B, per session
    trial construction → _run_glm_analysis → fit_pooled_session_glm
```

`_run_glm_analysis` Pass 2 now does exactly one thing:
`reason = None if t["admitted"] else (t["admit_reason"] or "not_admitted")`.
**Nothing re-derives validity from window statistics.**

Verified end to end on MS18 / ML28 / ER23: 229 trials, 31 rejected (13.5 %), every reason
`low_valid_fraction`, one condition and one distinct β per session.

---

## B · What changed on 2026-08-16 (this session)

| # | change | files |
|---|---|---|
| 1 | `GLM_CONDITION_FIELD` `"label"` → `"none"` — one β per session | `airflow_config.py` |
| 2 | `CYCLE_RP_MAX` `10.0` → `7.0` s | `airflow_config.py` |
| 3 | **Dead trial gates deleted** — eleven booleans, shape centroid, `nk2_has_cycles`, `_hampel_flag_cycles`, `score_ceiling`, and 12 config constants | `airflow_glm.py`, `airflow_config.py` |
| 4 | `qc_session` no longer calls the deleted `flag_excursions`; returns `signal_clean = signal_raw`, no masking | `airflow_qc.py` |
| 5 | `airflow_qc_show.ipynb` cell 9 title/shading driven by the cycle report; cell 27's borrowed `QC_MIN_VALID_FRACTION` → local `min_cov = 0.5` | notebook |
| 6 | `verify_spec_numbers.py` repaired; two MS18/eve sum-of-squares numbers genuinely moved (32.1→31.5 %, 33.1→32.4 %) because `signal_clean` is now literally `signal_raw` | `_scratch/` |
| 7 | Spec revised: §0, §2, §3.1–3.3, §4, §6 scope warning, §7.3, new §8 §9 §10 | `cycle_rejection_spec.md` |
| 8 | **E3 fixed** — `_run_glm_analysis` no longer re-blanks the wide `win_start:win_end` span; Pass A's `response_range_sec` blank is the only one | `airflow_glm.py` |
| 9 | **Residue swept** — deleted `robust_cutoff`, `flat_session_report`, `cohort_breath_reference`, `_butter_bandpass_filter`, per-cycle `std`, `n_cycles_in_window`, and `AIRFLOW_HIGHPASS` / `AIRFLOW_AMPLITUDE_Z_THRESHOLD` / `QC_FLAT_SESSION_RATIO` (all zero-caller) | `airflow_glm.py`, `airflow_qc.py`, `airflow_config.py` |
| 10 | `Airflow/CLAUDE.md` banner rewritten — it asserted the opposite of the truth | `Airflow/CLAUDE.md` |
| 11 | `airflow_glm_input_show.ipynb` marked **BROKEN — DO NOT RUN** in a new first cell | notebook |
| 12 | **New:** `_scratch/debug_pipeline.py` — full-cohort run + spec §4 invariant checks. ALL INVARIANTS PASSED | `_scratch/` |
| 13 | **New:** `_scratch/IMPLEMENTATION_BRIEF.md` — cold-start brief and copy-paste prompt for the implementation session | `_scratch/` |

**Nothing is committed.** `airflow_glm.py`, `airflow_qc.py`, `airflow_config.py` and
`airflow_qc_show.ipynb` all carry uncommitted edits, some of them predating this session.

---

## C · Every statistical measure currently in place

### Rejection path

| # | where | statistic | value |
|---|---|---|---|
| 1 | `extreme` gate, `airflow_qc.py:59-70` | robust z on **log(RA)**: centre = session median of RA; scale = **frozen cohort** MAD×1.4826 | `CYCLE_LOGRA_SCALE = 0.2378`, `k = 6.0` → cut at **4.17 × session median** |
| 2 | abstention, `airflow_qc.py:60-64` | reference-adequacy test | `CYCLE_MIN_REF_CYCLES = 30`; below that, or median RA ≤ 0, `cut = NaN` and `"extreme"` is listed in `report["abstained"]` |
| 3 | `rate_implausible` | absolute threshold, no statistic | `CYCLE_RP_MAX = 7.0 s` |
| 4 | `no_inspiration`, `lost_lock` | integer peak counts, no statistic | `n_peaks == 0`; `n_peaks >= 3` |
| 5 | `recovery`, `gap_fill` | deterministic adjacency, no statistic | booleans |
| 6 | `admit_trials` | fraction of overlapping cycles valid | `TRIAL_MIN_VALID_FRAC = 0.60` |

**The scale in #1 is the only session-relative reference in the whole rejection scheme, and
its spread is frozen at cohort level on purpose** — with each session's own MAD the same k
means a different gate in every recording (measured spread 1.59× to 11.59×).

### Signal path

| # | where | statistic |
|---|---|---|
| 7 | `despike_signal_raw`, `airflow_glm.py:195-196` | median + MAD×1.4826, `GLM_DESPIKE_K = 50` |
| 8 | `phase1_filter_downsample`, `:256-258` | **robust** z of `raw_z`: `(x − median) / (MAD×1.4826)` |
| 9 | `session_breath_size`, `airflow_qc.py:28` | `A = 2 × robust_sd(bandpassed trace)` |
| 10 | `attach_cycle_features`, `airflow_glm.py:363` | per-cycle `np.std` — **computed and never read by anything.** Dead field; it was the input to a `flat` gate that no longer exists |

### Scoring path

| # | where | statistic |
|---|---|---|
| 11 | `zscore_subject_series`, `:508` | `np.mean` / `np.std` pooled per subject — **the one non-robust statistic left, and the open decision (§E1)** |
| 12 | basis construction | `spm_orth` serial Gram-Schmidt; peak-normalise to unit range; design mean-centring (PsPM `centering = 1`) |
| 13 | β estimation, `:601`, `:715` | OLS via `np.linalg.pinv` on finite rows only |

### Group level (notebooks only, after every β is fit)

| # | statistic |
|---|---|
| 14 | `scipy.stats.wilcoxon` — paired signed-rank, Eve vs Mor, N = 17 (×3 uses); one `ttest_rel`; `sem` for error bars |
| 15 | Balance reports (probe scripts): Mann-Whitney U over per-session % of breath-seconds removed. **A report, never a gate, never a tuning target.** |

Note the asymmetry: the signal path (#7–9) and the rejection path (#1) are all
median/MAD-based, and only #11 uses mean/SD.

---

## D · Traps for the next implementation agent

1. **`Airflow/CLAUDE.md` is wrong in both directions.** Its banner says the cycle code is
   absent (it is present); its historical section says a fixed `CYCLE_RP_MAX` removes
   *Evening* preferentially (it removes **Morning** — 9.33 % vs 5.52 % of in-window breath
   seconds at 7 s). Do not trust its p-values (W=36.0, W=28.0) either.
2. **Scope.** Only **36.5 %** of cycles (6218/17028) and 6.56 h of 18.50 h lie inside the
   analysis window. Any gate count, balance figure or by-eye example computed over the whole
   recording is ~3× inflated and answers a different question. See spec §6.
3. **`airflow_glm_input_show.ipynb` cannot run** — it calls
   `detect_cycles(..., kurt_win_sec=...)`, an argument that does not exist.
4. **Two blanking spans disagree.** `admit_trials` blanks a rejected trial's
   `response_range_sec` in Pass A, but `_run_glm_analysis` blanks
   `win_start_idx:win_end_idx` (baseline start → response end) again at fit time. Spec §5
   says it should be the narrow span only. Unresolved — see §E3.
5. **Cache invalidation is automatic.** The notebook fingerprint hashes every JSON-able
   config value, so any config change invalidates the Layer-2 scoring cache. The Layer-1
   25 Hz session cache (`airflow_cache.pkl`) is untouched by all of this.
6. **`AIRFLOW_SHAPE_*` must stay.** The BxB path (`airflow_amp_processor.py`) reads it.
   Likewise `airflow_qc.flag_deep_breaths`, still called by `verify_spec_numbers.py`.
7. **Acceptance is by eye on the trace**, per subject, in `airflow_qc_show.ipynb` style —
   never by the Eve-vs-Mor Wilcoxon. Recompute the endpoint **once**, at the very end, after
   the rejection rules are frozen.

---

## E · Open decisions — settle these BEFORE writing code

| # | decision | state |
|---|---|---|
| **E1** | **`zscore_subject_series`.** Options, best first: (a) divide each subject's series by the **median RA over their accepted breaths** — a physiological unit, resistant by construction, makes β read as "in units of a typical breath"; `session_artifacts` already carries `cycles` and `cycle_rejected` at the call site, so it needs one extra argument. (b) keep the structure, swap `np.std` → `airflow_qc.robust_sd` — fixes resistance, keeps the circularity of scaling by a statistic of the scored series. (c) drop the rescaling and normalise at group level, as PsPM does. **Safe to decide on methodology alone:** any per-subject rescale applies the same factor to that subject's Eve and Mor, so it cannot flip a sign. Centring is provably irrelevant — there is an intercept column. | **OPEN — awaiting the user** |
| **E2** | **`n_peaks` / NK2** (spec §9). NK2 `khodadad2018` runs on `raw_z` and vetoes 2 of the 5 gates, violating invariant 3. Either qualify `no_inspiration` by amplitude, or implement the spec's own-signal prominence rule so one detector owns the verdict. `lost_lock` at K=3 fires on 3 cycles cohort-wide; K=2 is where the data points. | **OPEN — parked by the user as "not crucial at the moment"** |
| **E3** | **Which span a rejected trial blanks** (trap 4 above). | **OPEN** |
| **E4** | Per-cycle `std` (#10) is dead — delete it, or wire it to something. | **OPEN, cosmetic** |
| **E5** | BxB path (`peak_excursion_normalized`) still uses the old `AIRFLOW_*` gates. Keep as a comparison path or retire? | **OPEN** |

Settled and not to be reopened without saying so: `CYCLE_LOGRA_SCALE`/`K` (frozen cohort
constants), `TRIAL_MIN_VALID_FRAC = 0.60`, `CYCLE_RP_MAX = 7.0`, no `CYCLE_RP_MIN`, no repair
/ no sample-level gate, `response_range_sec` as the trial epoch, one β per session.

---

## F · Deliverables not yet built

1. **Paged session view** — 60 s rows over `signal_raw`, every cycle boxed and coloured by
   verdict, gate name printed on each rejected cycle. Cell 9 now prints the gate *summary*;
   the paged per-cycle view does not exist.
2. **Per-session walkthrough** — raw → verdicts → knots before/after → `y` with holes →
   event train → design columns → rows entering the solve. Prototyped in
   `cycle_rejection_logic.html`; needs to be a function over any session.
3. **Condition-blind viewer** — hide `eve`/`mor` while judging.
4. Rewrite or mark stale: `README.md`, `GLM_METHOD_FOUNDATIONS.md`, `DISCUSSION.md`,
   `Airflow/CLAUDE.md`. Re-publish the *One Verdict Per Breath* artifact.
5. Run `/cross-pipeline-audit` and `/pspm-respiration-audit` — neither depends on the endpoint.

---

## G · Where everything lives

| file | what |
|---|---|
| `airflow_glm.py`, `airflow_qc.py`, `airflow_config.py` | **authoritative** — this is what runs |
| `_scratch/cycle_rejection_spec.md` | the design + 2026-08-16 revisions; §8 condition field, §9 `n_peaks` defect, §10 z-score analysis |
| `_scratch/verify_spec_numbers.py` | regenerates spec §6 from the cache — ALL CHECKS PASSED |
| `_scratch/probe_peak_gates.py`, `probe_peak_gates2.py` | evidence for E2 (whole-recording scope — re-run in-window) |
| `_scratch/probe_rp_max.py` | `CYCLE_RP_MAX` sweep + balance report |
| `_scratch/probe_zscore.py` | evidence for E1 |
| `_scratch/pipeline_now.html` | source of the *One Verdict Per Breath* artifact — predates 2026-08-16 |
| `_scratch/cycle_pipeline_design.md` | superseded prototype from 10 Aug — do not use |
