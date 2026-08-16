# Cycle-based rejection — technical spec

Written 2026-08-11. **Revised 2026-08-16 — the design is now IMPLEMENTED and live.**
`airflow_qc.classify_cycles`, `apply_neighbour_rules`, `GATE_ORDER`, `airflow_glm.admit_trials`,
`build_continuous_series(..., rejected=)` and the `CYCLE_*` / `TRIAL_*` constants are all on
disk and run under `CYCLE_REJECTION_ENABLED = True`. Nothing is committed yet.
Every line reference below is against the working tree as of `8627561` + the uncommitted
edits already present in `Airflow/airflow_glm.py` / `airflow_config.py`.

### What changed on 2026-08-16 (this revision)

| # | change | where |
|---|---|---|
| 1 | `GLM_CONDITION_FIELD` `"label"` → `"none"` — **one β per session**, no Negative/Neutral split | §8, `airflow_config.py:115` |
| 2 | `CYCLE_RP_MAX` `10.0` → **`7.0`** s, set by the user from the 12–20 brpm clinical range | §3.3, §7.3 |
| 3 | `CYCLE_RP_MIN` confirmed **absent from all code** — §3.3's recommendation stands; §3.1's gate table corrected | §3.1, §3.3 |
| 4 | **Scope correction:** every descriptive number must be *in-window*. Only 36.5 % of cycles (6218/17028) and 6.56 h of 18.50 h lie inside the analysis window | §6 |
| 5 | §7.3's condition direction was **backwards** — Morning loses more, not Evening | §7.3 |
| 6 | `n_peaks` as shipped ≠ as specified: NK2 `khodadad2018` on `raw_z`, not prominent peaks on `signal_raw` | §3.2, §9 |
| 7 | z-scoring (`zscore_subject_series`) analysed — centring is a no-op; scale affects cross-subject ranking only | §10 |
| 8 | **Dead trial gates DELETED** — the eleven booleans, the shape centroid, `nk2_has_cycles`, the Hampel path, and `score_ceiling` are gone, with 12 config constants | §2, §3.3 |

The conceptual case (with figures) is at
`Airflow/_scratch/cycle_rejection_logic.html` → https://claude.ai/code/artifact/e5b160a0-b3c7-4d7b-8430-674a2419d93b
This file is the reviewable version: call chain, function signatures, data contracts,
what gets deleted, and how to verify.

---

## 0 · Status of the surrounding docs

| doc | trust |
|---|---|
| `Airflow/CLAUDE.md` | **Stale in the opposite direction now.** Its banner asserts `classify_cycles` / `admit_trials` / `GATE_ORDER` / the `CYCLE_*` constants are absent — they are present and live as of ~2026-08-15. Its p-values (W=36.0 / W=28.0) and its claim that a fixed `CYCLE_RP_MAX` removes *Evening* preferentially are both still wrong; see §7.3. |
| `Airflow/_scratch/STATUS.md` | §A ("the cycle-based scheme exists only as a design") is **false** as of 2026-08-16. §B/§C/§D are still usable. |
| `Airflow/_scratch/pipeline_now.html` → artifact *One Verdict Per Breath* | Closest to current, but predates this revision: shows `CYCLE_RP_MAX = 10 s`, lists "one β per session" as open, and labels `n_peaks` as measured on raw. |
| `Airflow/_scratch/cycle_pipeline_design.md` | Prototype from 2026-08-10. Predates the rules agreed since: no recovery-breath rule, no gap-fill, blank span is the cycle's own span rather than knot-to-knot, `SUBJECTS_EXCLUDE` not applied. Thresholds in it are not carried forward. |
| `Airflow/README.md`, `GLM_METHOD_FOUNDATIONS.md`, `DISCUSSION.md` | Stale w.r.t. both. |
| `airflow_glm.py`, `airflow_qc.py`, `airflow_config.py` | **Authoritative** — this is what runs. |
| `Airflow/airflow_glm_input_show.ipynb` | Cannot run: it imports `airflow_qc.classify_cycles` and calls `detect_cycles(..., kurt_win_sec=...)`, neither of which exists. |

---

## 1 · The call chain as it exists today

Layer 1 (cache, requires MFF):

```
airflow_glm.process_session                 airflow_glm.py:71
  └ native_envelope                         airflow_glm.py:33
     → cache: signal_raw @ CACHE_SFREQ=25 Hz, trials_meta, env_min/env_max
```

Layer 2 (scoring, cache only):

```
run_glm_scoring(sessions_cache, config)                       airflow_glm.py:704
  ├ airflow_qc.qc_session(signal_raw, sfreq, config)          :757   → signal_clean, masked_spans
  │    └ session_breath_size                                  airflow_qc.py:23
  │    └ flag_excursions   (interpolates ≤ QC_MAX_RUN_SEC)    airflow_qc.py:39
  ├ phase1_filter_downsample(signal_for_pipeline, …)          :775   → raw_z @ 10 Hz
  │    └ despike_signal_raw (skipped when QC_ENABLED)         airflow_glm.py:168
  ├ detect_cycles(raw_z, sfreq, signal_for_pipeline, …)       :787   → [{onset_time, assign_time, RP, RA, RFR}]
  ├ airflow_qc.flag_deep_breaths(cycles, QC_MAX_BREATH_RATIO) :810   → drops cycles, adds spans to missing
  ├ _hampel_flag_cycles(cycles, GLM_ARTIFACT_K)               :817   (only under two of the three methods)
  ├ build_continuous_series(cycles, n, sfreq, hp, lp, missing):844   → series{RP,RA,RFR} + times_full
  ├ zscore_subject_series(session_series, config)             :873
  ├ _nk2_peak_trough_times(raw_z, sfreq, config)              :893
  ├ trial construction (win_start_idx / win_end_idx, …)       :895–985
  └ _run_glm_analysis(trials, raw_z, series, sfreq, config)   :983 → :989
       ├ per-trial gates → t["rejected"], t["rejection_reason"]     :1152–1170
       ├ per_trial:      fit_trial_glm                              airflow_glm.py:545
       └ pooled_session: blank rejected windows, fit_pooled_session_glm  :1187–1197 → :581
```

**Two spans, do not confuse them** (`extras/trial_epochs.py:262`, `:269-270`):

* `baseline_range` = `(d105_sample, code_sample)` — fixation to stimulus.
* `response_range` = `(code_sample, next trial's code_sample)`, D124 for the last trial.
  **This is the trial epoch used by rejection and admission** (§5): `[code_n, code_{n+1})`,
  min 7.23 s, median 16.43 s, max 25.11 s.

What the GLM currently *fits* is the union of the two: `win_start_idx:win_end_idx` =
`baseline_range_sec[0]` → `response_range_sec[1]` (`airflow_glm.py:902-907`), i.e. this trial's
fixation onset through the **next** trial's stimulus — measured over all 1463 trials, **min
13.02 s, median 23.15 s, max 31.65 s**. Neither span is a fixed length; see `PIPELINE_AUDIT.md` O3.

---

## 2 · What rejection *was* — DELETED 2026-08-16

> **Historical.** All of the following is gone from `airflow_glm.py`. `_run_glm_analysis`
> Pass 2 now reads `t["admitted"]` and nothing else:
> `reason = None if t["admitted"] else (t["admit_reason"] or "not_admitted")`.
> Also deleted: the shape-centroid block, the second `_nk2_peak_trough_times` call and
> `nk2_has_cycles`, `_hampel_flag_cycles` and its call sites, `flagged_onset_times`, the
> `hampel_reject_trials`+`pooled_session` guard, `rate_max_in_window`, `max_gap_sec`,
> `baseline_std`/`response_std`, `amp_artifact`, `shape_msd`, and the `score_ceiling` block.
> Removed from `airflow_config.py` (12): `GLM_Z_SCORE_THRESHOLD`, `GLM_MIN_STD_RATIO`,
> `GLM_POST_MIN_STD_RATIO`, `GLM_RATE_ARTIFACT_THRESHOLD`, `GLM_MIN_RATE_THRESHOLD`,
> `GLM_MIN_CYCLES_IN_WINDOW`, `GLM_SCORE_MAX`, `GLM_ARTIFACT_METHOD`, `GLM_ARTIFACT_K`,
> `GLM_MAX_ABS_Z`, `QC_TRIAL_WINDOW_SEC`, `CYCLE_PEAK_METHOD`.
> **Kept deliberately:** `AIRFLOW_SHAPE_*` (read by the BxB path), `airflow_qc.flag_deep_breaths`
> (still called by `verify_spec_numbers.py`), `baseline_mean` (25 notebook references),
> `n_cycles_in_window` (descriptive only).
>
> `score_ceiling` was deleted rather than kept as a disabled guard: rejecting a trial because
> its own β is large is the circularity the whole design exists to remove.
>
> Verified on MS18/ML28/ER23 — 229 trials, 31 rejected (13.5 %), every reason
> `low_valid_fraction`/`no_cycles`, one condition and one distinct β per session.

Eleven trial-level booleans, all computed in `_run_glm_analysis`, all OR-ed into one flag:

| reason | line | driven by |
|---|---|---|
| `no_cycles_found` | 1095 | `GLM_MIN_CYCLES_IN_WINDOW`=2 **or** `nk2_has_cycles` false |
| `noisy_baseline` | 1097 | `GLM_Z_SCORE_THRESHOLD`=3.0 on baseline std |
| `flat_signal` | 1102 | `GLM_MIN_STD_RATIO`=0.1 |
| `flat_response` | 1108 | `GLM_POST_MIN_STD_RATIO`=0.1 |
| `rate_artifact` | 1114 | `GLM_RATE_ARTIFACT_THRESHOLD`=40 bpm |
| `cycle_gap` | 1128 | `GLM_MIN_RATE_THRESHOLD`=5 bpm over `max_gap_sec` |
| `atypical_shape` | 1047 | `AIRFLOW_SHAPE_*` centroid distance on `raw_z` |
| `amplitude_artifact` | 1137 | `_hampel_flag_cycles`, only under `GLM_ARTIFACT_METHOD="hampel_reject_trials"` |
| `low_information` | 1140 | `QC_MIN_KNOTS`=5 |
| `low_coverage` | 1144 | `QC_MIN_VALID_FRACTION`=0.5 finite fraction of `y` |
| `score_ceiling` | 1199 | `GLM_SCORE_MAX`=None (disabled) |

Two cycle-level actions exist but are not a rejection scheme:
`flag_deep_breaths` (`RA > QC_MAX_BREATH_RATIO=5.0 ×` session median) drops the cycle **and**
NaNs its span; `flag_excursions` interpolates sample runs ≤ 1 s.

### The two verified defects

1. **A bad breath inside an admitted trial reaches `y` at full weight.** Verified on the cache:
   MS18/eve cyc 110 (633.10–638.40 s, RA 4.31× median) and cyc 111 (638.40–643.80 s, 2.28×) are
   **100 % finite** in `series["RA"]`, and trial 22 (onset 620.4 s) is *not* rejected, so the
   pooled blank at `:1196` never touches them. Those four breaths in MS18/eve (cyc 18, 19, 110,
   111) are 20.9 s of 637.3 s analysed — 3.28 % of samples — and carry **61.1 %** of the
   session's total sum of squares in `y`.
2. **`flag_excursions` cannot see multi-second artifacts.** Its reference is a 1 s rolling median
   (`airflow_qc.py:47`), which rises with the excursion. Over all of RP06/mor it flags
   **0 runs** at k=5, although the session contains a peak of 0.00327984 at t=1340.40 s =
   **14.93 × A** (A = `session_breath_size` = 0.000219662).

---

## 3 · Target design — function by function

### 3.1 `airflow_qc.py`

**New — `classify_cycles(cycles, config, report=True)`**

```python
def classify_cycles(cycles, config):
    """One verdict per breath. Returns (rejected, reason, report).

    rejected : np.ndarray[bool], len == len(cycles)
    reason   : list[str | None], len == len(cycles); the FIRST gate that fired,
               in GATE_ORDER; None where rejected is False
    report   : dict with per-gate counts AND per-gate seconds:
               {"gate_counts": {g: int}, "gate_seconds": {g: float},
                "kept_frac": float, "kept_seconds_frac": float,
                "abstained": [gate names whose session reference was unusable],
                "n_ref_cycles": int, "median_RA": float, "median_std": float}
    """
```

Gates, evaluated in `GATE_ORDER` so attribution is deterministic:

| gate | rule | reference |
|---|---|---|
| `unmeasurable` | `RA` non-finite or ≤ 0 | absolute |
| `extreme` | `log(RA) − log(median RA)` > `CYCLE_LOGRA_K` × `CYCLE_LOGRA_SCALE`, i.e. `RA` > **4.17 ×** session median | session centre, frozen cohort scale |
| `rate_implausible` | `RP` > `CYCLE_RP_MAX` = **7.0 s** (no lower bound — `CYCLE_RP_MIN` does not exist) | absolute (§7.3) |
| `no_inspiration` | `n_peaks == 0` inside the span | absolute (§7.2, §9) |
| `lost_lock` | `n_peaks >= CYCLE_LOSTLOCK_MIN_PEAKS` = 3 | absolute (§9) |

As implemented in `airflow_qc.py:66-76`, matching this order.

Features are measured on **`signal_raw`**, never on `raw_z` — `raw_z` is bandpassed and z-scored
and exists only to time the onsets (`pspm_resp_pp.m`'s `resp` vs `newresp`).
**Exception, and it is a defect:** `n_peaks` is the one feature that violates this. It is a
count of NeuroKit2 `khodadad2018` peak times computed on `raw_z` (`airflow_glm.py:446-455`,
called at `:852`), not prominent peaks on `signal_raw` as §3.2 specifies. It drives 2 of the
5 gates. See §9.

**New — `apply_neighbour_rules(rejected, reason, config)`**

Two passes, in this order, each returning a new mask so the input is never mutated in place:

1. **recovery** — for every `i` where `reason[i] == "extreme"`, set `rejected[i+1] = True`,
   `reason[i+1] = "recovery"` if it is not already rejected. Applies to the amplitude gate
   only; a `flat` or `rate_implausible` cycle does not drag its neighbour.
2. **gap-fill** — for every interior `i`, if `rejected[i-1] and rejected[i+1]` then
   `rejected[i] = True`, `reason[i] = "gap_fill"`. Single pass over the *pre-fill* mask, so
   the rule cannot cascade down a long run.

**Retire:** `flag_deep_breaths` (:81) — superseded by `extreme`.
`flag_excursions` (:39) keeps its detection half and **loses its interpolation half** (§7.1);
until that is settled, call it with `QC_ENABLED=False` so nothing is repaired.
`robust_sd`, `robust_cutoff`, `session_breath_size`, `_rolling_median` are unchanged and reused.

### 3.2 `airflow_glm.py`

**`detect_cycles` (:265)** — add the features the gates need, without changing detection:

```python
def detect_cycles(raw_z, sfreq, signal_raw, native_sfreq,
                  median_win_sec=1.0, refractory_sec=1.0,
                  env_min=None, env_max=None,
                  peak_prominence=None):          # NEW
```
Each cycle dict gains `n_peaks` (prominent inspiratory peaks inside the span, measured on
`signal_raw`) and `std` (of `signal_raw` over the span). `RP`/`RA`/`RFR`/`onset_time`/
`assign_time` keep their current meaning and units.

**NOT WHAT SHIPPED (2026-08-16).** `detect_cycles` has no `peak_prominence` argument and
`CYCLE_LOSTLOCK_PROMINENCE` was never added. Instead, `attach_cycle_features`
(`airflow_glm.py:356-366`) counts NK2 peak *times* falling in each span, and those times come
from `_nk2_peak_trough_times(raw_z, sfreq, config)` — a second, independently-tuned detector,
run on the filtered/z-scored copy. `std` is attached there too, on `signal_for_pipeline`.
This divergence is unresolved; see §9.

**`build_continuous_series` (:408)** — the one real change. Today the caller passes
`missing_spans`, and every cycle in `cycles` becomes a knot. New contract:

```python
def build_continuous_series(cycles, n_samples, sfreq, hp=0.001, lp=1.0,
                            missing_spans=None, rejected=None):   # NEW
```
* knots are built from `cycles[i]` where `not rejected[i]`;
* the function derives the **bridged intervals** itself: for every maximal run of rejected
  cycles `i…j`, the blanked span is `[assign_time[i-1], assign_time[j+1])` — last surviving
  knot to next surviving knot — because a value is stored at the *next* onset, so the samples
  it controls extend one cycle past its own span;
* NaN is written **after** `lfilter`, exactly as today (:449), so the causal IIR is not poisoned;
* `missing_spans` stays, for the session crop and any externally supplied gaps.

Returning the derived spans (`series, times_full, blanked_spans`) lets the notebook shade
precisely what was removed.

**New — `admit_trials(trials, cycles, rejected, config)`**

```python
def admit_trials(trials, cycles, rejected, config):
    """Sets t["admitted"], t["valid_frac"], t["n_cycles_win"], t["admit_reason"].

    The window is the trial's OWN response epoch, (r0, r1) = meta["response_range_sec"]
    = (code_sample, next trial's code_sample, or D124 for the last trial), defined at
    extras/trial_epochs.py:269-270. No fixed-length box. A cycle overlaps it when
    assign_time > r0 and onset_time < r1. Admitted iff valid_frac >= TRIAL_MIN_VALID_FRAC,
    with no absolute count floor (a count is not condition-neutral: breathing rate
    differs between sessions).
    """
```

**`run_glm_scoring` (:704)** — rewire, delete nothing else:
`detect_cycles` → `classify_cycles` → `apply_neighbour_rules` → `build_continuous_series(...,
rejected=...)` → `admit_trials`. Drop the `flag_deep_breaths` block (:810) and the
`_hampel_flag_cycles` block (:817). `SUBJECTS_EXCLUDE` handling (:725) is untouched.

**`_run_glm_analysis` (:989)** — delete the eleven gates of §2 and read `t["admitted"]`.
Keep: label/condition assignment (:1001–1018), `epoch_anal`/`times_anal`, the pooled blanking
block (:1187–1197, but see §5), and `fit_trial_glm` / `fit_pooled_session_glm` untouched.

**`fit_pooled_session_glm` (:581)** needs **no change**: it already computes
`valid = np.isfinite(y) & np.all(np.isfinite(Xr), axis=1)` (:687) and solves
`np.linalg.pinv(X[valid]) @ y[valid]`. NaN rows are skipped; the design matrix is still built
over `n_samples` on the original grid, so nothing is trimmed and no timestamp moves.

### 3.3 `airflow_config.py`

**`CYCLE_RP_MIN` — measured 2026-08-15, recommendation: do not add it.** PsPM's only
period-plausibility rule is a 1 s floor, and it is applied by *deleting the onset*, not by
rejecting a cycle (`pspm_resp_pp.m` Stage 4: `ibi = diff(respstamp); indx = find(ibi < 1);
respstamp(indx + 1) = [];`). `detect_cycles` already mirrors it exactly via
`GLM_REFRACTORY_SEC = 1.0` (`airflow_glm.py:309-317`). Because that pass only removes onsets,
surviving gaps can only grow, so no cycle can reach `classify_cycles` with RP < 1 s.
Confirmed on the cohort: 17097 cycles / 37 sessions, **minimum RP 1.20 s**, none under 1.0 s.
(Re-counted 2026-08-16 under the current config: 17028 cycles / 37 sessions — the small
difference is exclusions, not detection.) A `CYCLE_RP_MIN` gate would be dead code like
`unmeasurable`. A floor *stricter* than PsPM's (the old design used 2.0 s) is a departure from
the reference and needs its own physiological argument.

**Confirmed 2026-08-16: `CYCLE_RP_MIN` was never added, and should not be.** `grep -rn
"CYCLE_RP_MIN" --include="*.py"` returns nothing. The gate table in §3.1 previously specified
`RP < CYCLE_RP_MIN or RP > CYCLE_RP_MAX`, contradicting this section; it has been corrected to
the upper bound alone, which is what `airflow_qc.py:71-72` implements.

**As shipped (`airflow_config.py:92-107`, verified 2026-08-16):**

```python
CYCLE_REJECTION_ENABLED  = True
CYCLE_LOGRA_SCALE        = 0.2378
CYCLE_LOGRA_K            = 6.0
CYCLE_RP_MAX             = 7.0      # was 10.0 until 2026-08-16
CYCLE_PEAK_METHOD        = "khodadad2018"   # DEAD — never read; see §9
CYCLE_LOSTLOCK_MIN_PEAKS = 3
CYCLE_MIN_REF_CYCLES     = 30
CYCLE_RECOVERY_RULE      = True
CYCLE_GAPFILL_RULE       = True
TRIAL_MIN_VALID_FRAC     = 0.60
```

`CYCLE_LOSTLOCK_PROMINENCE` and `CYCLE_RP_MIN` were never added — correctly, per §3.3 above
and §9 below. `CYCLE_PEAK_METHOD` **is** present but is never read by any function;
`_nk2_peak_trough_times` uses `RSP_CLEAN_METHOD` / `RSP_PEAK_METHOD_CLEANING` instead. Delete it.

**`CYCLE_LOGRA_SCALE = 0.2378`, `CYCLE_LOGRA_K = 6.0` — DECIDED by the user, 2026-08-15.**
The `extreme` gate is a robust z on `log(RA)`, not on `RA`: RA is a positive, strongly
right-skewed variable, and a MAD-based rule applied to it directly puts the cutoff at
1.48–3.20× the median (median 1.82×) — below the cohort's own p95 breath (1.94×), i.e. it
would reject ordinary deep breathing. In log space the rule behaves.

`CYCLE_LOGRA_SCALE` is a **frozen cohort constant**, `median|log RA − median(log RA)| × 1.4826`
pooled over 37 sessions / 6411 breaths on 2026-08-15, **before any Evening-vs-Morning
comparison**. It is never recomputed per session: with each session's own log-scale the same
k spans cutoffs of 1.59× to 11.59× (7.3× spread), so k would mean something different in
every recording. Do not re-derive it when sessions are added or excluded without saying so.

`CYCLE_LOGRA_K = 6.0` puts the cut at `exp(6 × 0.2378)` = **4.17 × the session median**,
removing 48 breaths cohort-wide (0.75 % of 6411). The rule fixes the *shape* of the decision;
k fixes its *severity*, and k = 6 was pinned by the hand-labelled artifacts — it is the
largest integer k that still catches the MS18/eve cough at 4.31× (k = 7 cuts at 5.28× and
misses it), while OM16/mor 7.21× and RP06/mor 7.07× are caught at every k considered.
Calibrated against eye-labels, frozen before any endpoint was recomputed. Not retunable.
No `TRIAL_WINDOW_SEC`: the admission window is the trial's own `response_range_sec`, so
there is no fixed-length constant to set.

Remove once §3.2 lands: `GLM_ARTIFACT_METHOD`, `GLM_ARTIFACT_K`, `GLM_Z_SCORE_THRESHOLD`,
`GLM_MIN_STD_RATIO`, `GLM_POST_MIN_STD_RATIO`, `GLM_RATE_ARTIFACT_THRESHOLD`,
`GLM_MIN_RATE_THRESHOLD`, `GLM_MIN_CYCLES_IN_WINDOW`, `GLM_MAX_ABS_Z`, `QC_MIN_KNOTS`,
`QC_MIN_VALID_FRACTION`, `QC_MAX_BREATH_RATIO`, `QC_TRIAL_WINDOW_SEC`, `AIRFLOW_SHAPE_*`.
Leave the `AIRFLOW_*` names used by the BxB path (`airflow_amp_processor.py`) alone.

---

## 4 · Invariants a reviewer should check

1. **Nothing is trimmed.** `len(series[m])` is unchanged by rejection; only values become NaN.
   Assert `len(y) == n_samples` before and after.
2. **One decision point.** Validity is produced by `classify_cycles` + `apply_neighbour_rules`
   and consumed by exactly two places (`build_continuous_series`, `admit_trials`). No stage
   re-derives it from window statistics.
3. **Measured on `signal_raw`.** No gate reads `raw_z`; `raw_z` is timing only.
   ⚠ **Currently violated** by `n_peaks`, which is derived from NK2 peaks computed on `raw_z`
   and drives `no_inspiration` and `lost_lock` (§9).
4. **Condition-blind.** No function in this path may read the session key, `label`, or
   `has_sound`. Balance Eve-vs-Mor is *reported* after the fact, never used as a gate.
   Holds through §8: `GLM_CONDITION_FIELD = "none"` means the design carries no condition
   split at all, and each session is fit separately, so Eve/Mor exists only in the pairing
   done after every β is estimated.
5. **NaN after filtering.** Blanking happens post-`lfilter`, otherwise the recursion propagates
   the hole forward for the rest of the session.
6. **Abstention is explicit.** With fewer than `CYCLE_MIN_REF_CYCLES` cycles or a zero MAD, a
   session-relative gate must return "abstained" in the report, never a silent pass.

---

## 5 · The trial epoch — settled 2026-08-12

**`response_n = [code_n, code_{n+1})`**, taken from `meta["response_range_sec"]`
(`extras/trial_epochs.py:269-270`; the last trial of a session ends at D124 instead).
The baseline segment `[d105_n, code_n)` plays no part in cycle rejection or trial admission.

This is both what admission counts breaths over and what a rejected trial blanks — one span,
so the earlier "12 s box vs. full D105→next-D105 window" question does not arise. Two
properties make it the right object:

* **It tiles.** Consecutive response epochs are adjacent by construction, so every sample
  between the first and last `code` belongs to exactly one trial — no gaps, no double cover,
  nothing orphaned when a trial is blanked.
* **It is the trial's own signal.** Measured over the 1463 trials: **min 7.23 s, median
  16.43 s, max 25.11 s**. The 35 trials shorter than 12 s are all last-of-session (they end at
  D124); every non-final trial runs ≥ 12.39 s.

Breaths per epoch, all 1463 trials: 0→2, 1→7, 2→34, 3→89, 4→227, 5→458, 6→395, 7→167, 8→52,
9→24, 10→8 — median **5**, mean 5.33; only **43 trials (2.9 %)** carry ≤ 2 breaths.
(The fixed 12 s box gave median 4 and 97 trials at ≤ 2, i.e. a coarser fraction and a
knife-edge at 4 breaths where one bad breath is 25 % of the evidence.)

Blanking stays gated on `GLM_ESTIMATION` — `pooled_session` only. Under `per_trial` there is
nothing to blank: a rejected trial is simply not fit.

---

## 6 · Verification — numbers to reproduce

> **Scope correction, 2026-08-16 — read before quoting any number here or elsewhere.**
> A session recording is roughly 2.7× longer than the stretch we analyse. Cohort-wide,
> **6218 of 17028 cycles (36.5 %)** and **6.56 h of 18.50 h** lie inside the analysis window
> (the union of the trials' `response_range_sec`, e.g. MS18/mor = 187.5–831.8 s of a 1538.5 s
> recording). Gate counts computed over the whole recording are ~3× inflated and their
> Eve-vs-Mor balance is a different quantity. Any descriptive statistic, balance report or
> by-eye example must be restricted to the window. In-window RP percentiles:
> p50 **3.50 s**, p90 4.80, p95 **5.90**, p99 **10.10**, p99.9 18.27.
>
> This does *not* mean out-of-window signal is inert: under `pooled_session` the whole
> session's `series` enters `fit_pooled_session_glm`, so out-of-window samples still shape
> the intercept and the residual. They are irrelevant for *judging* the gates, not for the fit.

All computed from `airflow_output/_cache/airflow_cache.pkl` with the code currently on disk.
An implementation is wrong if these move. **Regenerate them yourself:**

```
.venv/bin/python Airflow/_scratch/verify_spec_numbers.py      # all checks pass as of 2026-08-12
```

That script re-derives every figure below from the cache and prints PASS/FAIL per line. It
reads nothing but the cache and the pipeline modules, writes nothing, and needs no MFF reload.
Cycle ids are the tagging view's (`airflow_qc_show.ipynb` cell 11): 1-based within the analysis
window, after `flag_deep_breaths`.

**MS18 / eve** (analysis window 229.3–866.6 s, 167 cycles, median RA 0.0029192)

| cycle | span | RP | RA / median | knot at |
|---|---|---|---|---|
| 109 | 629.80–633.10 | 3.30 | 0.92× | 633.10 |
| 110 | 633.10–638.40 | 5.30 | **4.31×** | 638.40 |
| 111 | 638.40–643.80 | 5.40 | **2.28×** | 643.80 |
| 112 | 643.80–647.50 | 3.70 | 0.71× | 647.50 |

* `extreme` fires on 110; `recovery` takes 111; bridged blank = **[633.10, 647.50)** = 144
  samples = **32.4 %** of the session's sum of squares. Blanking only the two own spans
  ([633.10, 643.80), 107 samples) would be **31.5 %** — the extra 3.7 s costs 0.9 pp.
  (Was 33.1 % / 32.1 % before 2026-08-16; the sample gate's deletion means `signal_clean`
  is now literally `signal_raw`, so those breaths are no longer partly flattened.)
* Window 618–662 s: **441 rows, 144 NaN, 297 into the solve.**
* Trial 22, response epoch [620.41, 639.14) = 18.73 s, breaths 106–111 → 4/6 = 0.67 → admitted.
  Trial 23, [639.14, 656.22) = 17.08 s, breaths 111–115 → 4/5 = 0.80 → admitted.
  **No event is removed.**
* Cycles 18, 19, 110, 111 = 20.9 s of 637.3 s (3.28 %) = **61.1 %** of `Σy²`.

**RP06 / mor** (analysis window 1036.0–1713.0 s, 255 cycles)

* `session_breath_size` A = 0.000219662; peak 0.00327984 at t = 1340.40 s = 14.93 × A.
* `flag_excursions(k=5, win=1.0 s)` → **0 runs** in the whole session.
* Cycle 1337.40–1341.20 (RP 3.80, RA **7.10×**) is what `flag_deep_breaths` currently drops;
  the next cycle 1341.20–1350.70 (RP **9.50 s**, RA 2.10×) is the recovery and is 100 % finite
  in `y`, worth 21.4 % of `Σy²` on its own.

**MS18 / mor** (129 cycles, median RA 0.0025244)

* Cycles 95–107 = 85.4 s of 671.1 s (12.73 %) = 24.6 % of `Σy²`.
* With an illustrative gate (`RP ≥ 7 s` or `RA ≤ 0.10 ×`), gap-fill flips exactly **101 and
  103** and nothing else.
* Trial 29, response epoch [662.72, 680.91) = 18.19 s, breaths 103–105 → 1/3 = 0.33 < 0.60 →
  rejected: the only place in these examples where the right-hand branch fires.

---

## 7 · Parked problems (one at a time, on request)

1. **The sample gate — measured 2026-08-12, RECOMMENDATION ONLY, awaiting the user's call.**
   Proposed: keep S1, move its reference window 1 s → 10 s, drop the interpolation. Evidence over
   all 38 sessions:

   | reference window | OM16/mor 271.0 s | RP06/mor 1337.4 s | MS18/eve 633.1 s | % of time flagged (median / p90 / max) |
   |---|---|---|---|---|
   | **1 s** (today) | 2.4×A **miss** | 4.9×A **miss** | 2.4×A **miss** | 0.018 / 0.194 / 0.316 |
   | **10 s** | 16.5×A hit | 15.4×A hit | 5.8×A hit | 0.280 / 0.723 / 1.329 |
   | 30 s | 16.2×A hit | 15.2×A hit | 5.7×A hit | 0.272 / 0.689 / 1.422 |
   | 60 s | 16.1×A hit | 15.1×A hit | 5.8×A hit | 0.272 / 0.689 / 1.565 |

   A 1 s median rides up with anything longer than a second, so it misses every artifact we have
   identified by eye. 10 s catches all three; 30 s and 60 s land within 0.01 pp of 10 s and buy
   nothing, so take the shortest window that works.

   **The redundancy claim above was measured wrong and is withdrawn (2026-08-13).** It read `RA`
   off `signal_clean`, i.e. off breaths the sample gate had already flattened, so the gate was
   credited for hiding artifacts from the amplitude gate. Re-measured with `RA` from the untouched
   `signal_raw` against a 30 s local reference:

   | threshold | `RA` from `signal_clean` | `RA` from `signal_raw` |
   |---|---|---|
   | > 5×A | 162 breaths / 29 sessions | 153 breaths / 28 sessions |
   | > 8×A | 36 breaths / 13 sessions | 27 breaths / 11 sessions |
   | > 12×A | 11 breaths / 5 sessions | **3 breaths / 2 sessions** |

   EV15/eve and MS13/mor — the two strongest cases for keeping S1 — leave the list entirely: on
   the recording those breaths exceed 5× the session median and `flag_deep_breaths` takes them
   unaided. Of the 3 survivors, two sit at `RA` 4.36× and 4.97× (just under the cut) and the third
   (YR08/mor 716.3 s, RP 18.0 s) is a lost-lock span, not an amplitude case.

   **DECIDED by the user, 2026-08-15 — no repair.** A flagged run marks its breath invalid; it is
   never interpolated. `RA` is measured on `signal_raw`, so a breath is judged on what was
   recorded. Config: `QC_MAX_RUN_SEC` deleted, `flag_excursions`' interpolation branch removed
   (`airflow_qc.py:68-77`), and `detect_cycles` fed `signal_raw` rather than `signal_for_pipeline`
   (`airflow_glm.py:787`).

   **DECIDED by the user, 2026-08-15 — S1 is deleted.** There is exactly one amplitude decision
   point, and it is at the breath: `RA` vs the session median, measured on `signal_raw`. No
   sample-level gate, so `QC_EXCURSION_K` / `QC_EXCURSION_WIN_SEC` / `QC_MERGE_GAP_SEC` /
   `QC_MAX_RUN_SEC` all go, `flag_excursions` is removed, and `qc_session` keeps only
   `session_breath_size` (`A` is still used elsewhere). `masked_spans` disappears from
   `missing_spans`; a rejected breath's own span is what gets blanked.

   Accepted cost, on the record: 3 breaths in 2 sessions carry a >12×A sample while their own `RA`
   stays under 5× the session median — YR08/mor 458.9 s (RA 4.36×) and ER23/eve 1253.7 s
   (RA 4.97×) are near-misses of the cut, and YR08/mor 716.3 s (RP 18.0 s) is a lost-lock span
   that the period/peak-count gate should take instead.
2. **"Flat" is not low variance.** RP06/mor 1341–1350 s reads flat but its rolling SD is 1.48 ×
   the session median, so a variance gate misses it. It should be caught as *no detected
   inspiration in a 9.5 s span* — hence `no_inspiration` in §3.1 rather than a `std` gate.
3. **Upper period bound — DECIDED by the user, 2026-08-16: `CYCLE_RP_MAX = 7.0 s`.**

   **The direction recorded here and in `Airflow/CLAUDE.md` was backwards.** Median RP is
   3.30 s Evening vs 3.60 s Morning, so Morning breathes *slower*, so a fixed upper bound
   removes **more Morning** breaths, not more Evening. Measured in-window:

   | `CYCLE_RP_MAX` | removed | % of in-window breath time | Eve % | Mor % | Mor/Eve | MWU p |
   |---|---|---|---|---|---|---|
   | off | 0 | 0.00 | 0.00 | 0.00 | — | — |
   | 10 s | 58 cyc / 786 s | 3.36 | 1.84 | 4.72 | 2.57 | 0.0561 |
   | 8 s | 117 cyc / 1316 s | 5.62 | 3.59 | 7.34 | 2.05 | 0.1476 |
   | **7 s** | **178 cyc / 1775 s** | **7.58** | **5.52** | **9.33** | **1.69** | **0.2070** |
   | 6 s | 264 cyc / 2331 s | 9.96 | 7.36 | 12.20 | 1.66 | 0.1576 |
   | 5 s | 493 cyc / 3583 s | 15.31 | 10.36 | 19.56 | 1.89 | 0.1106 |

   Regenerate with `.venv/bin/python Airflow/_scratch/probe_rp_max.py` (that script is
   whole-recording; the table above is the in-window re-run — see §6). Balance is a **report**,
   never a gate and never a tuning target.

   **Justification for 7.0 s.** The clinical adult respiratory rate is 12–20 breaths/min
   (Chourpiliadis & Bhardwaj, *Physiology, Respiratory Rate*, StatPearls), i.e. a single cycle
   of **3–5 s**; Islam et al. 2026 (*Breathing Cycle Detection for Respiratory Tele-health
   Systems*, Arab. J. Sci. Eng. 51:321-339, `papers/s13369-025-11052-6.pdf`, §4.2) states that
   range explicitly and calls 3 s "the minimum expected duration for a healthy person". 7.0 s
   is 1.4× the slowest normal breath and sits at in-window RP **p97** (p95 = 5.90 s,
   p99 = 10.10 s). It was chosen from the physiology, not from an outcome.

   **Note the balance moves the right way.** Tightening 10 s → 7 s removes 3× more signal but
   *halves* the condition asymmetry (2.57× → 1.69×), because the cycles above 10 s are
   disproportionately Morning while the 7–10 s band is more evenly split. The 10 s bound was
   the worse choice on balance exposure, not the better one.

   **Interaction with §9.** The 119 kept cycles with `n_peaks == 2` had RP 3.2–10.0 s
   (median 6.50 s). A 7 s ceiling now takes roughly half of them as `rate_implausible`,
   pre-empting the `lost_lock` question for those spans.

---

## 8 · The GLM condition field — settled 2026-08-16

`GLM_CONDITION_FIELD` (`airflow_config.py:115`) names the trial attribute that defines the
GLM's **conditions**. Each condition gets its own onset vector, which is convolved into its
own regressor (design column), which carries its own β — PsPM's standard structure.

It was `"label"`, so every session's ~86 trials were split by picture valence into Negative
and Neutral: 2 onset vectors, 2 CRF columns, **2 βs per session**, each resting on ~43 trials,
and a trial's `score_ra` was whichever β matched its valence. The endpoint is Evening vs
Morning, not Negative vs Neutral, so that split bought nothing and halved the events behind
each estimate.

**Now `"none"`** — every admitted trial is one condition (`airflow_glm.py:1118-1119` sets
`t["condition"] = 1`). For RA (derivative on) the per-session design matrix is
`[intercept, CRF, dCRF]` — **one β per session**, fit on all admitted trials. `t["condition"]`
has exactly one reader, `fit_pooled_session_glm` (`:710-753`), so the change is self-contained.

**Eve/Mor is not a condition and never enters the design.** Each session is its own fit
(17 subjects × 2 sessions = 34 βs); the contrast is formed afterwards by pairing a subject's
two βs. That separation *is* invariant 4.

The notebook cache fingerprint hashes every JSON-able config value, so this change invalidates
the Layer-2 scoring cache automatically. The Layer-1 25 Hz session cache is untouched.

---

## 9 · `n_peaks` — the open defect (was parked problem 2)

Parked problem 2 ("flat is not low variance") was answered by `no_inspiration`, but the
implementation took a different route from §3.1/§3.2 and the substitution is unresolved.

**What runs:** `_nk2_peak_trough_times(raw_z, sfreq, config)` (`airflow_glm.py:446-455`,
called `:852`) runs NeuroKit2 `rsp_process` with `khodadad2018` over the whole session and
returns peak times; `attach_cycle_features` (`:356-366`) counts how many land in each cycle.
So a **second, independently-tuned detector, running on `raw_z`,** holds veto power over the
first. PsPM has no equivalent — `pspm_resp_pp.m` is zero crossings plus a 1 s refractory,
one detector, one verdict. NK2 measures nothing that reaches the score; it can only delete.

**Measured cohort-wide** (whole-recording scope — re-run in-window before quoting; see §6):
`n_peaks` distribution 0 → 434 (2.55 %), 1 → 16381 (**96.20 %**), 2 → 182 (1.07 %), ≥3 → 31
(0.18 %). Reproduce with `Airflow/_scratch/probe_peak_gates.py` and `probe_peak_gates2.py`.

1. **`lost_lock` at K = 3 is near-inert:** 3 cycles / 24.1 s in 3 of 37 sessions. Of the 31
   cycles at `n_peaks ≥ 3`, 28 are already taken earlier in `GATE_ORDER` by `rate_implausible`
   (21) or `extreme` (7). Meanwhile 119 kept cycles sat at `n_peaks == 2` with median RP
   6.50 s against the 3.40 s single-breath median — two breaths in one span, which is exactly
   what `lost_lock` is documented to catch. K = 2 is where the data points; §7.3's 7 s ceiling
   now absorbs about half of that population anyway.
2. **`no_inspiration` bundles two phenomena.** Its 414 cycles have median RA 0.33× the session
   median breath but span 0.00004× to 4.95×. The bottom is dead signal (SH25/eve, a 332 s span
   at 0.00004×); **26.7 % sit above 0.50× median** and the largest are 1.4–5.0× median at
   ordinary RP (2.6–4.8 s) — real breaths NK2 failed to mark, which we then delete.
3. **It is session-bound, not breath-bound:** 0–47 firings per session (LB17/mor 0, MG14/eve 1;
   LO21/mor 47, AH19/eve 43, YR08/eve 39). Five sessions carry ~45 % of all firings, which
   reads as NK2's detection quality varying by session rather than those sessions breathing worse.

**The decision, unmade:** should a second detector veto the first at all? Either (a) keep NK2
and qualify `no_inspiration` by amplitude so it stops taking large breaths, or (b) implement
§3.2 as written — prominent peaks on `signal_raw`, one detector owning the verdict. Judge by
eye on the trace before choosing. Either way `CYCLE_PEAK_METHOD` is dead and `n_peaks`
currently breaks invariant 3.

---

## 10 · Z-scoring the series (`zscore_subject_series`) — analysed 2026-08-16

`GLM_ZSCORE_METRIC = {"RP": False, "RA": True, "RFR": True}` (`airflow_config.py:132`), so the
primary metric **is** z-scored, per subject, pooled across that subject's sessions
(`airflow_glm.py:514-544`), using `np.mean` / `np.std`.

**PsPM does not do this.** It normalises in three places, all of them the *design*, never the
data: `pspm_resp_pp.m` mean-centres the raw trace before filtering (to find zero crossings
only — RA is then measured on the untouched native-rate `resp`); `pspm_glm.m` peak-normalises
the *basis functions* to unit range; `pspm_glm.m` mean-centres the *convolved design matrix*
(`model.centering = 1`). The comparison is at the same pipeline stage: PsPM's `y` is also the
interpolated, per-modality-filtered RP/RA/RFR series out of `resp_pp`. (`pspm_glm.m` exposes a
`model.norm` z-transform flag believed to default off; not in our reference extracts —
confirm via `/pspm-respiration-audit` before relying on it.) So this is a **project addition**,
justified on between-subject anatomy (ribcage/lung volume), which PsPM handles at group level.

**Measured on the series as it actually arrives** — after filtering, cycle rejection at
`CYCLE_RP_MAX = 7.0`, and NaN blanking (`Airflow/_scratch/probe_zscore.py`): SD / robust-SD is
**1.487 median, 2.540 worst (LB17)**, i.e. rejection does *not* collapse the gap. But a 1.5×
ratio is also what a legitimately heavy-tailed series gives — startle responses are transients —
so this is not evidence of surviving artifact.

**Impact is smaller than it looks, in two parts:**

1. **Centring is a no-op for the score.** `fit_pooled_session_glm` carries an intercept column
   (`beta[0]`; the score is `beta[1 + j]`), so subtracting the mean versus the median moves
   β₀ and never β₁. Half of this item cannot affect any result.
2. **Scale bites only across subjects.** Switching SD → robust-SD multiplies a subject's β by
   their own ratio (1.12–2.54). Because the z-score pools that subject's two sessions, the
   *same* factor applies to Evening and Morning, so the **sign** of each subject's Eve−Mor is
   invariant. Only the cross-subject magnitude ranking changes — which moves a Wilcoxon
   signed-rank but not a sign test.

Open: whether to keep the z-score at all (and move between-subject scaling to the group
comparison), and if kept, whether to use a robust centre/scale.
