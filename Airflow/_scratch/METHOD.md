# Airflow respiration GLM — method description

Describes what the code in `Airflow/airflow_glm.py`, `airflow_qc.py` and `airflow_config.py`
does, as of 2026-08-17. Written for review.

This document is **descriptive**. It states what each stage computes, the value of every
parameter, and where each value came from. It does not argue that any choice is correct, and
it does not mark anything as settled. Provenance is labelled in three categories:

- **PsPM** — the value or procedure is taken from `pspm_resp_pp.m` / `pspm_glm.m` /
  `pspm_bf_r*rf_e.m` (Bach et al. 2016, *J Neurosci Methods* 270:147-155).
- **Literature** — taken from a cited source other than PsPM.
- **Investigator** — chosen by the investigator for this project. No external reference.
  Where a rationale was recorded, it is quoted; where none was, that is stated.

Every quantitative claim below is reproducible with:

```
.venv/bin/python Airflow/_scratch/verify_spec_numbers.py
.venv/bin/python Airflow/_scratch/debug_pipeline.py
```

---

## 1 · Data

40-minute EEG-lab recordings, one Airflow (nasal flow) channel, 24 subjects × 2 sessions
(morning and evening), 45 sessions in the cache, 37 after exclusions. Each session contains
~40 picture-viewing trials; a subset carry an acoustic startle probe.

Trial structure comes from DIN triggers (`extras/trial_epochs.py`): `D105` fixation onset,
`D{code}` picture onset, `D110` startle probe, `D124` session end. Two spans per trial:

- `baseline_range_sec` = `[d105, code)` — fixation to picture onset.
- `response_range_sec` = `[code_n, code_{n+1})` — picture onset to the next trial's picture
  onset; the last trial ends at `D124`. Over 1463 trials: min 7.23 s, median 16.43 s,
  max 25.11 s. These epochs **tile**: consecutive epochs are adjacent by construction.

The union of a session's `response_range_sec` is the **analysed window**. It is roughly
37 % of the recording: cohort-wide 6218 of 17028 detected cycles and 6.56 h of 18.50 h lie
inside it.

Two hand-maintained exclusion lists, both applied before anything else:

- `SUBJECTS_EXCLUDE` = `["DA01", "ES29/eve", "EV15/eve", "LG07/mor", "MG14/mor", "MS13/eve",
  "NB03", "YL26"]`. **Investigator.** The config file's own comment reads: *"Unresolved
  placeholder, not an established rejection rule — chosen by eyeballing one outlier scan and
  applied inconsistently across similarly-contaminated sessions."*
- `MANUAL_BAD_SPANS` = `{"EV15/mor": [(253.001, 621.416)]}`. **Investigator**, hand-entered.

---

## 2 · Stage 0 — cache (`process_session`)

Requires the raw MFF. Runs once; everything downstream reads the cache.

| step | value | provenance |
|---|---|---|
| channel pick | `AIRFLOW_CHANNEL = "Airflow"` | data |
| anti-alias lowpass | 2nd-order Butterworth, `AIRFLOW_LOWPASS = 70.0` Hz, `filtfilt`, at native rate | Investigator (anti-alias for the resample below) |
| resample | `CACHE_SFREQ = 25.0` Hz | Investigator. Project constraint: full-rate traces are not cached, to bound load time |
| highpass | **none** | PsPM — `pspm_resp_pp.m` measures amplitude on the unfiltered `resp`, so `signal_raw` deliberately retains DC |

The cached array is `signal_raw`. It is the only amplitude reference used downstream.

> **The cache in use predates the current code.** It was written 2026-07-13 and its session
> dicts contain only `channel`, `sfreq`, `signal_raw`, `times_raw`, `trials_meta`.
> `process_session` as written also emits `env_min`, `env_max`, `d101_time` and
> `native_sfreq`; none are present. See §9.1 — this has a direct consequence for how RA is
> measured.

---

## 3 · Stage 1 — detection copy (`phase1_filter_downsample`)

Produces `raw_z`, used **only** to time breath onsets. No amplitude is ever read from it.
This is `pspm_resp_pp.m`'s split between `newresp` (filtered, for timestamps) and `resp`
(untouched, for amplitude).

| step | value | provenance |
|---|---|---|
| despike | `GLM_DESPIKE_ENABLED = False` → **does not run** | Investigator |
| mean-centre | subtract `nanmean` | PsPM |
| lowpass | 1st-order Butterworth, 0.6 Hz, `filtfilt` | PsPM (`filt.lpfreq=0.6`, `lporder=1`, `direction='bi'`) |
| highpass | 1st-order Butterworth, 0.01 Hz, `filtfilt` | PsPM (`filt.hpfreq=0.01`, `hporder=1`) |
| resample | `GLM_TARGET_SFREQ = 10.0` Hz, after filtering | PsPM (`filt.down=10`) |
| scale | robust z: `(x − median) / (MAD × 1.4826)` | Investigator. PsPM does not z-score here |

`GLM_CYCLE_BANDPASS = (0.01, 0.6)` holds the two cutoffs. The 0.6 Hz lowpass also serves as
the anti-alias filter for the 10 Hz resample (0.6 Hz is 8.3× below the 5 Hz Nyquist).

---

## 4 · Stage 2 — cycle detection (`detect_cycles`)

| step | value | provenance |
|---|---|---|
| median filter of `raw_z` | `GLM_MEDIAN_WIN_SEC = 1.0` s, odd kernel | **Investigator.** `pspm_resp_pp.m` Stage 1 has no median filter |
| onset rule | negative-going zero crossings of the median-filtered `raw_z` (`+ → −`) | PsPM |
| refractory | `GLM_REFRACTORY_SEC = 1.0` s. Single pass over the **original** candidate list: a candidate is dropped if its gap to the *preceding candidate* — not the last kept one — is under 1 s | PsPM (`ibi = diff(respstamp); respstamp(indx+1) = []`) |

A **cycle** is `[onset_i, onset_{i+1})`. Three features per cycle:

| feature | definition | measured on |
|---|---|---|
| `RP` | `onset_{i+1} − onset_i`, seconds | the 10 Hz onset clock |
| `RA` | `max(seg) − min(seg)` | `signal_raw` at 25 Hz, over `ceil(onset·sr) … ceil(next·sr)+1` |
| `RFR` | `RA / RP` | — |

Each value is anchored to `assign_time` = the cycle's **closing** onset, not its opening one.
**PsPM** (`interp1(respstamp(2:end), respdata, …)`).

`detect_cycles` accepts optional `env_min`/`env_max` arrays — per-cache-sample extremes of
the native-rate trace, so that `RA` recovers the true peak height rather than the 25 Hz
sampled one. **They are not supplied**, because the cache does not contain them (§9.1).

---

## 5 · Stage 3 — peak count (`attach_cycle_features`)

`_nk2_peak_trough_times` runs NeuroKit2 `rsp_process` with
`RSP_CLEAN_METHOD = RSP_PEAK_METHOD_CLEANING = "khodadad2018"` over the whole session, on
**`raw_z`**, and returns inspiratory peak times. `attach_cycle_features` counts how many fall
inside each cycle's `[onset, assign)` span and stores it as `n_peaks`.

**Investigator.** PsPM has no second detector; it uses zero crossings and the 1 s refractory
alone. Measured distribution over 17028 cycles (whole-recording scope): `n_peaks` = 0 on
2.55 %, **1 on 96.20 %**, 2 on 1.07 %, ≥3 on 0.18 %.

This is the only feature not measured on `signal_raw`. It drives two of the five gates below.

---

## 6 · Stage 4 — one verdict per breath

### 6.0 The analysed window

Added 2026-08-17. Every reference statistic and every reported count in this pipeline is
computed over one span per session, defined in `airflow_glm.py:845-855`:

```
anal_hi = max(response_range_sec[1])                      # last trigger boundary (D124)
anal_lo = min( min(response_range_sec[0]),                # first picture code
               first d105_time − ANAL_PRE_BASELINE_SEC )  # 60 s of pre-task rest
```

`response_range_sec` comes from the DIN triggers alone (`extras/trial_epochs.py`:
`[code_n, code_{n+1})`, last trial closed by D124) — no signal property enters it. A cycle is
in-window if `onset_time < anal_hi and assign_time > anal_lo`; a sample is in-window if
`anal_lo ≤ t < anal_hi`. **6960 of 17097 cycles (40.7 %)** qualify.

Carried as `window_mask=` into `classify_cycles` and `_cycle_report` (§6.1, §6.5) and as
`window_masks=` into `zscore_subject_series` (§8.1). `admit_trials` was already window-only.
Rationale: a subject is judged only on its scored window, so no pre-task or post-task signal
may set a threshold, a scale, or a reported rate. **Investigator.**

Two things this deliberately does *not* do:

- **Gates still run on every cycle**, in-window or not. An out-of-window artifact is still
  rejected so it is blanked out of the causal filter's warm-up span. Only the *reference*
  and the *report* are restricted.
- **`build_continuous_series` still reads the whole pre-window signal.** The RA sensitivity
  filter is causal with a 0.001 Hz high-pass (τ = 159 s), so its output inside the window
  depends on the preceding minutes; that span is warm-up, and its own output is already NaN
  before the window. Measured: cropping the filter's input instead moves the in-window RA
  series by 0.112 robust-SD median / 0.507 worst-session, and a fixed 30 s or 60 s pre-roll
  does not substitute (0.098 / 0.087 median). `lfilter_zi` initialisation (0.135) and
  median-centring the knots (0.115) were tested and both failed. See §9.6.

`ANAL_PRE_BASELINE_SEC = 60.0` is **Investigator**. D101, the true task-block start, is not
in the Layer 1 cache (§9.1), so the boundary is a fixed offset before the first fixation,
applied identically to every session; all 37 have > 139 s of pre-task recording, so none is
truncated. It adds 11–26 breaths per session (median 19, +11.3 %), whose median RA is 1.03×
the trial-window median (range 0.57–1.51). Effect of adding it: the combined reference median
moved +0.1 % (range −6.0 % to +5.9 %) and one trial admission changed.

### 6.1 `classify_cycles` — five gates, first match wins

Evaluated in `GATE_ORDER`, so attribution is deterministic. Each cycle gets one verdict and
one reason, or is kept.

| # | gate | rule | value | provenance |
|---|---|---|---|---|
| 1 | `unmeasurable` | `RA` non-finite or ≤ 0 | — | Investigator. Has never fired; `max−min` over a finite non-empty segment cannot be ≤ 0 |
| 2 | `extreme` | `RA > median_RA × exp(k·s)` | `CYCLE_LOGRA_SCALE = 0.2378`, `CYCLE_LOGRA_K = 6.0` → **4.17 × the session median** | Investigator (see 6.2) |
| 3 | `rate_implausible` | `RP > CYCLE_RP_MAX` | **7.0 s** | Investigator, referenced to literature (see 6.3) |
| 4 | `no_inspiration` | `n_peaks == 0` | — | Investigator. No PsPM equivalent |
| 5 | `lost_lock` | `n_peaks ≥ CYCLE_LOSTLOCK_MIN_PEAKS` | **3** | Investigator. No PsPM equivalent, no recorded rationale for the value 3 |

There is **no lower bound on `RP`**. `CYCLE_RP_MIN` does not exist in any `.py` file. The only
period floor is the 1 s refractory in `detect_cycles` (§4); the minimum `RP` observed over the
cohort is 1.20 s.

**Abstention.** Gate 2 is the only session-relative gate. Its reference — `median_RA` and the
usable-cycle count `n_ref` — is taken over **in-window cycles only** (§6.0). If a session has
fewer than `CYCLE_MIN_REF_CYCLES = 30` usable in-window cycles, or a non-positive median
`RA`, the cut is set to `NaN`, the gate does not fire, and `"extreme"` is appended to
`report["abstained"]`. No session abstains at the current window (smallest in-window count
is 120 cycles).
`CYCLE_MIN_REF_CYCLES = 30` is **Investigator**, no recorded rationale.

### 6.2 Provenance of the `extreme` gate

The rule is a robust z on `log(RA)`, not on `RA`. Recorded reason: `RA` is positive and
right-skewed, and a MAD rule applied to it directly put the cutoff at 1.48–3.20× the median
(median 1.82×), below the cohort's own p95 breath (1.94×).

- `CYCLE_LOGRA_SCALE = 0.2378` is `median|log RA − median(log RA)| × 1.4826`, pooled over
  37 sessions / 6411 breaths, computed 2026-08-15. It is **frozen at cohort level and never
  recomputed per session**: recorded reason is that with each session's own log-scale, the
  same `k` spans cutoffs from 1.59× to 11.59×. Recomputing the same statistic over the
  current analysed window (§6.0, 6960 breaths) gives **0.2439**. The constant is deliberately
  **not** updated to match — a frozen threshold that is refitted whenever the window
  definition moves is not frozen. See §9.6.
- `CYCLE_LOGRA_K = 6.0` gives `exp(6 × 0.2378)` = 4.17× the session median, removing 53
  in-window breaths cohort-wide (0.76 % of 6960). Recorded reason: it is the largest integer `k` that
  still catches a hand-labelled cough at 4.31× (k = 7 cuts at 5.28× and misses it), while two
  other hand-labelled artifacts at 7.21× and 7.07× are caught at every `k` considered.

**Both were calibrated against hand-labelled artifacts, i.e. against the investigator's own
visual judgement, not against an independent standard.**

### 6.3 Provenance of `CYCLE_RP_MAX = 7.0 s`

Set 2026-08-16. **Literature-referenced, investigator-chosen.** The clinical adult
respiratory rate is 12–20 breaths/min (Chourpiliadis & Bhardwaj, *Physiology, Respiratory
Rate*, StatPearls), i.e. a cycle of 3–5 s; Islam et al. 2026 (*Breathing Cycle Detection for
Respiratory Tele-health Systems*, Arab. J. Sci. Eng. 51:321-339, §4.2,
`papers/s13369-025-11052-6.pdf`) states that range and calls 3 s the minimum expected
duration for a healthy person. 7.0 s is 1.4× the slowest breath that source calls normal.

Neither source specifies an exclusion bound; the step from "5 s is the slow end of normal" to
"7 s is where a breath stops being a breath" is the investigator's. In-window `RP`
percentiles: p50 3.50 s, p90 4.80, p95 5.90, p99 10.10, p99.9 18.27 — so 7.0 s sits at
about p97.

The previous value was 10.0 s. What the gate removes, in-window, by value:

| `CYCLE_RP_MAX` | cycles | % of in-window breath seconds |
|---|---|---|
| 10 s | 58 | 3.36 |
| 8 s | 117 | 5.62 |
| **7 s** | **178** | **7.58** |
| 6 s | 264 | 9.96 |
| 5 s | 493 | 15.31 |

### 6.4 `apply_neighbour_rules` — two deterministic passes

Applied after `classify_cycles`, each returning a new mask.

1. **`recovery`** — if cycle *i* was rejected as `extreme`, cycle *i+1* is also rejected,
   reason `"recovery"`. Applies to the amplitude gate only. `CYCLE_RECOVERY_RULE = True`.
2. **`gap_fill`** — for interior *i*, if *i−1* and *i+1* are both rejected in the
   **pre-fill** mask, *i* is rejected, reason `"gap_fill"`. Single pass, so it cannot cascade.
   `CYCLE_GAPFILL_RULE = True`.

Both are **Investigator**, no PsPM equivalent. No rationale is recorded for `recovery`'s
restriction to `extreme`, nor for `gap_fill`'s single-pass depth.

### 6.5 Reporting

`classify_cycles` returns per-gate **counts and seconds**, `kept_frac`, `kept_seconds_frac`,
the abstention list, `n_ref_cycles`, `median_RA` and the computed cut. All counts and
fractions are over **in-window cycles only** (§6.0); `_cycle_report` filters by `window_mask`
before tallying, so `n_cycles` is the in-window count, not the recording's. Before
2026-08-17 these were whole-recording tallies, which made a session read e.g. "182/182
breaths kept" beside a gate list of 19 rejections that all lay outside its window.

---

## 7 · Stage 5 — continuous series (`build_continuous_series`)

Converts discrete per-breath values into the regression target.

1. Rejected cycles' knots are **dropped**. For each maximal run of rejected cycles *i…j*, the
   bridged span `[assign_time[i−1], assign_time[j+1])` is added to the blank list —
   last surviving knot to next surviving knot, because a value is stored at the *closing*
   onset (§4) and therefore governs samples one cycle past its own span.
2. Surviving knots `(assign_time, value)` are linearly interpolated onto the full 10 Hz grid
   (`np.interp`), edges held constant.
3. A 1st-order Butterworth **bandpass** is applied with `lfilter` — causal, not `filtfilt`.
   `GLM_FINAL_LP = 1.0` Hz for all metrics; `GLM_FINAL_HP = {"RP": 0.01, "RA": 0.001,
   "RFR": 0.001}`. **PsPM** per-modality sensitivity filter.
4. `NaN` is written over the blank list **after** filtering, so the causal recursion is not
   poisoned. **PsPM** in intent (`model.missing`), though PsPM passes missing epochs into the
   design rather than writing NaN into `y`.

The array length is unchanged by rejection; no timestamp moves. Interpolating and then
blanking (rather than blanking first) is **Investigator**.

---

## 8 · Stage 6 — per-subject scaling, admission, GLM

### 8.1 `zscore_subject_series`

`GLM_ZSCORE_METRIC = {"RP": False, "RA": True, "RFR": True}`. For each enabled metric, the
finite **in-window** samples (§6.0, `window_masks=`) of **all of one subject's sessions** are
pooled and the whole series — warm-up included — is transformed `(x − mean) / std`.
**Investigator.** PsPM does not rescale `y`; it normalises the design
(basis to unit range, convolved columns mean-centred). Recorded reason: removes
between-subject rib-cage/lung-volume differences. Pooling across the subject's sessions
rather than within each is deliberate, so the between-session contrast is not normalised away.

This is the only `mean`/`std` statistic in the pipeline; every other centre/scale is
median/MAD. Measured over the series as it reaches this function (after rejection and
blanking), `std / robust_sd` is 1.487 median and 2.540 at worst.

### 8.2 `admit_trials`

Window = the trial's own `response_range_sec`. A cycle overlaps if
`onset_time < w1 and assign_time > w0`. Admitted iff the fraction of overlapping cycles that
are valid is ≥ `TRIAL_MIN_VALID_FRAC = 0.60`. Reasons: `no_cycles` (no overlapping cycle),
`low_valid_fraction`.

**Investigator**, no PsPM equivalent and no external convention behind 0.60. A *fraction* is
used rather than an absolute count because breathing rate differs between sessions (median
`RP` 3.30 s evening vs 3.60 s morning), so a fixed count would demand a different proportion
of each session's breaths.

Under `GLM_ESTIMATION = "pooled_session"`, a rejected trial's `response_range_sec` is added to
the blank list in Pass A — before the series is built — and its event is excluded from the
design. The two are the same span.

### 8.3 `fit_pooled_session_glm`

One design matrix per session per metric:

```
y(t) = β₀·1 + Σ_c [ β_c·(δ_c * CRF)(t) + γ_c·(δ_c * dCRF)(t) ]
```

| element | value | provenance |
|---|---|---|
| conditions | `GLM_CONDITION_FIELD = "none"` → every admitted trial is one condition, so there is **one β per session** | Investigator |
| basis | Gaussian `exp(−(t−τ)²/2σ²)` over −10 … +30 s | PsPM (`pspm_bf_r*rf_e.m`) |
| τ, σ | RP (4.20, 1.65); RA (8.07, 3.74); RFR (6.00, 3.23) | PsPM |
| derivative | appended for RA and RFR, not RP | PsPM (`bf_type` defaults) |
| orthogonalisation | serial Gram-Schmidt (`spm_orth`), then peak-normalise to unit range | PsPM |
| centring | convolved columns mean-centred over valid samples before the intercept is prepended | PsPM (`model.centering = 1`) |
| latency | fixed — basis convolved at the literal onset | PsPM (`model.latency = 'fixed'`) |
| solve | `β = pinv(X[valid]) @ y[valid]`, `valid` = finite `y` **and** all-finite design row | Investigator (PsPM handles missing data through `model.missing`) |

The score is the CRF column's β. `GLM_PRIMARY_METRIC = "RA"`; RP and RFR are computed and
carried but are not the reference metric.

Evening vs Morning is **not** a condition and does not enter any design matrix. Each session
is fit separately (17 subjects with both sessions → 34 βs); the contrast is formed afterwards
by pairing each subject's two βs and applying a paired Wilcoxon signed-rank test.

---

## 9 · Known divergences and open items

Stated as facts, without recommendation.

### 9.1 The native-rate envelope is not in the cache

`process_session` computes `env_min`/`env_max` — per-cache-sample extremes of the native-rate
trace — and `detect_cycles` will use them for `RA` when supplied. The cache in use
(written 2026-07-13) contains **none of the 45 sessions with `env_min`**; its session dicts
hold only `channel`, `sfreq`, `signal_raw`, `times_raw`, `trials_meta`.

Consequence: `RA` is currently `max − min` over the **25 Hz** `signal_raw`. The code's own
documented reason for building the envelope is that at 25 Hz "a sharp inspiratory-flow apex
spans ~3 samples and max-min underestimates it." So the amplitude measurement underlying the
primary metric is running in the mode the code describes as inadequate, and the correction
built for it is inert. Regenerating the cache requires a Layer-1 rebuild from the MFF files.

The size of the resulting bias has not been measured. `d101_time` and `native_sfreq` are
likewise absent from the cache.

### 9.2 `n_peaks` is measured on the filtered copy

Two of the five gates (`no_inspiration`, `lost_lock`) read `n_peaks`, which comes from
NeuroKit2 running on `raw_z`. Every other feature is measured on `signal_raw`. An earlier design specified prominent inspiratory peaks measured on `signal_raw` via a
`peak_prominence` parameter on `detect_cycles`; that parameter was never added.

Measured consequences (whole-recording scope):

- `lost_lock` at K = 3 rejects **3 cycles cohort-wide** (24.1 s, 3 of 37 sessions). Of the 31
  cycles with `n_peaks ≥ 3`, 28 are already taken earlier in `GATE_ORDER`.
- 119 kept cycles have `n_peaks == 2`, median `RP` 6.50 s against a 3.40 s single-breath
  median.
- `no_inspiration` rejects 414 cycles whose `RA` spans 0.00004× to 4.95× the session median;
  26.7 % are above 0.50× median, and the largest are 1.4–5.0× median at ordinary `RP`
  (2.6–4.8 s).
- It fires 0–47 times per session (0 in LB17/mor, 47 in LO21/mor); five sessions account for
  about 45 % of all firings.

### 9.3 Parameters with no external reference and no recorded rationale

`CYCLE_LOSTLOCK_MIN_PEAKS = 3`, `CYCLE_MIN_REF_CYCLES = 30`, `TRIAL_MIN_VALID_FRAC = 0.60`,
`GLM_MEDIAN_WIN_SEC = 1.0`, and the `recovery`/`gap_fill` rules' scope.

### 9.4 Calibration against the investigator's own labels

`CYCLE_LOGRA_K = 6.0` was pinned by hand-labelled artifacts (§6.2), and the method's stated
acceptance criterion is visual inspection of what it rejects. There is no independent
ground truth for breath validity in this dataset.

### 9.5 Scope

Roughly 59 % of each recording lies outside the analysed window (§6.0). As of 2026-08-17 no
reference statistic or reported count is computed over that span; before then, all of them
were. What still reads out-of-window signal is (a) the causal sensitivity filter's warm-up,
deliberately (§6.0), and (b) NK2's peak thresholds (§9.2), not yet quantified.

Under `pooled_session` the solve still includes the samples between
`first d105 − GLM_PRE_FIXATION_SEC (15 s)` and the first picture code, where no regressor has
support — see §9.6.

### 9.6 Boundary constants that are not derived from the model

Three spans coexist and are set independently; none of the three is currently derived from
the design.

| constant | value | what it bounds |
|---|---|---|
| `ANAL_PRE_BASELINE_SEC` | 60 s | how far back the *reference statistics* look (§6.0) |
| `GLM_PRE_FIXATION_SEC` / `GLM_POST_CODE_SEC` | 15 s / 15 s | where the series holds valid *data rows* for the solve |
| CRF kernel support | −10 s … +30 s | where the *design* has any support at all |

Two consequences, both measured 2026-08-17 and neither resolved:

1. **Dead rows at the start of the solve.** The first d105 precedes the first picture code by
   5.2–8.2 s (median 6.7 s), so with `GLM_PRE_FIXATION_SEC = 15` valid data begins 21.7 s
   before the first onset while the first regressor's support begins at −10 s — and the
   Gaussian is already negligible there (τ − 3σ = −3.1 s). About 11.7 s of rows carry no
   regressor at all, ~18.6 s carry none of consequence. Varying `GLM_PRE_FIXATION_SEC`
   against its current 15 s moves β_RA by, in units of its between-session SD (5.07e-01):
   0 s → 0.047 median / **1.04 worst**; 5 s → 0.034 / 0.48; 10 s → 0.016 / 0.07;
   30 s → 0.060 / 0.48; 60 s → 0.105 / 0.67. Deriving the span from the kernel instead
   (`win_start = min(code_time) − 10 s`) would remove the constant and the dead rows; not done.
2. **Unequal filter warm-up.** Available pre-window recording is 139 s / 326 s / 1091 s
   (min / median / max) against the RA high-pass τ = 159 s — ≥ 1 τ in 31/37 sessions, ≥ 2 τ in
   22/37, ≥ 3 τ in only **11/37**. The RA sensitivity filter is therefore not fully settled at
   window start in most sessions, by differing amounts per subject.

A third, smaller item: `CYCLE_LOGRA_SCALE` was frozen at 0.2378 from the pre-2026-08-17
window; the same statistic over the current window is 0.2439 (§6.2). It is left frozen
deliberately.

---

## 10 · Claims the method makes

Each is falsifiable from the code and the recordings.

1. `raw_z` passes the respiration band and nothing else, and is used only for timing.
2. Each detected onset corresponds to one real breath.
3. `RP`, `RA`, `RFR` are measured on the unedited recording, never on a filtered or repaired
   copy. (§9.1 qualifies the sample rate at which "unedited" is available; §9.2 is an
   exception for `n_peaks`.)
4. Each breath receives exactly one verdict, from its own measured properties, with a named
   reason; no later stage re-derives validity, and nothing is rejected on the score being
   scored.
5. Nothing is repaired or interpolated over: a rejected breath leaves a hole.
6. Nothing is trimmed. `len(series[m])` is invariant under rejection and no timestamp moves.
7. `NaN` is written after the causal filter, so a hole does not propagate.
8. The GLM solves on finite rows only.
9. No function in the analysis path reads the session key, the valence label, or `has_sound`.
   Between-session comparison happens only after every β is estimated.

Claims 4–9 are checked automatically by `Airflow/_scratch/debug_pipeline.py`, which currently
reports all invariants passing over the full cohort: 37 sessions, 1426 trials, 129 rejected
(9.0 %), reasons `low_valid_fraction` 107 and `no_cycles` 22, with no runtime warnings.
Claims 1–3 are not machine-checkable and are the substance of a visual review.
