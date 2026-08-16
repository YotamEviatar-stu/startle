# Airflow / Respiration GLM — Signal Processing Summary

Raw recording → scored trial. For review of the signal-processing chain only
(filtering, cycle detection, artifact handling, GLM scoring) — experimental
design and statistics are omitted below except where they constrain a signal
choice.

**Signal:** nasal airflow, one channel, EGI MFF recording, native rate
resampled to 25 Hz on cache.
**Method basis:** PsPM (Bach et al. 2016, *A linear model for event-related
respiration responses*), implemented against the PsPM v7.0.0 MATLAB source
(`extras/PsPM_v7.0.0/src/`). Line references below are to that source.

Cohort figures quoted here: 45 sessions, 24 subjects, 25 Hz.

> **⚠ Stale on rejection (checked 2026-08-12).** Stages **1b**, **4b** and **6**
> describe an excursion-gate / deep-breath-gate / per-SAMPLE scheme that is not
> the code on disk. What runs is a per-TRIAL gate set (`Airflow/_scratch/STATUS.md`
> §A); the cycle-based replacement is designed but unimplemented
> (`Airflow/_scratch/cycle_rejection_spec.md`). Stages 1a, 2, 3, 4, 5 and 7 —
> the filter, cycle detection, per-breath measurement, series construction and
> GLM — are current and were verified line-for-line against PsPM.
>
> Two known live defects, both reproducible from the cache: `flag_excursions`
> at k=5 flags **0 runs** in all of RP06/mor despite a 14.93 × `A` peak at
> t=1340.40 s (its 1 s rolling-median reference rises with the artifact); and a
> bad breath inside an admitted trial reaches `y` at full weight — MS18/eve
> cycles 18/19/110/111 are 3.28 % of samples and carry **61.1 %** of that
> session's Σy².

---

## Stage 1 — Signal integrity (QC)

### 1a. Session breath size `A`

```
A = 2 · 1.4826 · MAD( bandpass(signal_raw, 0.01–0.6 Hz) )
```

One number per session: the size of that subject's normal breath, in that
recording's own units — necessary because raw units are not comparable across
sessions (breath size spans `5.5e-7` to `1.6e-3` across the cohort, a factor of
2800).

- Band = PsPM's respiration band (`pspm_resp_pp.m` L86-89).
- `1.4826` = MAD→SD consistency constant. Exact only for a normal distribution;
  a breathing oscillation is not one, so this is conventional rescaling.
- `× 2` = one-sided spread → trough-to-peak, matching what PsPM's RA measures
  with `range()` (L141).

Validation: on a pure sine, `A` lands 4.8% above true peak-to-trough. On real
data a typical breath deviates 1.24·A from its own local centre (median over 45
sessions). Cohort median `A = 3.94e-4`.

### 1b. Excursion gate (fast glitches)

```
flag where  | signal_raw − rolling_median_1s |  >  5 · A
merge flagged runs separated by < 0.2 s
accept as artifact only if the run is shorter than 1 s
→ linear-interpolate across it, record the span in `mask`
```

A breath takes at least 1 s (PsPM's minimum inter-breath interval, L113-116),
so within a 1 s window the trace cannot legitimately travel more than one
breath's excursion.

- `5 · A`: a typical breath reaches 1.24·A; a sigh (2–3× tidal) reaches
  ~2.5–3.7·A; 5·A clears the largest plausible sigh by ~35%. This multiplier is
  a margin choice, not a derived constant — cohort p99.9 of the excursion ratio
  is ≤ 3.7 in all 41 non-degenerate sessions.
- Runs > 1 s are left untouched (sustained contamination is a session-level
  problem, not a glitch). Cohort-wide: 1783 flagged runs, median 0.08 s, p95
  0.32 s, exactly one exceeds the 1 s cap.
- Interpolation exists only so the downstream IIR filters stay well-behaved.
  `mask` records what was bridged; Stage 5 blanks it back to NaN
  (`pspm_prepdata.m` L74-85 + L143-145: interpolate → filter → restore NaN).

**Balance report:** Evening 8.87% of trials touched, Morning 8.67% — masking is
not condition-correlated. This is a *detector*, run after the fact: it exists to
expose a gate keying on something that co-varies with condition. It never passes
or fails a gate, and it is never a reason to move a threshold.

---

## Stage 2 — Filter (PsPM, unchanged)

`signal_clean` → `raw_z` @ 10 Hz:

1. Subtract the mean.
2. Lowpass 0.6 Hz, 1st-order Butterworth, zero-phase.
3. Highpass 0.01 Hz, 1st-order Butterworth, zero-phase.
4. Resample 25 → 10 Hz (0.6 Hz lowpass already anti-aliases — 8.3× below the
   5 Hz Nyquist).
5. Robust z-score (median / 1.4826·MAD).

Matches `pspm_resp_pp.m` L83-94 (`lpfreq=0.6/lporder=1`, `hpfreq=0.01/hporder=1`,
`direction='bi'`, `down=10`). Edge behaviour verified: PsPM's `pspm_filtfilt.m`
uses `nfact=3(nfilt−1)` odd-reflection padding with Gustafsson initial
conditions; `scipy.signal.filtfilt(method='pad')` matches.

---

## Stage 3 — Cycle detection (PsPM, unchanged)

`raw_z` → inspiration onsets:

1. Median filter, 1 s kernel.
2. Negative-going zero-crossings = inspiration onsets.
3. Refractory: drop onsets < 1 s apart, single pass over the original
   candidate list (`pspm_resp_pp.m` L113-116).

---

## Stage 4 — Per-breath measurement

For each consecutive onset pair:

| | |
|---|---|
| **RP** | respiration period, s — from onset times |
| **RA** | respiration amplitude — `max − min` over the cycle |
| **RFR** | flow rate — `RA / RP` |

**RA/RFR are measured on `signal_clean` (unfiltered trace), never on `raw_z`.**
Reproduces PsPM's two-variable split: `newresp` (filtered) locates breaths,
`resp` (untouched) measures them (L92-94 vs L140-141) — measuring depth on the
filtered copy would clip fast inspiratory-flow peaks, which matters more for a
flow sensor than PsPM's bellows signal. Each value is assigned to the
**following** onset.

### 4b. Deep-breath gate (slow excursions)

```
drop a breath whose  RA > 5 × median(RA of this session)
```

Catches slow, large excursions that Stage 1b structurally cannot see: a swing
lasting several seconds never deviates fast from the 1 s rolling median, but
because RA is `max − min` over the whole cycle it accumulates into a huge
amplitude. Worked example (RP06/eve, t=943–947s): sample-level deviation peaks
at 3.77·A (below the 5·A gate — correctly not flagged as a glitch), but the
resulting RA = 11.3× the session median, propagating a spike into the RA
series ~20× the surrounding signal.

Expressed as a ratio to the session median (not MAD units) on purpose:
MAD/median varies between sessions, so a fixed Hampel-k means different things
in different recordings (see Stage 6 note below). Cohort RA/median: p95=2.36,
p99=4.35, p99.9=25.39, max=62.6 — sighs occupy the 2–4× band, artifacts sit
above 20×. Gate drops 0.80% of 17,816 cycles.

Dropped breaths are recorded in the same mask as 1b and blanked at Stage 5.

---

## Stage 5 — Continuous series

`cycles[]` + `mask` → `series{RP, RA, RFR}` @ 10 Hz:

1. Linearly interpolate the one-value-per-breath knots across the session.
2. Causal sensitivity filter (`lfilter`, not `filtfilt` — no look-ahead):
   highpass 0.01 Hz (RP) / 0.001 Hz (RA, RFR), lowpass 1 Hz.
3. **Then** blank the masked spans to NaN.

Ordering is load-bearing: `lfilter` is recursive, so a NaN fed in propagates
through the feedback to every later sample (bridge → filter → blank).

Measured effect of blanking vs. letting the bridge stand as data: session mean
`score_ra` changes by a median of 15% (r=0.987 across sessions) — not
cosmetic.

**Known property of the method:** 20,299 knots across 807,945 series samples —
2.5% of what the GLM sees is measurement, 97.5% is the straight lines between
knots. Inherent to Bach et al.'s design (per-breath quantities interpolated
into a continuous measure). Any within-trial R², SE, or t-statistic computed
from these residuals would be meaningless (none is reported); the unit of
analysis downstream is the subject, not the sample.

---

## Stage 6 — Rejection: per-SAMPLE, not per-trial

All rules a-priori and symmetric across conditions/sessions. **There is no
per-trial gate on the GLM path.** Validity is a property of the continuous
session signal; invalid samples become NaN in `series` and drop out of the fit.

Array A (`compute_sample_validity`), decided from local signal properties only,
no trial context:

| Component | Rule | Basis |
|---|---|---|
| cycle coverage | sample outside any detected cycle | nothing measured there |
| amplitude spike | RA > median + 3.5·MAD·1.4826 (upper bound only) | Hampel identifier, `GLM_ARTIFACT_K` |
| rate plausibility | `60/RP` outside 5–40 bpm | see note below |
| flat / noisy | rolling local SD of `raw_z` (window = session median RP) below 0.1× or above 3 MADs of its robust reference | AASM apnea criterion (≥90% reduction) for the flat side |

Then Stage 4.6's `GLM_MAX_ABS_Z` ceiling (default 3.0) NaNs any sample with
`|RA(z)|`/`|RFR(z)|` above it.

A trial is marked `rejected` only when it has **zero** valid samples
(`no_valid_signal`) — bookkeeping, not a gate, so a trial with no data isn't
handed a pooled β it contributed nothing to. `valid_fraction` is reported but
never thresholded. Current: Eve 721/730 kept (98.8%), Mor 743/771 (96.4%).

The old per-trial gates (`no_cycles_found`, `low_information`, `low_coverage`,
`flat_signal`/`flat_response`, `noisy_baseline`, `atypical_shape`,
`rate_artifact`, `cycle_gap`) were removed: they were computed over
`[baseline_start, response_end)`, which bleeds into the next trial's fixation
period (true min trial→trial gap 12.49 s vs. a ~30 s response window), and
imposed a baseline-vs-response split with no meaning on one continuous signal
deconvolved as a single GLM.

**`rate_artifact = 40 bpm` is not literature-sourced.** PsPM's own bound is
period < 1s = 60 bpm, but the 1s refractory makes >60 bpm structurally
unreachable (0 of 20,381 cycles) — a 60 bpm gate would be a no-op. 40 bpm
("twice the upper limit of normal resting adult rate, 10–18/min") flags 0.28%
of cycles; 25 bpm would flag 4.5%.

**Both the ratio gate and the RA Hampel gate are live — this doc previously
claimed the ratio gate had replaced Hampel, which is not what the code does.**
`flag_deep_breaths` (`QC_MAX_BREATH_RATIO`=5.0) drops cycles before Array A,
and Array A then applies `_hampel_flag_cycles` (`GLM_ARTIFACT_K`=3.5) on the
survivors. The standing critique of the Hampel rule still applies and is
unresolved: MAD-k is not comparable across sessions since MAD/median varies
(k=3.5 → 1.64× min discarded depth, k=6 → 2.11×, k=10 → 2.95×, k=15 → 4.20×),
and because depth is the quantity RA measures, discarding shallow-but-normal
breaths biases the score downward. The ratio bound is stable by construction.
Whether Hampel should be dropped in favour of the ratio gate alone is open.

---

## Stage 7 — GLM scoring

Canonical response function: Gaussian `h(t) = exp(−(t−τ)²/2σ²)`, τ/σ fixed to
published values (RP: τ=4.20s; RFR: τ=6.00s; RA: τ=8.07s — from the
literature, not fit to this dataset).

Basis `[CRF, dCRF/dt]` orthogonalized (`spm_orth`) and unit-range normalized
before the intercept is added — matches `pspm_bf_r*rf_e.m`.

Two estimation architectures (`GLM_ESTIMATION`):

- **`pooled_session`** (current): one design matrix for the whole session, one
  event train per condition, one solve per metric → one β per condition,
  written onto each accepted trial. Faithful to `pspm_glm.m` (convolve →
  highpass the design → mean-centre → orthogonalize).
- **`per_trial`**: independent regression per trial — enables trial-level
  diagnostics, not what Bach et al. describes.

Score = **β₁**, the CRF regressor's weight — how strongly the trial matches
the expected response shape after the intercept absorbs resting level. Not a
peak amplitude, not a baseline-to-response difference.

---

## Known limitations

1. **Linear time-invariance** — one fixed response shape for every trial,
   subject, session. A shape change (e.g. habituation) is representable only
   as a smaller β₁, indistinguishable from a genuinely weaker response.
2. **Response timing is imported** — if this population's true latencies
   differ from the published τ, every score is scaled against a slightly
   wrong template; a systematic bias no p-value reveals.
3. **No overlap model** — trials are scored in isolation; a response bleeding
   past its own window is not detected or corrected.
4. **97.5% interpolation** (Stage 5) — per-trial β₁ is a low-information
   estimate; the pooled estimator is better-conditioned for this reason.
5. **`rate_artifact = 40 bpm`** has no primary literature source (Stage 6).
