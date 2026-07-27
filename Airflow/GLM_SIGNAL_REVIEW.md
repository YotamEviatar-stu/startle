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

**Balance check:** Evening 8.87% of trials touched, Morning 8.67% — masking is
not condition-correlated.

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

## Stage 6 — Trial rejection gates

All gates a-priori and symmetric across conditions/sessions.

| Gate | Rule | Basis |
|---|---|---|
| `no_cycles_found` | < 2 onsets in window, **or** NeuroKit2 (khodadad2018) finds no real breath | 2 onsets = arithmetic minimum for one period |
| `low_information` | < 5 breaths in window | fit has 3 parameters; 5 knots leaves ≥2 residual df |
| `low_coverage` | < 50% of window samples finite | a trial fitted on a fraction of its window must not carry equal weight |
| `flat_signal` / `flat_response` | baseline/response SD < 0.1 × session median | AASM apnea criterion (≥90% reduction) |
| `noisy_baseline` / `atypical_shape` | > median + 3·MAD·1.4826 | Hampel identifier, k=3 |
| `rate_artifact` | any cycle > 40 bpm | see note below |
| `cycle_gap` | longest inter-onset gap implies < 5 bpm | PsPM flags period > 10s (L207) |

**`rate_artifact = 40 bpm` is not literature-sourced.** PsPM's own bound is
period < 1s = 60 bpm, but the 1s refractory makes >60 bpm structurally
unreachable (0 of 20,381 cycles) — a 60 bpm gate would be a no-op. 40 bpm
("twice the upper limit of normal resting adult rate, 10–18/min") flags 0.28%
of cycles; 25 bpm would flag 4.5%.

**The RA Hampel gate was replaced by the ratio gate in Stage 4b.** The old
rule (`median(RA) + 3.5·MAD·1.4826`) dropped 5.60% of cycles cohort-wide, and
the shallowest discarded breath was only 1.64× median depth — normal
breathing, not artifact. Because depth is the quantity RA measures, that gate
biased the score downward. Root cause: MAD-k is not comparable across
sessions since MAD/median varies (k=3.5 → 1.64× min discarded, k=6 → 2.11×,
k=10 → 2.95×, k=15 → 4.20×). The Stage 4b ratio bound is stable by
construction and drops only 0.80% while still catching the 11.3× artifact
above.

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
