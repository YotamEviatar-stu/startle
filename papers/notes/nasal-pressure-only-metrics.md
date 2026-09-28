# What you can derive from nasal pressure alone (no other sensors)

Short version: **yes, if every metric is scale-free.** `K` in `P = K·V̇|V̇|` is unknown, subject-specific, and drifts whenever the prongs move — so absolute flow is unrecoverable, but *ratios, timings and shapes* are. Montserrat et al. 1997 say this outright in their conclusion: *"no calibration is required to detect relative changes in nasal flow."*

---

## 0. First, correct the model — the linear term is NOT negligible

Montserrat 1997 (AJRCCM 155:211, `papers/Evalutaion of Nasal Prongs for flow montserrat 1998.pdf`) fitted Rohrer to nasal prongs vs. pneumotachograph in healthy subjects and reported (inspiration, mean ± SD, n=6):

| Coefficient | Value | Range across subjects |
|---|---|---|
| K₁ (linear) | **1.14 ± 1.0** cmH₂O/(L/s) | 0.15 – 2.86 |
| K₂ (quadratic) | **2.39 ± 1.0** cmH₂O/(L/s)² | 1.47 – 4.72 |

At a typical peak inspiratory flow of **0.5 L/s**:
- linear term = 1.14 × 0.5 = **0.57 cmH₂O**
- quadratic term = 2.39 × 0.25 = **0.60 cmH₂O**

**Roughly half the signal is linear at tidal flows.** So `P ∝ V̇²` is an idealization; the effective exponent sits near 1.5. Consequences, from their own worked example (0.5 → 0.25 L/s, i.e. a true 50% flow reduction):

| Signal | Reported reduction | Error |
|---|---|---|
| Raw NP | **69 ± 3%** | +19 points (over-reads) |
| √NP | **44 ± 3%** | −6 points (under-reads) |
| True flow | 50% | — |

√ is not exact, but it cuts the bias to **one third**. Note their empirical 69% vs. the pure-square prediction of 75% — that gap *is* the linear term. Also: intersubject variability in K₁, K₂ is large (K₁ spans 19×), and they had to re-fit after remounting the prongs — so **never port a K across subjects or across a remount.**

---

## 1. Tier A — exact, no assumptions, no transform needed

Squaring is monotonic and odd, so it moves no zero crossings. Everything below is identical on raw and √ signals:

- **Ttot** (breath period), **Ti**, **Te**, **Ti/Ttot** (duty cycle), **Ti/Te**
- **Respiratory rate** = 60/Ttot
- **Post-inspiratory / expiratory pause duration**
- **Breath-to-breath variability of timing**: SD and CV of Ttot, Ti, Te; Poincaré SD1/SD2 on the Ttot series; sample/approximate entropy of the interval series; autocorrelation of Ttot
- **Event timing**: breath onsets relative to a stimulus, phase of stimulus within the respiratory cycle, inspiratory-onset latency, respiratory-phase resetting
- **Apnea/pause detection** by duration
- **Rank ordering of breath sizes** (Spearman with true flow amplitude = 1.0; only Pearson degrades)

Bach et al. 2016 (`papers/A linear model for event-related respiration responses bach 2016.pdf`) build their event-related respiration GLM on exactly this class: respiration period (RP), amplitude (RA), and flow rate (RA/RP), explicitly declining to estimate the subject-specific constant — *"we use the term RA rather than V_T throughout."* That is the right posture for nasal pressure too.

## 2. Tier B — relative amplitude (K cancels in a within-window ratio)

Apply √ first, then normalize to a **local** baseline. All of these are unitless "% of baseline":

- **Relative peak inspiratory flow**: `√|P|peak(breath) / median(√|P|peak over baseline window)`
- **Relative tidal volume**: `∫√|P| dt` over inspiration, ÷ the same integral over baseline breaths
- **Relative minute ventilation**: relative V_T × RR
- **Relative mean inspiratory flow** (V_T/Ti analogue) — the drive index
- **Flow-reduction events**: % drop vs. baseline, with duration
- **Amplitude variability**: CV of relative V_T, Poincaré on the amplitude series

**Hard constraint: the baseline window must be short.** K drifts within a session (prong seating, nostril patency, head position, congestion cycle). This is not a nuisance detail — it is why AASM defines baseline as *the 2 minutes preceding the event*, or the mean of the 3 largest breaths in those 2 minutes if breathing is unstable. Use a rolling local baseline; never normalize to a session-wide constant, and never compare raw amplitudes across sessions or subjects.

## 3. Tier C — shape / morphology (dimensionless by construction)

Normalizing each breath by its own peak or mean cancels K exactly, so these are the most robust NP-only features:

- **Flattening index** — the standard is the SD of the middle 50% of the inspiratory flow segment, after normalizing that segment to its own mean. Low SD = flat top = inspiratory flow limitation. *Published cutoffs vary with the exact definition (windowing, normalization, filtering); fit your own on your data rather than importing a number.*
- **Time-to-peak-flow / Ti** (inspiratory peak position) — flow-limited breaths peak early
- **Inspiratory-contour skewness and kurtosis**
- **Crest factor** (peak/RMS) of the inspiratory half-cycle
- **Snore / high-frequency oscillation power** — use the **raw, wideband** signal (needs ≥100 Hz preserved), not √
- **I:E contour asymmetry** — usable as a *within-subject* feature only; see the caveat below

Two morphology caveats specific to NP:
1. **K differs between inspiration and expiration** (nasal valve narrows inward; the expiratory jet impinges on the prongs). Montserrat found the insp/exp Rohrer differences were not statistically significant in their n=6, but the point estimates differed — so treat inspiratory and expiratory amplitude as two separate channels with their own gains, and do not read an I:E *amplitude* ratio off the trace.
2. **Filtering can fabricate or destroy flattening.** DC-couple, or high-pass ≤ 0.03 Hz. At 0.1 Hz the feature is gone (Berry 2012 Fig. 1).

## 4. What you genuinely cannot get from NP alone

| Not derivable | Why | Minimum extra sensor |
|---|---|---|
| Absolute V_T (mL), V̇ (L/s), V_E (L/min) | K unknown | pneumotachograph / spirometer calibration |
| **AASM hypopnea, RERA, AHI** | definitions *require* ≥3% desaturation **or** EEG arousal | pulse oximeter + EEG |
| Obstructive vs. central classification | needs thoracoabdominal paradox and/or effort | RIP/PVDF belts or esophageal manometry |
| Any apnea call that survives scrutiny | a mouth-breathing switch is indistinguishable from cessation | oronasal thermal sensor |
| Respiratory effort / upper-airway resistance | inferred only, never measured | esophageal manometry |
| Hypoventilation | needs PCO₂ | et/tc CO₂ |

The mouth-breathing blind spot is the one that bites hardest outside sleep labs: in an awake, task-engaged, or emotionally aroused subject, oral switching is common, and it looks exactly like a large ventilation drop. Budget for a rejection rule (e.g. flag windows where relative V_T collapses without a matching change in timing or shape) rather than assuming it away.

---

## 5. Reference values for a healthy adult — and what's already on disk

**Normative resting breathing pattern** (Tobin et al. 1983, *Chest* 84:202, healthy adults, seated, awake — the standard citation):

| Metric | Value |
|---|---|
| RR | 16.6 ± 2.8 /min |
| V_T | ~0.50 ± 0.10 L |
| Ti | ~1.6 s |
| Te | ~2.2 s |
| Ti/Ttot | ~0.41 ± 0.05 |
| V_T/Ti | ~0.32 L/s |
| V̇_E | ~8.1 ± 2.2 L/min |
| breath-to-breath CV of V_T | ~25% (variability is large and normal) |

Only the **timing** rows and **Ti/Ttot** are directly checkable against an NP-only recording. V_T, V̇_E and V_T/Ti are usable as *targets for a relative measure*, not as absolute comparisons.

**Papers already in `startle-1/papers/` that carry the relevant reference values:**

| File | Use it for |
|---|---|
| `Evalutaion of Nasal Prongs for flow montserrat 1998.pdf` — Montserrat 1997 | **The** nasal-prong calibration paper: Rohrer coefficients, the 69%/44%/50% worked example, "no calibration required for relative changes" |
| `fressure to flow 2.pdf` — Berry 2012 AASM | Filter settings, flattening→flow-limitation, threshold definitions, sensor hierarchy |
| `A linear model ... bach 2016.pdf` + `.txt` excerpt — Bach 2016 | Event-related respiration GLM; RP/RA/RFR as deliberately uncalibrated measures; their filter sweep (0.001–1 Hz high-pass) |
| `boiten.main.pdf` — Boiten 1998 | Affect-related changes decomposed into respiratory *cycle components* (Ti, Te, V_T) — normative effect sizes for emotional stimuli |
| `gomez.main.pdf` — Gomez 2004 | Respiratory responses to affective picture viewing — valence/arousal → rate vs. depth |
| `wientjes 1998.main.pdf` — Wientjes 1998 | Drive vs. timing decomposition of breathing pattern under mental load |
| `The human ventilatory response to stress rate or depth tipton 2017.pdf` — Tipton 2017 | Review: ventilatory response to stress, "rate or depth?" — the framing question for any NP-only stress metric |
| `Characterizing and modeling breathing dynamics napoli 2022.pdf` — Napoli 2022 | Modeling breathing dynamics: flow rate, rhythm, period, frequency — distributional reference |

Also worth pulling if you don't have them: **Farré et al. 2001** (AJRCCM 163:494, "Relevance of linearizing nasal prongs for assessing hypopneas and flow limitation during sleep") and **Thurnheer, Xie & Bloch 2001** (AJRCCM 164:1914, accuracy of nasal cannula pressure for assessing ventilation during sleep) — these are the two follow-ups that quantify how much the linearization actually matters in practice.

---

## 6. Practical recipe for an NP-only pipeline

1. Record DC or high-pass ≤0.03 Hz; keep wideband if you want snore. Note the sampling rate and every filter — they are part of the measurement.
2. Detrend with a **very** slow high-pass (≤0.01 Hz) or a running median over ≥30 s, to kill prong-seating drift without touching the breath waveform.
3. Split at zero crossings → derive **all Tier A timing metrics on the raw signal**. These are your most trustworthy numbers.
4. Apply `√|P|·sign(P)` → derive Tier B relative amplitudes against a **rolling 1–2 min local baseline**. Low-pass *before* the √ (the derivative of √ blows up at zero crossings, so noise concentrates exactly at the I/E transition).
5. Derive Tier C shape indices per breath, each normalized to its own mean/peak.
6. Report every amplitude metric as "% of local baseline," never in cmH₂O or L — and say so explicitly in the methods.
