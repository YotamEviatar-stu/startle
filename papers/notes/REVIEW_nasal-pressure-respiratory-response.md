# Nasal pressure as a channel for evaluating the respiratory response

*What the literature supports when an AC-coupled nasal cannula is the only respiratory
channel and the question is response magnitude and structure — not apnea scoring.*

Every claim carries the sensor it was measured on; a method is not portable across
sensors. Tags: `V` = read in the source · `V-abs` = abstract only · `derived` = arithmetic
on published values. Full source table:
`.claude/skills/nasal-pressure-flow-volume/references/source-map.md`.

---

## 1. The sensor measures pressure, not flow

A cannula senses the pressure drop across an orifice, and an orifice is quadratic:
`P = K·V̇·|V̇|`. The nose is better described by Rohrer, `ΔP = K₁V̇ + K₂V̇²`. Montserrat
et al. 1997 — nasal prongs measured **simultaneously against a pneumotachograph**, n = 6
awake adults, the one direct-sensor validation — fitted Rohrer with "an excellent fit… in
all cases" `V`. At 0.5 L/s the quadratic term is 3.1× the linear one on **inspiration**
but only ~1.05× on **expiration**, giving effective exponents of **n ≈ 1.69 inspiratory,
1.43 expiratory** `derived`.

`K` is a property of the mounting, not the subject. Removing and refitting the prongs
changed `K₁+K₂` "in some cases of more than 100%" `V`; Thurnheer 2001 found √-transformed
pressure tracking pneumotachograph flow at r² = 0.88–0.96 over ten breaths "but… highly
variable if comparisons were extended over an entire night" `V-abs`. **The gain is
locally faithful and globally untrustworthy** — every rule below follows from that.

## 2. Timing is exact and transform-free

`sign(P)·√|P|` is monotone and odd: it maps 0→0 and moves no point in time. Onsets, Ttot,
respiratory rate, Ti, Te, Ti/Ttot, time-to-peak and the rank order of breath sizes are
therefore **identical before and after the transform**. Amplitude, peak, contour and the
integral are not. Montserrat's warrant is explicitly temporal: "the excellent time
response of NP allows the precise detection of each different component of or event
within the breathing cycle" `V`. BreathMetrics (Noto 2018, nasal airflow, DC-coupled)
implements it — onsets by zero crossing on the trace, 94% within 100 ms on clean data `V`.
Nothing in this literature derives Ti or Te from a volume integral.

**The threat to timing is the zero, not the transform.** Because pressure ≈ flow², the
waveform dwells near zero, so a pressure-level error `ε` lands at flow level `√ε`. On a
4 s sinusoidal breath, a zero error of 1% of peak pressure costs 6.4% in Ti; 5% costs
**14.4%**; 10% costs 20.5% — while **Ttot is 4.000 s in every case** `derived`, since a
constant offset shifts both crossings the same way. So period and rate are robust to the
baseline and the inspiratory/expiratory split is not. A ~14% Ti error exceeds the
affective Ti effects in this literature. Two cases break outright: **Te under a
post-expiratory pause** (BreathMetrics refuses a zero crossing there, substituting an
amplitude-histogram noise range `V` — report Te as Ttot − Ti or detect pauses), and
**time-to-peak on a flattened contour**.

## 3. Amplitude: the field is split, and the split is quantitative

**Raw as flow:** ATS/Pamidi 2017 — raw is "a useful first approximation", and the workshop
"recommended continued **recording** of flow signals without linearization" `V`;
Hosselet 1998 (47,685 breaths) — "nearly linear in the range of normal breathing", and
breath classification "not significantly affected" by the square root `V`; AASM 2012 —
"with **or without** square root transformation" `V` *(consensus, not evidence)*.
**Square root first:** Montserrat 1997 — √ "acceptably fitted" pneumotachograph flow and
is "the simplest method of correcting for the observed nonlinearity" `V`;
Thurnheer 2001 `V-abs`; and ATS 2017 itself, in the same section, permits software
linearization: "some current algorithms [use] the square root–transformed pressure
signal" `V`.

**Read the disagreement precisely.** ATS's objection is not the exponent — they state the
relation "is close to quadratic" `V` — it is the missing zero (§4). The two are different
problems. The numbers, from Montserrat's worked example of a true **50%** inspiratory flow
reduction `V`:

| Signal | Apparent reduction | Bias |
|---|---|---|
| raw pressure | 69 ± 3% | **+19 points** (over-reads) |
| √pressure | 44 ± 3% | **−6 points** (under-reads) |

√ cuts the bias to one third. Generally: a 30% drop in raw pressure is a **16.3%** true
flow drop, and a 90% drop is only **68.4%** `derived` — so **a percentage change is not
transform-invariant; always state which signal it was computed on.** On the *expiratory*
limb the same calculation gives +12.8 raw versus −11.0 after √ `derived` — **√ buys almost
nothing there**, and `K_insp ≠ K_exp` besides. Hence: **amplitude and volume metrics
inspiration-only**; I:E *timing* is unaffected, since timing does not depend on gain.

## 4. Volume: the integral is right, the baseline is the problem

Volume is the integral of flow, so the order is forced — **transform, then integrate**
(`∫P dt` is not a volume; `√(∫P dt)` is meaningless). Integrate **within a breath, never
across** (BreathMetrics: "the integral of the airflow amplitudes within each breath's
onset and offset" `V`). The reason is arithmetic: an offset δ of 5% of a 0.5 L/s peak
accumulates **~60 L of phantom volume** over 40 unreset minutes `derived`. Resetting
bounds the error without removing it — for a half-sine inspiration, δ biases per-breath
volume by `1.571·(δ/A)`: **+7.9% at δ = 5%** `derived`, and because the error scales with
Ti while the signal scales with A·Ti, it **systematically inflates small breaths relative
to large ones**. A pause inside the window is pure bias.

This is ATS 2017's objection verbatim, the key sentence for AC-coupled hardware `V`:

> "…there is potentially an incorrect 'zero' to the signal that arises from applying the
> 'linearization,' and **this error cannot be detected from the hardware output or
> reversed in software**."

Note what makes it bite: **Montserrat's √ was validated on a signal sitting on the physical
no-flow zero** — the paper reports no high-pass filtering and the Validyne MP45 is
DC-responsive. (Their referring the prongs to mask pressure `V` is a *rig correction* that
recovers free-breathing conditions inside the mask, not an extra reference an ordinary
cannula lacks — the gap is **AC coupling**.) So **zero estimation is a load-bearing,
explicitly reported stage** carrying more validity than the transform. No
method for it is validated on this sensor; candidates in descending order of support: a
genuine no-flow reference (end-expiratory pause or breath-hold), the amplitude-histogram
mode, a running low-percentile envelope. Avoid the running *mean* — that is what AC
coupling already imposes and what ATS names as the failure mode. **The alternative** is to
skip the integral and use peak inspiratory √-amplitude: one step less exposed to the
baseline, and what AASM's amplitude rules are written on — but a *flow* proxy, blind to
the short-deep vs long-shallow distinction.

## 5. It is a reference, not litres

Three groups on three sensors refuse to name the quantity a volume. Bach 2016 (chest
bellows belt) declines the subject-specific constant: "we use the term RA rather than V_T
throughout" `V`. Noto 2018 (nasal airflow): ground-truth volume "requires temperature and
barometric information", so they report **normalized** volume `V`. Hosselet 1998 (nasal
cannula, uncalibrated) gives the template on exactly this sensor — each measurement
referred to that subject's own quiet wakefulness, "defined as… 100% for that subject" `V`.

The reference window must be **short and local**: AASM's baseline is the 2 min preceding
the event (or the 3 largest breaths in it if unstable) `V` — a standard that assumes
non-stationarity — and the nasal cycle runs **2.02 ± 1.7 h** awake (Kahana-Zweig 2016,
n = 33, `V-abs`), so a 40-minute session sits inside one half-cycle and can drift
monotonically for non-experimental reasons. Carley 1997 (pneumotachograph, acoustic
stimulation) supplies the response architecture: 3 prestimulus vs 4 poststimulus breaths,
normalized to each subject's prestimulus mean `V`. **Report "relative inspiratory volume
(a.u., % of local baseline)", naming the baseline window and the zero-estimation method.
Never V_T, never mL, never L/min.**

## 6. What the channel cannot deliver

Absolute V_T / flow / minute ventilation (K unknown) · cross-session amplitude comparison
(the cannula is re-placed) · ground-truth volume even if calibrated (needs temperature and
barometry) · obstructive-vs-central (needs effort) · AASM hypopnea/RERA/AHI (need
desaturation or EEG arousal) · any defensible apnea call · snore (needs ~100 Hz).

**Mouth breathing is the failure mode with no software fix.** Nasal pressure "cannot
detect or estimate the magnitude of oral airflow" `V`; Zelano 2016 found their effect
"dissipated when breathing was diverted from nose to mouth" `V-abs`. It does two damages —
amplitude becomes ambiguous between hypoventilation and an oral switch, *and* the zero
moves, since unidirectional mouth expiration breaks the inspired = expired assumption the
floating baseline rests on. **A collapsed stretch is a rejection, not a small breath.**
Awake, aroused subjects switch often; budget for epoch loss and fix the rule before
looking at condition labels.

Filtering can manufacture or destroy the morphology: AASM requires DC or high-pass
**≤0.03 Hz**, and shows that at 0.1 Hz flattening is destroyed `V`; ATS asks ≥10 Hz
sampling for rate and **25–50 Hz for inspiratory shape** `V`.

## 7. Open questions

1. **The exponent has never been measured on an uncalibrated, AC-coupled cannula** — every
   direct validation had a simultaneous calibrated reference and a DC-coupled signal. A self-calibrated
   Rohrer form (curvature ratio `r = K₁/K₂` borrowed, median ≈0.15; `K₂` anchored locally)
   is roughly unbiased on relative change where plain √ under-reads by ~6 points
   *(digest)*. Cost: one borrowed constant plus a per-epoch gain anchor.
2. **Whether canonical response functions transfer is untested.** Bach 2016's RP λ = 4.20 s,
   RA λ = 8.07 s, RFR λ = 6.00 s were fitted on **rib-cage circumference**, and the
   supporting "RA linearly relates to tidal volume" is a circumference argument — on a
   cannula, per-cycle range is a peak-*pressure* quantity ≈ flow². Also: in Bach's own
   data, **85 dB vs 65 dB white noise was discriminated by none of RP, RA or RFR** `V`.
3. **Timing and volume may not share an arousal range.** Gomez 2008 (calibrated RIP,
   n = 37) found the volume family strong under defensive activation (Vi F = 9.01,
   Vi/Ti F = 15.68, MV F = 20.50) with Table 2 timing contrasts null `V` — but their §4.5
   model makes it a **crossover, not a timing null**: Ti shortening carries low-to-moderate
   arousal, Vi takes over at high arousal `V`. Gomez 2004's Ti finding is unreplicated and
   sign-unstable; never cite it without 2008. Corollary: timing carries the range this
   sensor reads well, volume the range where effects are largest and it reads worst.

---

## Summary

A nasal cannula is a **pressure** sensor reading approximately the square of nasal flow
(effective exponent ≈1.7 inspiratory, ≈1.4 expiratory). **All timing metrics are exact and
transform-free** — the strongest use of the channel — though only period and rate survive
a baseline error; Ti and Te do not. **Amplitude requires an unsettled choice**: raw
over-reads a relative change by ~19 points, √ under-reads by ~6, and on expiration √
barely helps at all, which is the argument for making amplitude and volume
**inspiration-only**. **Volume is the integral of the transformed signal, reset at every
inspiratory onset**, and rests less on the transform than on an estimated zero the
hardware does not supply. The result is a **within-session relative quantity in arbitrary
units against a short local baseline — not litres**, the posture Bach 2016, Noto 2018 and
Hosselet 1998 each adopted in their own words.

*Note: the inspiratory/expiratory Rohrer assignment follows*
`references/transfer-function.md` §3, *which reproduces the paper's worked example; Table 1's
OCR columns are ambiguous, and Montserrat report the insp/exp difference was observed but*
**not statistically significant** *at n = 6.*
