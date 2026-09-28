# Nasal pressure ↔ flow: signal-to-signal reference (healthy adult, quiet nasal breathing)

## 1. The transfer function

A nasal cannula at the nares senses the pressure drop of an **orifice/jet**, not a linear resistance:

```
P = K · V̇ · |V̇|          (sign-preserving square; |P| ∝ V̇²)
V̇ = sign(P) · √(|P| / K)  (inverse: the "square-root transform")
```

`K` is an unknown, subject- and fit-specific gain (cannula geometry, nostril geometry, prong seating). **PSG nasal pressure is never volume-calibrated** — you recover flow *shape*, never absolute L/s, unless you calibrate against a pneumotachograph in that session.

**Reality check on the exponent.** The nose is better described by Rohrer: `ΔP = K₁V̇ + K₂V̇²`. Montserrat's *inspiratory* coefficients (run 1) are **K₁ = 0.76 ± 0.6 cmH₂O/(L/s), K₂ = 4.76 ± 3.7 cmH₂O/(L/s)²** — at 0.5 L/s the quadratic term (1.19) is ~3× the linear one (0.38), so inspiration is close to pure-quadratic and √ works well there. *Expiration* is the opposite case (K₁ = 1.14, K₂ = 2.39 → terms roughly equal at 0.5 L/s), which is why the paper found linearization "better for the inspiratory phase." A fitted power law therefore lands near n ≈ 1.8 in inspiration, ≈1.5 in expiration. This is the mechanistic reason AASM 2012 permits nasal pressure "with **or without** square-root transformation" rather than mandating it.

## 2. Physical magnitudes to expect (awake or NREM, healthy adult)

| Quantity | Typical value |
|---|---|
| Respiratory rate | 12–16 /min → fundamental **0.2–0.27 Hz** |
| Tidal volume | 0.4–0.6 L |
| Ti / Te | ~1.5–1.8 s / ~2.2–2.8 s; Ti/Ttot ≈ 0.35–0.45 |
| **Peak inspiratory flow** | **~0.4–0.6 L/s** (≈1.5–1.6 × mean inspiratory flow) |
| Total nasal resistance | ~1.5–3 cmH₂O/(L/s) at 0.5 L/s (≈0.15–0.3 Pa·s/cm³) |
| Trans-nasal ΔP at peak inspiration | **~0.5–1.5 cmH₂O (50–150 Pa)** |
| Cannula-recorded swing | a fraction of that, ~0.1–1 cmH₂O; **strongly device-dependent, usually plotted in arbitrary units** |
| Sign convention | inspiration = negative at the cannula (suction); PSG displays it **inverted** so inspiration is up |

## 3. What actually changes between the two signals

### Preserved exactly (the squaring is monotonic and odd)
- **Zero crossings** → breath onsets, Ti, Te, Ti/Ttot, respiratory rate, IBI variability. All transform-invariant. If your metric is timing-based, **don't bother transforming.**
- **Sign / direction** of flow.
- **Rank order** of breath sizes (monotonicity) → Spearman correlation between raw-NP amplitude and true flow amplitude is 1.0; only Pearson degrades.

### Distorted
- **Amplitude ratios** — the big one, see §4.
- **Waveform shape.** Raw NP is a *sharpened* version of flow: peakier crest, broader dwell near zero. For a sinusoidal flow input, the crest factor (peak/RMS) goes **1.414 → 1.633**.
- **Harmonic content.** Because `V̇·|V̇|` is half-wave antisymmetric, it generates **odd harmonics only** — no frequency doubling. For sinusoidal flow: H3 = **20%** of the fundamental, H5 = 2.9%, H7 = 1.0%, **H2 = 0**. Practical consequence: spectral respiratory-rate estimation on *raw* nasal pressure is safe (the fundamental still dominates and there's no 2× artifact), but any harmonic-ratio or spectral-shape feature is contaminated by the transform, not by physiology.
- **Integral.** ∫P dt is **not** tidal volume. You must √-transform *first*, then integrate — and even then only up to the unknown `1/√K`.

### Created by the inverse transform
- **Noise blow-up at zero crossings.** d(√x)/dx → ∞ as x → 0, so the √-transformed signal is noisiest exactly at the inspiratory–expiratory transitions. Filter *before* transforming, not after.

## 4. Amplitude mapping — the table to keep

| True flow drop | Raw NP drop | | Raw NP drop | True flow drop |
|---|---|---|---|---|
| 10% | 19% | | 30% | **16.3%** |
| 20% | 36% | | 50% | 29.3% |
| **30%** | **51%** | | 75% | 50.0% |
| 50% | 75% | | **90%** | **68.4%** |
| 68.4% | **90%** | | | |
| 90% | 99% | | | |

Two consequences that make the AASM text concrete:

1. **The apnea/hypopnea confusion.** AASM scores apnea at a ≥90% drop in peak excursion. On a *raw* nasal pressure trace, 90% is reached when true flow has fallen only **68%** — a hypopnea. That is exactly the paper's "the signal underestimates low flow rates and could result in a hypopnea appearing to be an apnea," and it's why nasal pressure is only an *alternative* apnea sensor.
2. **Why transformed AHI comes out slightly lower.** A 30% drop scored on raw NP corresponds to a true flow drop of only **16%** — a permissive threshold. Apply the same 30% on the √-transformed signal and you now require a genuine 30% flow drop (= 51% on raw). Fewer events qualify → the transformed AHI is lower, matching Berry 2012 / Thurnheer.

## 5. Morphology in a healthy, unobstructed adult

- Inspiratory contour should be **rounded / roughly sinusoidal**; flattening index ≈ 0. Flow limitation is *absent* in a healthy awake adult breathing nasally.
- Squaring **increases** the flat-vs-rounded contrast, which is why scorers do fine on raw NP: a flow-limited flat top stays flat when squared, while a normal rounded breath becomes visibly more pointed.
- **I:E amplitude asymmetry in NP is not the I:E flow asymmetry.** `K` differs between inspiration (nasal valve narrows, air accelerates inward) and expiration (jet impinges on the prongs), so the two half-cycles have different gains. Don't read expiratory/inspiratory amplitude ratio off raw NP.
- **Mouth breathing collapses the signal** with a fully patent airway. Any large NP amplitude drop in a healthy adult is ambiguous between reduced ventilation and an oral switch — the reason AASM requires a thermal sensor alongside.

## 6. Filtering (from Berry 2012 §3.1.1, Fig. 1)

- Record **DC-coupled**, or AC-coupled with a high-pass cutoff **≤ 0.03 Hz**. At **0.1 Hz the flattening feature is destroyed** — a flow-limited breath is restored to a rounded contour. Given RR ≈ 0.2 Hz, a 0.1 Hz high-pass is only half a decade below the fundamental, which is nowhere near enough for a flat-topped (harmonic-rich) waveform.
- Low-pass **100 Hz** if you want the superimposed snoring oscillations; nasal pressure is an accepted snore sensor, PAP-device flow is not (filtered/undersampled).
- Filter settings were **not standardized across labs** as of 2012 — a real confound when pooling nasal-pressure morphology across datasets.

## 7. Decision rule

| If your metric is… | Transform? |
|---|---|
| Rate, Ti/Te, timing, variability, zero crossings | **No** — invariant |
| Tidal-volume surrogate, minute-ventilation surrogate | **Yes**, then integrate |
| Relative amplitude drop vs. a threshold | **Yes**, if the threshold is defined on *flow*; state which you used |
| Flattening / flow-limitation morphology | Either, but be consistent; raw slightly enhances contrast |
| Spectral shape, harmonic ratios | **Yes** — raw adds a 20% third harmonic that is pure transform artifact |
| Snore / high-frequency content | **No** — use raw, wideband |
