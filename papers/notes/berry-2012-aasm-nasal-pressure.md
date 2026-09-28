# Berry et al. 2012 — AASM Sleep Apnea Definitions Task Force
"Rules for Scoring Respiratory Events in Sleep: Update of the 2007 AASM Manual"
*J Clin Sleep Med 2012;8(5):597–619* — doi:10.5664/jcsm.2172
(file: `papers/fressure to flow 2.pdf`)

**What it is:** not a physiology/signal-processing paper. It is the consensus document (modified RAND process, 13-member task force) that defines which *sensor* is authoritative for which *event*, and what numeric thresholds count. Most recommendations are graded (Consensus) — i.e. expert agreement, not level-1 evidence. That matters for citing it: it is the *standard*, not the *evidence*.

---

## 1. The pressure→flow relationship (the core physics claim)

- A nasal cannula connected to a pressure transducer produces a signal **proportional to the square of flow**: `P ∝ V̇²` (ref 36, Montserrat/Farré line of work).
- Consequence of the non-linearity: **the signal underestimates low flow rates**. A shallow breath looks disproportionately smaller than it is. Practically: *"the signal underestimates low flow rates and could result in a hypopnea appearing to be an apnea"* (p. 600).
- **Square-root transform** (`V̇ ∝ √P`) "more closely approximates flow and minimizes this problem."
- Two important caveats the paper adds, which are the reason the transform is *optional* not mandatory:
  1. **The transform's accuracy degrades over a night** of monitoring — cannula position shifts, nares change (ref 23, Thurnheer).
  2. **The AHI effect is small**; the AHI computed on the transformed signal is *slightly lower* than on the raw signal (refs 1, 23).
- Net rule: nasal pressure "with or without square root transformation" is acceptable for adults. For children the 2007 manual specified the **untransformed** signal; the 2012 update harmonizes to "with or without."

## 2. Why nasal pressure is the *recommended* hypopnea sensor

Explicit reasons given (p. 602):
- **Simplicity**
- **Sensitivity** — "sensitive to even subtle changes in airflow"
- **Shape information** — "the ability for scorers to easily recognize changes in flow based on changes in *shape* as well as *amplitude*"

Gold standard for comparison is a **pneumotachograph** (pressure drop across a linear resistance, in a full-face mask) — explicitly "not practical for clinical studies."

Head-to-head evidence cited:
| Comparison | Result | Ref |
|---|---|---|
| NP vs. uncalibrated RIPsum, event-by-event vs. pneumotach | **NP agreement higher** than RIPsum | Heitman (38) |
| NP vs. calibrated RIPflow, flow-limitation detection | **NP more sensitive** | Clark (25) |
| NP (transformed or not) vs. calibrated RIPflow, AHI bias vs. pneumotach | **similar bias** | Thurnheer (23) |
| PVDFsum vs. NP | PVDFsum "not as sensitive for detecting events (based entirely on flow)" | (6) |

## 3. Why nasal pressure is *not* the recommended apnea sensor

- **Mouth breathing.** NP "may show decreased excursions during mouth breathing" (ref 32) — a subject breathing orally with fully patent airway produces a near-flat NP trace. NP "cannot detect or estimate the magnitude of oral airflow." This is called out as *the* major disadvantage.
- Plus the flow² non-linearity above (hypopnea → false apnea).
- So: **oronasal thermal sensor is recommended for apnea**; NP is listed as an *alternative* sensor (used only if the recommended sensor fails / is unreliable). Other alternatives: RIPsum, RIPflow, PVDFsum (adults, Acceptable), end-tidal PCO₂ (children, Acceptable).
- Converse asymmetry: **thermal sensors are bad for hypopnea.** Their signal "is not proportional to flow and often overestimates flow as flow rates decrease" (ref 33). Excursions do drop during hypopnea "although not as prominent as those in the nasal pressure signal" (ref 19). Thermal sensors are a *presence/absence* detector; NP is an *amplitude* detector.

## 4. Waveform shape: flattening = inspiratory flow limitation

- **Flattening of the inspiratory portion of the NP waveform is a surrogate for airflow limitation** (refs 19, 24, 29–31). This is the single most transferable idea in the paper for anyone doing morphology-based respiratory analysis.
- It is load-bearing in three rules:
  1. **RERA** (§4.3): ≥10 s sequence of breaths with increasing effort *or* inspiratory flattening of the NP signal, leading to arousal, not meeting apnea/hypopnea criteria. The paper concedes effort is *inferred* here, not measured — some authors therefore prefer **"flow limitation arousal"** (ref 31) over "RERA."
  2. **Obstructive hypopnea** (§4.2.1): score obstructive if ANY of — snoring during the event / **increased inspiratory flattening of NP vs. baseline** / thoracoabdominal paradox during but not before the event.
  3. **Central hypopnea**: score central if NONE of those three are met. Note flattening is judged *relative to that subject's baseline breathing*, not against an absolute template — "flattening is present but unchanged from baseline" counts as central.
- Task force explicitly notes RIP excursion amplitude **cannot** separate obstructive from central hypopnea (both decrease); morphology + paradox + snore do the work.

## 5. Filter settings — the most concretely actionable technical content

This is §3.1.1 + **Figure 1**, and it is the part most people miss.

- To see flattening at all, the NP signal must be recorded either as **DC** or as **AC with a low-frequency cutoff ≤ 0.03 Hz**.
- **Figure 1** shows the same NP epoch at DC, 0.01, 0.03, and 0.10 Hz: *"At a low filter setting of 0.1 Hz, the ability to demonstrate airflow flattening is impaired."* The high-pass filter differentiates away the flat top of a flow-limited breath and re-creates a rounded/peaked contour — i.e. **your filter choice can manufacture or destroy the flow-limitation feature.**
- **High-frequency cutoff 100 Hz** preserves the snoring oscillations riding on the NP signal (ref 32). Hence NP (unfiltered) is one of the three accepted **snore sensors**, alongside piezoelectric neck vibration and acoustic microphone.
- The PAP device flow signal **cannot** show snoring — "either filtered or too under-sampled to show the high-frequency vibrations." Same channel role, different bandwidth.
- The task force formally recommends that high/low filter settings for NP "be specified in future revisions of the scoring manual" — i.e. as of 2012 they were *not* standardized, which is a real confound when pooling across labs/datasets.

## 6. Amplitude rules that use nasal pressure

- **Adult hypopnea:** ≥30% drop in **peak signal excursion** of NP from pre-event baseline, ≥10 s, **AND** (≥3% desaturation **OR** arousal). (2007 recommended rule was 30% + ≥4% desat; alternative "4B" was 50% + 3%/arousal.)
- **Pediatric hypopnea:** same 30% NP drop, but duration ≥ **2 breaths** instead of 10 s, + ≥3% desat or arousal.
- **Removed:** the old requirement that the qualifying drop occupy **>90% of the event duration** — for both apnea and hypopnea. (Figure 3 shows the case this fixes: a 24 s flow drop inside a 38 s event that was previously unscoreable.)
- **Baseline definition** (carried over from the 1999 Chicago paper): mean amplitude of stable breathing in the **2 minutes preceding** the event; if breathing is unstable, **mean of the 3 largest breaths** in those 2 minutes. If baseline can't be determined, terminate the event on a clear sustained rise in amplitude, or on ≥2% resaturation.
- **Event duration** is measured on the NP signal for hypopnea, on the oronasal thermal signal for apnea (PAP device flow for both during titration).
- Threshold sensitivity is large: Table 5 (Ruehland, n=320) — % with AHI ≥5/h was **92% (Chicago) vs 59% (2007 recommended, 4% desat) vs ~85% (2007 alternative)**. Same recordings, different definition → nearly a factor of 1.6 in diagnosis rate.

## 7. Sensor hierarchy, condensed

| Purpose | Recommended | Alternatives |
|---|---|---|
| Apnea (diagnostic) | Oronasal thermal (incl. PVDF airflow) | **NP** (±√), RIPsum, RIPflow, PVDFsum (adult, Acceptable), etCO₂ (child, Acceptable) |
| Hypopnea (diagnostic) | **Nasal pressure transducer (± √ transform)** | Oronasal thermal, RIPsum, RIPflow, dual RIP belts, PVDFsum (adult, Acceptable) |
| Apnea + hypopnea (PAP titration) | PAP device flow | none specified |
| Effort | Esophageal manometry, dual RIP belts | dual PVDF belts (adult, Acceptable) |
| Snore (Optional) | **Unfiltered NP**, piezo, microphone | — |
| SpO₂ | Pulse oximetry (appropriate averaging time) | — |

Both an oronasal thermal sensor **and** a nasal pressure transducer are to be recorded in every diagnostic study — the two are complementary, not redundant.

---

## Transferable takeaways for a pressure-based respiratory signal

1. **Raw nasal/mask pressure is a squared-flow signal.** If you compare amplitudes across breaths of different size, the compression at low flow is real and biases small breaths downward. √-transform to compare amplitudes; keep raw if you care about high-frequency content (snore) or if drift over a session would make the transform unstable.
2. **The √ transform is not free** — its calibration drifts within a session, and its effect on aggregate indices is small. Decide per-metric, not globally.
3. **Shape carries information amplitude does not.** Inspiratory flattening ≈ upper-airway flow limitation, and it is defined *relative to that subject's own unobstructed baseline*.
4. **High-pass filtering destroys flattening.** ≤0.03 Hz or DC. A 0.1 Hz high-pass is enough to make flow-limited breaths look normal — a silent, purely-preprocessing-induced artifact.
5. **Nasal pressure is blind to oral flow.** Any drop in a nasal-only pressure signal is ambiguous between "reduced ventilation" and "switched to mouth."
6. **Baseline is a 2-minute local window**, not a whole-recording constant — the standard itself assumes non-stationarity.
