# Amplitude/Gain Normalization Across Subjects — GLM (RA) Scoring Review

Scope: whether the GLM path's raw-unit β₁ (`score_ra`, `airflow_glm.py`) should be
rescaled for a subject's own dynamic range (breath depth / lung size), the way the
BxB path (`airflow_amp_processor.py`) already does by dividing each trial by that
session's median NeuroKit2 `RSP_Amplitude`. This is a scoring-normalization
question, checked against PsPM/Bach et al. (2016) as primary source plus the
broader psychophysiology normalization literature.

Sources consulted: `papers/main.pdf` (Bach, Gerster, Tzovara, Castegnetti 2016, *J
Neurosci Methods* 270:147-155 — the paper the GLM path is modeled on),
`papers/bjy045.pdf` (Noto, Zhou, Schuele, Templer, Zelano 2018, *Chemical Senses*
43:583-597 — BreathMetrics), the live `bachlab/PsPM` `develop` branch source
(`src/pspm_glm.m`, fetched directly — see below), and the project's own
`pspm-respiration-audit` skill references (`pspm-glm-respiration.md`,
`pspm-preprocessing.md`), which already held relevant PsPM ground truth and were
checked before any external search. `papers/AnnalsATS.201704-318WS.pdf` and
`papers/fphys-12-772295.pdf` were checked (grepped in full) and contain no
normalization-relevant content for this question. `papers/papers.txt` lists three
additional non-PDF references (Wilhelm et al. 1999; two PMC review articles) not
held locally as files — not re-fetched since they weren't germane to a targeted
grep of what's on disk mattering here.

---

## 1. Direct answer: does PsPM/Bach et al. recommend per-subject amplitude normalization for the GLM path?

**Not silence — an explicit, tested, opt-in option, off by default.**

`pspm_glm.m` (bachlab/PsPM, `develop` branch) has a documented `model.norm` field:

```matlab
% ├─────.norm:  [optional] normalise data; default 0
```

and later, before the design matrix is built:

```matlab
% 11.1 normalise if desired --
if model.norm
  no_nan = ~isnan(Y);
  Y = (Y - mean(Y(no_nan)))/std(Y(no_nan));
end
```

This is a **within-session z-score of the entire continuous channel** (mean/std
taken over the whole recorded session for that one data file), applied *before*
the GLM design matrix is constructed and β is estimated — not a per-trial
normalizer, and not derived from post-stimulus windows specifically (see §5 on
why that distinction matters for circularity). It defaults to **off** (`0`), i.e.
**raw units are PsPM's default for the GLM path** — a user must opt in.

Bach et al. (2016) ran exactly this comparison and report it as "raw data" vs.
"Z-scored data" in their Table 3 (experiment 4, N=20):

| measure | raw: overall effect | z-scored: overall effect |
|---|---|---|
| RA | t(19)=1.89, p=0.074 (n.s.) | t(19)=2.74, **p=0.013** |
| RFR | t(19)=4.91, p<0.001 | t(19)=5.17, p<0.001 |
| RP | t(19)=−2.68, p=0.015 | t(19)=−3.11, p=0.006 |

Their own conclusion, quoted directly: *"When using z-scored data for
construction of response functions and model inversion, results were essentially
unchanged (Table 2); the overall positive response in RA was significant now [it
wasn't in raw units]."* — i.e. z-scoring **did not change which effects existed**,
but shifted RA from non-significant to significant, consistent with removing
between-subject nuisance variance in the same direction our BxB path's
normalization is designed for.

They also state the motivation is imported from SCR methodology, not native to
respiration: *"Previous studies on SCR have suggested an improved sensitivity by
z-scoring raw data before analysis (Bach, 2014; Bach et al., 2009), which is only
possible in a within-subject design."*

Separately, in their Discussion, Bach et al. name the actual physiological reason
raw RA needs a caveat at all: *"the mapping between rib cage circumference and
lung volume might differ between participants such that our RA and RFR measures
contain between-subject variance of no interest, due to individual anatomy
(Binks et al., 2007)... one can usefully approximate true RA with our RA measures
up to a linear constant that varies between subjects."* They leave this
unresolved (proposing a two-belt calibrated system as a future fix), and explicitly
say their own group-level inference approach is the **standard summary-statistics
approach**: *"we enter single-participant response estimates into a group level
t-test... in keeping with the bulk of psychophysiological literature."* This is
the same architecture this project's confirmatory Wilcoxon uses (per-subject mean
β₁ → group-level test), just with a non-parametric test substituted for small N.

**Bottom line for Q4**: PsPM is not silent — it offers within-session z-scoring as
a named, tested, optional feature, motivated by the SCR literature and shown (in
Bach et al.'s own data) to improve sensitivity for RA specifically without
changing the qualitative pattern of results. But it is **off by default**, and the
paper's own headline group-level analyses (Table 2, Fig. 4) are reported on raw
units — z-scoring is presented as a robustness check, not the primary analysis.
Raw-unit β is the accepted default convention; per-subject/per-session
normalization is a documented, legitimate, but non-default alternative.

---

## 2. Survey across psychophysiology domains

### SCR (skin conductance response)

- **Range correction** — Lykken, D.T. & Venables, P.H. (1971). "Direct measurement
  of skin conductance: a proposal for standardization." *Psychophysiology*, 8(5).
  Establishes expressing a subject's SCR/SCL in proportion to that subject's own
  observed min–max range, to remove between-subject differences in electrode/skin
  gain before pooling. Extended in Lykken (1972), "Range correction applied to
  heart rate and to GSR data," *Psychophysiology* 9(3) — the same logic applied to
  HR.
- **Boucsein, W. (2012). *Electrodermal Activity* (2nd ed.), Springer.** — the
  standard EDA reference text; documents range-correction and log-transform as
  the two conventional between-subject normalization approaches for SCR amplitude,
  motivated by the same skin-conductance-gain heterogeneity problem as Lykken &
  Venables.
- **Bach's own PsPM-adjacent SCR work** — Bach, Flandin, Friston & Dolan (2009),
  "Time-series analysis for rapid event-related skin conductance responses,"
  *J. Neurosci. Methods*; Bach (2014), "A head-to-head comparison of SCRalyze and
  Ledalab," *Biol. Psychol.* 103C — both cited directly inside `main.pdf` as the
  source of the z-scoring practice tested in §1. So even the SCR-side precedent
  Bach et al. imported into the respiration paper is itself an optional add-on
  in the SCR modeling literature, not a bedrock assumption.

### Respiration

- **Bach et al. (2016)** (this project's canonical GLM reference) — see §1: no
  respiration-specific normalization convention pre-existed; they imported
  z-scoring from SCR as an explicit borrow, tested once, kept optional.
- **Binks, A.P., Banzett, R.B., Duvivier, C. (2007).** "An inexpensive, MRI
  compatible device to measure tidal volume from chest-wall circumference."
  *Physiol. Meas.* 28:149-159 — the individual-anatomy calibration problem Bach
  et al. cite as the reason raw RA is only interpretable "up to a linear constant
  that varies between subjects."
- **Noto, Zhou, Schuele, Templer, Zelano (2018), BreathMetrics** (`papers/bjy045.pdf`)
  — offers, as a user-selectable option (not the validated default pipeline),
  z-scoring of respiratory amplitudes: *"parameterized such that researchers can
  customize noise removal and drift correction, or z-score respiratory
  amplitudes if they choose, although custom options are not validated here"*
  (p.587). Separately advertises *"Breathing-rate-normalized breath waveforms for
  comparison between subjects"* as a distinct feature — but that normalizes
  **timing** (breath duration/phase), not amplitude/gain, and shouldn't be
  conflated with the amplitude-rescaling question here.
- No tidal-volume-percent-of-predicted convention (the kind used in clinical
  spirometry, e.g. percent-predicted FVC) appears in any of the four locally-held
  PDFs — that convention exists in pulmonary medicine but was not carried into
  the psychophysiological event-related-response literature Bach et al. sit in.

### HR/HRV

- **Task Force of the European Society of Cardiology and NASPE (1996).** "Heart
  Rate Variability: Standards of Measurement, Physiological Interpretation, and
  Clinical Use." *Circulation* 93:1043-1065. Documents normalized units (LF/HF
  expressed as a percentage of total power minus VLF) specifically to *"minimize
  the effect of changes in total power on the values of LF and HF"* — i.e. a
  within-subject relative-power normalization for frequency-domain HRV, not a
  between-subject amplitude rescale. This project's HR pipeline (`HR/`) reports
  BPM directly and is not itself doing frequency-domain HRV, so this convention
  is adjacent, not directly transferable, but worth naming since it's the field's
  standard-setting document on the general question of "when do you normalize a
  physiological amplitude/power measure before comparing."
- Percent-change-from-baseline HR (a distinct, simpler convention widely used in
  event-related HR designs, e.g. Castegnetti et al. 2016 cited inside `main.pdf`
  itself as the HR analogue of this project's GLM approach) is the more directly
  comparable convention — same architecture as Bach et al.'s per-participant β,
  no explicit cross-subject amplitude rescale beyond the model's own intercept.

### Pupillometry

- **Mathôt, S., Fabius, J., Van Heusden, E., Van der Stigchel, S. (2018).** "Safe
  and sensible preprocessing and baseline correction of pupil-size data."
  *Behavior Research Methods*, 50:94-106. Compares subtractive
  (`corrected = pupil − baseline`) vs. divisive (`corrected = pupil / baseline`,
  i.e. percent-of-baseline) correction and recommends **subtractive** by default,
  specifically because divisive correction is disproportionately sensitive to
  baseline artifacts and very small baseline values. This is a *within-trial*,
  *within-subject* baseline normalization (removing that trial's own pre-stimulus
  pupil size), analogous to this project's β₀ intercept absorbing each trial's
  own flat level (`GLM_METHOD_FOUNDATIONS.md` §3) — not a between-subject
  amplitude/gain rescale, and its recommendation (subtractive > divisive) argues
  if anything *against* a divisive-style normalizer (dividing by a median breath
  amplitude, as the BxB path does) on artifact-sensitivity grounds, though
  pupillometry's noise profile (blink artifacts) isn't a direct analogue to
  respiration's.

---

## 3. Tradeoffs for this project's specific design (small-N, paired within-subject, Wilcoxon signed-rank, Evening vs Morning)

The confirmatory test never pools raw amplitude *across* subjects — each subject
is only ever compared to their own Morning value. That means a fixed,
time-of-day-invariant per-subject gain (e.g. lung size, chest-strap placement)
cancels in each subject's own Eve−Mor contrast, regardless of whether the raw
units are ever rescaled. But three real, literature-grounded caveats remain:

**a. Percent signal change relative to a subject's own baseline (subtractive is
default; divisive is the "percent" variant).**
Mathôt et al. (2018) shows divisive/percent-change is more fragile than
subtractive when the baseline is near zero or contaminated by artifact — directly
relevant here since a near-zero pre-stimulus respiration baseline (a brief
breath-hold or a very shallow rest breath) would blow up a percent-change score.
The GLM's β₀ intercept is already the subtractive analogue (§3 of
`GLM_METHOD_FOUNDATIONS.md`); adding a *further* divisive step on top (dividing
β₁ by a session-median amplitude, as BxB already does) reintroduces exactly the
artifact-sensitivity Mathôt et al. warn about, unless the divisor is a robust
statistic over many cycles (which BxB's session-median already is, mitigating
but not eliminating the concern).

**b. Within-subject z-scoring (PsPM's `model.norm`, §1).**
Legitimate and precedented (Bach et al. 2016 test it explicitly), but note the
granularity PsPM implements it at: **per session/file**, not per subject pooled
across sessions. For this project, Morning and Evening are separate session
files. Z-scoring *within each session separately* would force each session's
respiration channel to its own mean-0/std-1 before fitting — which risks
partially removing the very between-condition (Eve vs Mor) gain difference the
study is testing for, if genuine reactivity differences manifest partly as
overall signal variance rather than purely as CRF-shaped bumps. A **per-subject,
cross-session** normalizer (pool both sessions' cycles, compute one
subject-level scale, apply to both) would avoid that specific failure mode and is
the correct granularity if this is ever adopted — matching what BxB already does
architecturally (one session-level divisor per session, though as currently
implemented BxB's divisor is also computed per-session, not pooled across
Eve+Mor — worth checking if BxB has the same granularity concern before treating
it as a clean reference implementation).

**c. Ipsative/min-max range normalization (Lykken & Venables 1971, Boucsein
2012).**
The SCR-native convention. Requires an independent, non-response-window-derived
estimate of that subject's min/max range (e.g. a resting/deep-breath calibration
period) — this project has no such calibration trial, so a defensible range
normalizer would have to be built from whole-session cycle amplitudes (as BxB's
session-median already approximates), not from the startle-response windows
themselves.

**d. Raw units, unnormalized (current GLM default).**
Not bias-free even under a paired design, in one specific sense: Wilcoxon
signed-rank does not merely check the *sign* of each subject's Eve−Mor
difference — it ranks the **magnitudes** of those differences across subjects. A
subject with an intrinsically larger raw dynamic range (a deep, large-excursion
breather) will produce a numerically larger raw Eve−Mor difference than a
shallow breather showing the *same relative* effect, and will therefore receive
disproportionately more weight (higher rank) in the test statistic. This does not
inflate Type-I error (the null distribution of signed ranks under permutation is
still valid regardless of each subject's fixed gain), but it can reduce power to
detect a genuinely relative (percent-scale) effect that is masked by unrelated
between-subject gain differences riding along in the rank ordering. This is the
concrete version of the "interpretability of raw β units" concern named in the
task brief — it is a power/interpretation issue, not a validity issue, for this
specific paired test.

**What would be double-dipping/circular if done wrong**: any normalizer whose
mean/std/range is computed **from the same post-stimulus trial windows** that
β₁ is then divided by — e.g. z-scoring using the mean/SD of the accepted trials'
own peak responses, or range-normalizing using the max response seen in that
session's startle trials. PsPM's `model.norm` avoids this by z-scoring the
**entire continuous session channel** (baseline + response + everything else
folded together) before any trial windows are even extracted, and BxB's
session-median divisor is computed over **all detected breaths**, not
specifically the accepted startle-response trials — both are defensible on this
count precisely because the normalizer's source data is broader than, and
computed independently of, the specific values being normalized.

---

## 4. Does this change `extras/emg_raw_potentiation.py` or `extras/trial_epochs.py`?

**No.** This is a scoring/normalization-layer question (how β₁ or a peak
amplitude gets rescaled after extraction), not an epoch-extraction or
trial-structure question. `extras/trial_epochs.py`'s job (ground-truth trial windows
from DIN triggers) and `extras/emg_raw_potentiation.py`'s canonical event/baseline logic
are upstream of and unaffected by whatever normalization convention the Airflow
GLM scorer eventually adopts. No change to either file is implied by anything
found here.

This finding also does not, on its own, argue for changing `airflow_glm.py`'s
current raw-unit default — it documents that raw units are PsPM's own default
too, names the real (if secondary, power-oriented rather than validity-oriented)
tradeoff versus normalizing, and flags the specific circularity and granularity
traps (§3b, §3-double-dipping) to avoid if a normalization scheme is adopted
later. Any such change remains the user's call, to be pre-registered rather than
chosen by watching the Eve-vs-Mor p-value, per this project's existing
multiplicity discipline (`GLM_METHOD_FOUNDATIONS.md` §7, `CLAUDE.md`).
