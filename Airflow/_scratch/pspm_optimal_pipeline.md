# The PsPM-faithful Airflow pipeline

Written 2026-08-10. Design only; no code in `Airflow/` was modified. Every stage
below cites `pspm_resp_pp.m` / `pspm_init.m` / `pspm_glm.m` as transcribed in
`.claude/skills/pspm-respiration-audit/references/`. Where PsPM is silent, that is
stated rather than filled in.

---

## The headline finding

**PsPM does no artifact rejection in preprocessing at all.** `pspm_resp_pp.m`
enforces exactly one rule — `ibi < 1 s` drops the later cycle (Stage 4) — and the
reference file states plainly: *"No artifact/amplitude-based cycle rejection …
PsPM leaves this to the downstream GLM's own missing-epoch mechanism
(`model.missing`), not to `pspm_resp_pp`."*

So both candidate schemes are outside PsPM. The current ten per-trial gates in
`airflow_glm.py:1088-1098` are beyond PsPM, and the five-gate per-cycle scheme
proposed earlier is equally beyond PsPM. The PsPM-shaped answer is neither: mark
**bad time epochs, in seconds**, and pass them to the GLM as `model.missing`, so
the exclusion is a property of the design matrix rather than a verdict attached to
a trial or a breath. `model.missing` is defined at file level in seconds and
*"passed to the GLM rather than pre-filtered out of the input signal — i.e. PsPM's
own rejection model integrates missing data into the design matrix rather than
deleting samples beforehand."*

This matters for the condition-balance problem. A gate that labels *trials* or
*breaths* produces a count that differs between Evening and Morning and demands a
balance test. A missing-epoch list removes *time*, and the GLM's β is estimated
from whatever samples remain — the estimator is unbiased under missing-at-random
regardless of how much time each condition lost, provided the epochs are defined
by signal measurability and not by response magnitude.

---

## Stage list

| # | stage | PsPM source | what it does |
|---|---|---|---|
| 1 | mean-centre raw, **cascaded** filters, ↓10 Hz | `resp_pp` Stage 1 | `resp − mean(resp)`; `butter(1) LP 0.6 Hz` filtfilt, then `butter(1) HP 0.01 Hz` filtfilt; downsample to 10 Hz inside `pspm_prepdata` |
| 2 | median filter | `resp_pp` Stage 2 | `medfilt1(newresp, ceil(newsr)+1)` = 11 samples at 10 Hz |
| 3 | cycle detection, bellows rule | `resp_pp` Stage 3 | `find(diff(sign(newresp)) == -2)/newsr` — falling zero-crossings on the signal itself, **not** its derivative (flow sensor = bellows physics) |
| 4 | IBI floor | `resp_pp` Stage 4 | `ibi = diff(respstamp); respstamp(find(ibi<1)+1) = []`. One pass over the original list. **No upper bound** — the 10 s marker is a plotting flag |
| 5 | RP / RA / RFR per cycle | `resp_pp` Stage 5 | `RP = diff(respstamp)`; `RA = range(resp(win))` on the **native-rate untouched** `resp`; `RFR = RA / ibi` |
| 6 | assign + interpolate | `resp_pp` Stage 6 | value anchored to the cycle's **closing** crossing, `interp1(…,'linear')` onto a regular grid, nearest-neighbour at the edges |
| 7 | per-modality sensitivity filter | `pspm_init.m` | RP: HP 0.01 Hz · RA/RFR: HP 0.001 Hz · all LP 1 Hz · order 1 · **unidirectional (causal)** · down 10 Hz |
| 8 | basis + design | `pspm_bf_r*rf_e.m` | Gaussian on −10…+30 s; RP (4.2, 1.65) no derivative; RA (8.07, 3.74) + derivative; RFR (6.0, 3.23) + derivative; `spm_orth`; unit-range normalise |
| 9 | fit | `pspm_glm.m` | one design per session, per-condition event trains, `centering=1`, `latency='fixed'`, missing epochs in the design |

## What already matches, and should not be touched

Verified against the code, not assumed:

- Per-modality HP split is correct — `GLM_FINAL_HP = {"RP": 0.01, "RA": 0.001,
  "RFR": 0.001}` (`airflow_config.py:57`), the exact distinction the reference
  calls "the critical, easy-to-miss point".
- Basis parameters match the published table to the decimal —
  `GLM_RF_PARAMS` (`airflow_config.py:60-64`) and `GLM_USE_DERIVATIVE`
  (`:65`, RP off / RA on / RFR on).
- Basis support is the real −10…+30 s, not 0…30 (`airflow_glm.py:574-576`),
  with `onset_off` aligning t=0 correctly.
- `orthogonalize_and_normalize_basis` (`:492`) reproduces `spm_orth` + unit-range;
  `_spm_orth_columns` (`:451`) is the serial Gram-Schmidt with the 'pad' zeroing.
- Design pipeline order matches `pspm_glm.m` §14.2-14.4: convolve → causal HP →
  mean-centre → post-convolution orthogonalisation (`airflow_glm.py:608-622`).
- `detect_cycles` (`:219`) already reads amplitude from `signal_raw` rather than
  the filtered `raw_z`, and anchors each value at `assign_time` — PsPM Stages 5-6.
- The refractory implementation (`:263-270`) reproduces MATLAB's single-pass
  semantics exactly, including the `[0.0, 0.9, 1.7] → [0.0]` case.

The GLM half of this pipeline is in good shape. The changes below are all in
preprocessing and in the rejection architecture.

---

## The three changes that matter, in order

### 1 · RA/RFR are measured at 25 Hz, and PsPM says native rate

`detect_cycles` measures `range()` on `signal_raw`, which is the **25 Hz cache**
(`CACHE_SFREQ`, `airflow_config.py:17`). PsPM Stage 5 measures on `resp` at the
native rate — 1000 Hz here.

Concretely: at 25 Hz consecutive samples are 40 ms apart. Peak inspiratory flow is
a sharp early-breath feature, and the reference file flags this project
specifically — *"a flow trace carries genuinely more high-frequency content than
the slow, rounded bellows signal PsPM's defaults were tuned against … peak
inspiratory flow in particular tends to be a sharp, early-in-breath feature."*
A flow peak with a ~120 ms apex is described by 3 samples at 25 Hz versus 120 at
1000 Hz; `max−min` over 3 samples systematically lands below the true peak, and
the shortfall is larger for sharper (i.e. larger) breaths. That biases RA
**downward, non-uniformly, in proportion to how startling the breath was** — the
worst possible direction for this study, because the startle response is exactly
the sharp breath.

The fix does not require caching 1000 Hz traces (which `CLAUDE.md` forbids for load
time): run detection on the 10 Hz copy as PsPM does, then compute RA/RFR at native
rate **during Layer 1 load** and cache the per-cycle table, not the trace. Cost is
one `FORCE_RELOAD`; the cycle table is a few hundred rows per session.

### 2 · One `butter(2, [0.01, 0.6], 'band')` is not PsPM's filter

`GLM_CYCLE_BANDPASS` (`airflow_config.py:53`) is designed as a single 2nd-order
bandpass. PsPM Stage 1 is two separate 1st-order `filtfilt` passes. The reference
is explicit that this is a real difference and not bookkeeping: *"the order for
bandpass and bandstop filters is equal to order = lporder + hporder is a developer
note about nominal filter order bookkeeping, not a statement that PsPM designs a
joint bandpass … the frequency response differs (steeper transition band,
different phase behavior) even though 'order 2' sounds equivalent on paper."*

The effect is on **where zero-crossings land**, which propagates into RP for every
breath and into the RA measurement window. I am not going to assert a magnitude
without measuring the two responses. This is Layer 2 and free to tune, so the
honest move is to implement both and report the RP distribution under each.

Related and cheaper: `GLM_ZSCORE_RAW = True` (`:51`) z-scores the detection copy.
PsPM mean-centres only. Scaling does not move a zero-crossing, so this is
cosmetic for detection — but it is a deviation worth removing simply because
nothing depends on it.

### 3 · Replace both gate sets with a missing-epoch list

Drop the ten per-trial gates and do not add the five per-cycle ones. Build instead
a single list of `[start_s, end_s]` epochs where the signal is **not measurable**,
and hand it to the GLM the way `model.missing` does.

The criterion must be about measurability, never about response size — that is
what keeps the exclusion independent of the contrast being tested, and it is why
an amplitude-outlier gate like `extreme` is the wrong shape for this: a breath
that is 8× the session median may be an artifact or may be the startle response
itself, and nothing in the amplitude alone distinguishes them. Defensible epoch
sources, in descending order of how well they generalise:

1. **Sensor detached / dead signal** — a flat span, AASM's apnea criterion
   (< 10% of baseline amplitude for ≥ 10 s) applied as a time criterion rather
   than a per-breath one. External standard, not fitted to this cohort.
2. **Detector lock lost** — a span where the zero-crossing detector emits one
   "cycle" that visibly contains several inspiratory excursions. This is a
   *detection failure*, i.e. a statement about the algorithm, not about the
   physiology.
3. **Hand-marked artifact** — the labelling export already in
   `Airflow/labeling_export.py`. Slow, but it is ground truth, and calibrating a
   threshold against hand labels is categorically different from calibrating it
   against a p-value.

Everything PsPM does not need — kurtosis gates, amplitude z-gates, baseline-std
gates, rate ceilings — comes out. `IBI ≥ 1 s` stays, because that one is in the
source.

---

## What this buys, and what it does not

It buys traceability: every constant in the pipeline then has a citation, and the
only project-specific additions are an explicit, short, defensible missing-epoch
list. It also removes the condition-balance failure *as a gate problem*, because
there are no longer per-trial or per-breath gates whose firing rates must be
balanced — though a missing-epoch list still needs its total removed time reported
per condition, and a large asymmetry there is still evidence of something real.

It does not buy a result. The RA endpoint on the current scheme runs opposite to
the hypothesis, and nothing here is a reason to expect that to flip. Change 1 is
the only one with a plausible mechanism for moving it, because it corrects a bias
that acts specifically on large breaths — and it could move it in either
direction. Fix it because PsPM says native rate and the bias argument is sound,
then report what comes out.

## Order of work

1. Change 1 (cycle table at native rate, `FORCE_RELOAD`) — largest effect, and it
   changes the cache, so everything else should be measured after it.
2. Change 3 (rejection architecture) — pure deletion plus a small epoch list.
3. Change 2 (filter cascade) — measure both responses, then decide.

Re-run the endpoint once, after all three. Not after each.
