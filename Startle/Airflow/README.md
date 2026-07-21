# Airflow GLM Pipeline

Implements the linear model from **"A linear model for event-related respiration responses"** (Dominik R. Bach 2016). 

## Run

```bash
.venv/bin/python -m Airflow.airflow_main
```

Set `SCORING_METHOD = "glm_deconvolution"` in `airflow_config.py`.

Output: `~/Desktop/airflow_output/airflow_trial_scores.csv` with columns `score_rp`, `score_ra`, `score_rfr`.

---

## Pipeline (4 Phases)

### Phase 1: Prepare Signal

**What:** Convert raw Airflow (25 Hz cache) to a z-scored 10 Hz detection trace (`raw_z`),
matching `pspm_resp_pp.m` Stage 1 exactly.

**Steps:**
- Despike (interpolate over brief, extreme-amplitude native samples — sensor glitches, not breaths)
- Mean-center the native-rate signal
- Two **cascaded 1st-order** Butterworth filters, each bidirectional (`filtfilt`):
  lowpass at 0.6 Hz, then highpass at 0.01 Hz — *not* a single combined 2nd-order bandpass
  design (PsPM designs these as two separate passes; the frequency response differs from
  one joint `butter(N=2, ...)` filter even though both are nominally "order 2")
- Downsample 25 Hz → 10 Hz (anti-aliased by the 0.6 Hz lowpass already applied — no
  separate anti-alias-only stage needed for a signal this narrowband)
- Z-score the entire session using robust statistics (median/MAD, not mean/std)
  - Robust z-score preserves cycle *timing* but rescales amplitude into session-relative SD units
  - Handles outlier spikes (sensor noise, movement artifacts) without inflating global noise floor
  - `raw_z` is used ONLY for cycle detection (Phase 2) — RA/RFR amplitude is measured on
    the native raw cache, not this trace (see Phase 2)

---

### Phase 2: Detect Cycles → Compute Three Features

**What:** Identify breath cycles on the Phase 1 trace, then extract RP, RA, RFR per cycle
from the native raw signal.

**Steps (from the paper):**
1. 1s median filter (smooths `raw_z` so only true breath oscillations trigger crossings)
2. Detect negative zero-crossings on `raw_z` → inspiration onsets (bellows-style rule —
   correct for a flow transducer like Airflow)
3. Per cycle, compute:
   - **RP (Respiration Period)** = duration from onset *i* to onset *i+1* (seconds)
     - Why: Period linearly relates to autonomic input (unlike rate, its inverse)
   - **RA (Respiration Amplitude)** = peak-to-trough of the **native raw cached signal**
     within the cycle window (native units, NOT z-scored/SD units) — measured on the raw
     signal, never on `raw_z`, matching PsPM's `resp` vs `newresp` separation. This
     matters more for a flow sensor than PsPM's bellows/chest-strap signal: flow ≈
     d(volume)/dt, so differentiation pushes real inspiratory-peak energy into
     frequencies the 0.6 Hz detection filter would otherwise clip.
     - Why: Linearly relates to tidal volume (Binks et al., 2007)
   - **RFR (Respiration Flow Rate)** = RA / RP (native units/sec)
     - Why: Linearly related to tidal volumetric flow rate (volume per unit time)


---

### Phase 3: Convert to Continuous Series

**What:** Turn per-cycle measures into smooth, session-wide time series.

**Steps (from the paper):**
1. Assign each cycle's RP/RA/RFR to the timestamp of the *following* inspiration onset
   - Why: Breaks dependency; each feature is "assigned to the start of the following inspiration cycle"
2. Linearly interpolate each metric to 10 Hz (full session)
3. **Filter with a unidirectional 1st-order Butterworth bandpass, PER METRIC** — PsPM does
   NOT share one high-pass across the three channels: RP gets 0.01 Hz, RA/RFR get 0.001 Hz
   (low-pass 1 Hz shared by all three)
   - High-pass removes DC (mean breathing rate) — more aggressively for RP than RA/RFR
   - Low-pass (1 Hz) removes jitter and high-frequency noise
   - Unidirectional (one-pass `lfilter`, not `filtfilt`) preserves causality
   - Paper: "Interpolated data were then filtered twice with a unidirectional first-order Butterworth band pass filter"

**Output:** Three continuous 10 Hz series: RP_series (seconds), RA_series (native signal
units), RFR_series (native units/sec) — full session length. None of these are in
z-scored/SD units; only the Phase 1 detection trace (`raw_z`) is z-scored.

---

### Phase 4: Fit Per-Trial GLM

**What:** For each trial, fit a canonical response function to each metric's series.

**Steps:**
1. Extract each metric's series over the trial window (`[baseline_start, response_end)`)
2. Design matrix: `[intercept, canonical RF regressor, (optional) derivative]`
3. Canonical response function (Gaussian):
   ```
   h(t) = exp(-(t - tau)² / (2 * sigma²))
   ```
   - RA: τ=8.07s, σ=3.74s (slow rise/fall, deeper breath ~8s post-trigger)
   - RP: τ=4.20s, σ=1.65s (faster cycle shortening)
   - RFR: τ=6.00s, σ=3.23s (elevated flow rate)
4. Fit via Moore-Penrose pseudoinverse (`np.linalg.pinv`)
5. **Score** = β₁ (the RF regressor coefficient)
   - Magnitude of template match: how well the trial's feature response matches the expected shape
   - Positive = response follows expected timing/shape
   - Three separate scores per trial: `score_rp`, `score_ra`, `score_rfr`

**Interpretation (by metric):**
- `score_rp` > 0 = cycles shortened (faster breathing)
- `score_ra` > 0 = breaths became deeper
- `score_rfr` > 0 = flow rate increased

---

## Rejection Gates

Trials are rejected if:
- `no_cycles_found`: Zero-crossing detector found no cycles AND NeuroKit2 cross-check disagreed
- `noisy_baseline`: Baseline std is an outlier (> 3.0 × session std)
- `flat_signal`: Baseline std too low (< 0.1 × session median std)
- `flat_response`: Response window std too low
- `rate_artifact`: Detected breathing > 40 bpm (movement artifact or cough)
- `atypical_shape`: Waveform deviates from session-median centroid

---

## Files

| File | Role |
|------|------|
| `airflow_config.py` | Tunable parameters (all phases, rejection thresholds) |
| `airflow_glm.py` | Phases 1–4 implementation |
| `airflow_main.py` | Orchestration: cache, scoring, CSV output |
| `airflow_amp_processor.py` | Layer 1 cache building (needed even for GLM) |
