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

**What:** Convert raw Airflow (25 Hz cache) to z-scored 10 Hz trace.

**Steps:**
- Downsample from 25 Hz → 10 Hz (covers respiration band at Nyquist)
- Anti-alias lowpass filter (1st-order Butterworth, 5 Hz)
- Z-score the entire session using robust statistics (median/MAD, not mean/std)
  - Robust z-score preserves cycle *timing* but rescales amplitude into session-relative SD units
  - Handles outlier spikes (sensor noise, movement artifacts) without inflating global noise floor


---

### Phase 2: Detect Cycles → Compute Three Features

**What:** Identify breath cycles and extract RP, RA, RFR per cycle.

**Steps (from the paper):**
1. Bandpass filter (2nd-order Butterworth, 0.01–0.6 Hz) + 1s median filter
2. Detect negative zero-crossings on mean-centered signal → inspiration onsets
3. Per cycle, compute:
   - **RP (Respiration Period)** = duration from onset *i* to onset *i+1* (seconds)
     - Why: Period linearly relates to autonomic input (unlike rate, its inverse)
   - **RA (Respiration Amplitude)** = peak-to-trough of z-scored signal within the cycle (SD units)
     - Why: Linearly relates to tidal volume (Binks et al., 2007)
   - **RFR (Respiration Flow Rate)** = RA / RP (SD/sec)
     - Why: Linearly related to tidal volumetric flow rate (volume per unit time)


---

### Phase 3: Convert to Continuous Series

**What:** Turn per-cycle measures into smooth, session-wide time series.

**Steps (from the paper):**
1. Assign each cycle's RP/RA/RFR to the timestamp of the *following* inspiration onset
   - Why: Breaks dependency; each feature is "assigned to the start of the following inspiration cycle"
2. Linearly interpolate each metric to 10 Hz (full session)
3. **Filter twice with unidirectional 1st-order Butterworth bandpass** (0.001–1 Hz)
   - High-pass (0.001 Hz) removes DC (mean breathing rate)
   - Low-pass (1 Hz) removes jitter and high-frequency noise
   - Unidirectional (one-pass `lfilter`, not `filtfilt`) preserves causality
   - Paper: "Interpolated data were then filtered twice with a unidirectional first-order Butterworth band pass filter"

**Output:** Three continuous 10 Hz series: RP_series, RA_series, RFR_series (full session, z-scored SD units)

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
