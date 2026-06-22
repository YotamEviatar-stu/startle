---
name: startle-experiment
description: Full briefing on the startle experiment design, data layout, CSV structure, DIN triggers, and EMG as the canonical reference pipeline. Invoke at the start of any new session before touching ECG, SpO2, RESP, or Airflow code.
---

# Startle Experiment — Full Context

## What this skill does

Loads everything an agent needs to work on this project without asking the user to re-explain the experiment. Read it once per session, then code.

---

## The experiment

Subjects participated in an overnight sleep study. Each subject completed two sessions:

- **Evening session** (`_1.csv`) — before sleep
- **Morning session** (`_2.csv`) — the morning after

In each session, subjects viewed a series of images. Each image was either **Negative** or **Neutral**. On some trials a **startle probe** (loud sound, D110 trigger) was played mid-image. Subjects rated each image for **valence** (1-9, low = negative) and **arousal** (1-9, high = arousing) after viewing.

The physiological signal of interest is whatever channel is being analysed (EMG, ECG, SpO2-Pulse, airflow). The core question is always: **how does the signal change in response to the startle probe?**

---

## The four comparison axes

| Axis | Values | Where it comes from |
|---|---|---|
| **Session** | Evening vs Morning | MFF filename keyword (`eve` / `mor`); CSV suffix (`_1` / `_2`) |
| **Image** | Negative vs Neutral | `image_type` column (objective) or arousal >= 7 / valence <= 3 (subjective) |
| **Sound** | Present vs Absent | `has_sound` column in CSV (True = startle trial) |
| **Signal change** | % change from pre-startle baseline | Computed per-trial from the physiological channel |

All pipelines filter to `has_sound == True` rows only. The no-sound trials exist in the CSV but are never analysed.

---

## CSV anatomy

```
has_sound          bool   True = startle probe was played (keep); False = no probe (drop)
arousalRating      float  1-9; high = more aroused
valenceRating      float  1-9; low = more negative
image_type         str    "negative" / "neutral"  (objective ground truth)
image / image_name str    image filename (varies by subject/session)
```

**Subjective label rule** (`USE_SUBJECTIVE_TRIAL_TYPE = True`):
- Negative: `arousalRating >= 7` OR `valenceRating <= 3`
- Neutral: everything else

**Objective label rule** (default, `USE_SUBJECTIVE_TRIAL_TYPE = False`):
- Read `image_type` column: `"negative"` -> label 1, `"neutral"` -> label 2
- Fall back to subjective if column is missing or ambiguous

**Session file naming:**
```
<SubjectID>_1.csv   Evening ratings
<SubjectID>_2.csv   Morning ratings
```
Located in `<SubjectFolder>/startle output/` (or directly in the subject folder).

---

## MFF file layout and DIN triggers

```
RAW_DATA_DIR/
  <SubjectID>/
    EEG/
      *_eve_*.mff    Evening EEG recording
      *_mor_*.mff    Morning EEG recording
    startle output/
      *_1.csv        Evening ratings
      *_2.csv        Morning ratings
```

**DIN trigger codes (same across all pipelines):**
```
D101  session start  crop recording here
D124  session end    crop recording here
D110  startle probe  t = 0 for every epoch
```

DIN channels have prefix `D`. Threshold at 90% of channel max. Contiguous above-threshold samples = one event (keep first sample only).

**Event alignment**: Nth D110 trigger <-> Nth `has_sound == True` CSV row. Sequential order, no timestamps.

---

## EMG (`emg_raw_potentiation.py`) — the canonical reference

**This script is the foundation. What works here works for every other channel.**

The full pipeline:
1. Load MFF, crop to D101-D124
2. High-pass filter (Butterworth 4th order, > 28 Hz) — EMG-specific; other channels do NOT filter
3. Rectify (`abs()`) then low-pass smooth (Butterworth 4th, < 30 Hz) — EMG-specific
4. For each D110: cut wide epoch `[-250 ms, +250 ms]`
5. Baseline = mean of `[-50 ms, 0 ms]` pre-stimulus window
6. Reject trial if baseline std > 3 SD (z-score) OR amplitude > 100 uV (hard limit)
7. Trim accepted epochs to `[-50 ms, +120 ms]` for analysis
8. Score = max peak in `[20 ms, 100 ms]` minus baseline mean

**Two-layer cache design (same for all pipelines):**
- `process_session()` loads MFF, cuts epochs, stores raw data -> written to pickle
- `apply_analysis_params()` computes baseline, rejection, score -> runs every time from cache
- Changing filter or wide window params requires `FORCE_RELOAD = True`
- Changing baseline/score/rejection params needs no reload

**Full plot suite in EMG (target for every other channel pipeline):**
```
plot_individual_trials()              one PNG per trial; rejected = grey, Neg = red, Neu = blue
plot_trial_scores_per_subject()       score vs trial number scatter per subject/session
plot_group_boxplot_four_conditions()  Eve-Neg / Eve-Neu / Mor-Neg / Mor-Neu with connecting lines
plot_group_ratio_boxplot()            Neg/Neu ratio per subject, Eve vs Mor, with statistics
plot_group_average_timecourse()       mean +/- SEM epoch per condition
plot_subject_average_timecourse()     per-subject mean epoch (one figure per subject)
plot_subject_psd()                    Welch PSD of raw signal per subject/session
plot_group_session_comparison()       group-level Eve vs Mor comparison
plot_group_eve_mor_ratio()            per-subject ratio of Evening to Morning score
plot_group_overall_eve_vs_mor()       collapsed across Neg/Neu: Eve vs Mor boxplot
plot_group_overall_ratio()            overall Neg/Neu ratio
plot_group_overall_timecourse()       grand mean timecourse all conditions
plot_stai_correlations()              STAI-T vs score / subjective ratings correlation grid
plot_subjective_negative_percentage() % of trials rated Negative, Eve vs Mor boxplot
```

---

## Channel-specific notes

| Channel | Nature | Filter? | Score units | Notes |
|---|---|---|---|---|
| **EMG** (E238 / E241) | Raw muscle potential (uV) | HP 28 Hz + rectify + LP 30 Hz | uV above baseline | Reference implementation |
| **ECG** (varies) | Raw cardiac signal (uV) | HP + LP Butterworth | HR in bpm via R-peaks | R-peak detection required |
| **SpO2-Pulse** | Pre-processed BPM from oximeter | None | % change from baseline | Steps every ~2 s; Gaussian smooth for display only (sigma >= 4 s) |
| **RESP / Airflow** | Respiration belt or airflow sensor | TBD | TBD | Not yet implemented |

SpO2-Pulse is NOT a raw waveform. It is hardware-computed BPM. **Do not filter it.** Gaussian smoothing is for display only.

---

## Subjects and known data issues

~24 subjects total. Subject IDs are alphanumeric (e.g. `AB22`, `AG05`, `AH19`).

| Subject / Session | Issue |
|---|---|
| ES29 eve | Corrupt MFF (missing `epochs.xml`) — skipped every run |
| MG14 mor | Corrupt MFF (XML parse error) — skipped every run |
| ML28 eve | macOS `._` hidden file picked up instead of real MFF — skipped every run |
| AH19 eve | SpO2 probe off entire session — 30/30 trials rejected |
| LO21 | MH20's MFF files are inside LO21's drive folder — LO21 loads MH20 data |

These are **data issues, not code bugs**. Graceful `try/except` around MFF loading is the correct fix. No subject ID hard-coding.

**Per-subject EMG channel overrides:**
```python
SUBJECT_CHANNEL_OVERRIDES = {"NB03": "E241", "AS09": "E241", "MH20": "E241", "LO21": "E241"}
```
Default EMG channel is `E238`. SpO2 uses `"SpO2-Pulse"` for all subjects.

---

## STAI-T scores

Trait anxiety scores live at:
```
/Volumes/My Passport/startle_raw/subjects.xlsx
```
Columns: `ID` (subject identifier) and `STAI-T` (trait anxiety score). Used in correlation plots.

---

## What this project is NOT

- No edge-case data: the subjects we have are the subjects we have.
- No filtering of SpO2-Pulse: it is already BPM.
- No real-time processing: all pipelines are offline batch analyses on MFF files.
- No statistical inference beyond Wilcoxon / paired t-test on group boxplots.
- Plotting and visualisation is the primary output.

---

## Color conventions (consistent across all pipelines)

```python
NEG_COLOR  = "#C0392B"   # deep red   Negative trials
NEU_COLOR  = "#2980B9"   # deep blue  Neutral trials
REJ_COLOR  = "#95A5A6"   # grey       Rejected trials
EVE_COLOR  = "#E67E22"   # orange     Evening session
MOR_COLOR  = "#8E44AD"   # purple     Morning session
```
