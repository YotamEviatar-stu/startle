---
name: cross-pipeline-audit
description: Audits HR and/or Airflow pipeline code against EMG raw as the canonical reference. Flags any conceptual deviation in event logic, trial structure, baseline, scoring, or cache design that would make results incomparable across channels. Invoke whenever HR or Airflow code is written or reviewed.
---

# Cross-Pipeline Consistency Audit

## Purpose

The EMG raw script (`emg_raw_potentiation.py`) is the **canonical reference** for this project.
HR and Airflow record the **same subjects, same sessions, same experimental events** — only the physiological channel differs.

Every design decision that does not depend on the channel itself **must be identical** across all pipelines. This audit enforces that.

---

## Step 1 — Load experiment context

Before reading any code, invoke the `startle-experiment` skill so you have the full experimental briefing (event codes, CSV structure, trial logic, subject list, known data issues). Do not skip this step.

---

## Step 2 — Read the reference and the target

Read these files in full:

**Reference (canonical):**
- `emg_raw_potentiation.py`

**Target — read whichever pipelines are being audited:**
- HR: `HR/hr_config.py`, `HR/hr_processor.py`, `HR/hr_main.py`
- Airflow: `Airflow/airflow_config.py`, `Airflow/airflow_processor.py`, `Airflow/airflow_main.py`

Do not skim. The deviations that matter are often one or two lines buried in helper functions.

---

## Step 3 — Audit checklist

For each item below, compare the reference to the target and mark:
- **PASS** — target matches the reference on this dimension
- **FAIL** — target deviates in a way that breaks experimental comparability
- **WARN** — target differs but the difference may be intentional; flag with a question for the user
- **N/A** — item is not yet implemented in the target (note what is missing)

### A. Session loading and cropping
| # | What to check | Why it matters |
|---|---|---|
| A1 | MFF files found by searching `EEG/` subfolder for `*_eve_*.mff` / `*_mor_*.mff` | Wrong path = wrong recording |
| A2 | Recording cropped to **D101 → D124** (session start / end triggers) | Including pre/post-session signal contaminates every baseline |
| A3 | `try/except` around MFF loading; no subject IDs hard-coded | Known corrupt files (ES29 eve, MG14 mor, ML28 eve) must be skipped gracefully |
| A4 | Session label (`eve` / `mor`) derived from MFF filename keyword, not hardcoded | Filename is the authoritative source |
| A5 | CSV matched by session: `_1.csv` = evening, `_2.csv` = morning | Wrong CSV = wrong ratings for those trials |

### B. Event detection
| # | What to check | Why it matters |
|---|---|---|
| B1 | DIN channels identified by prefix `D` | Other channels must not be scanned for triggers |
| B2 | Threshold = **90 % of channel maximum** per channel | Lower threshold = spurious events; higher = missed events |
| B3 | Contiguous above-threshold samples collapsed to **first sample only** | One trigger per event; duplicates corrupt alignment |
| B4 | D101 / D124 / D110 trigger codes are the same constants | Any drift here breaks every downstream step |
| B5 | Only **has_sound == True** CSV rows are used for trial alignment | No-sound trials are never analysed |
| B6 | Nth D110 event ↔ Nth has_sound==True CSV row (sequential order, no timestamps) | This is the only alignment mechanism; any deviation is fatal |

### C. Epoch cutting
| # | What to check | Why it matters |
|---|---|---|
| C1 | Wide epoch cut around D110: **[-250 ms, +250 ms]** | Must be wide enough to contain baseline + analysis window |
| C2 | Analysis window trimmed to **[-50 ms, +120 ms]** for scoring | Scoring window must be identical to compare magnitudes |
| C3 | Epoch cutting done in `process_session()` / equivalent, **not** in analysis | So baseline/rejection params can change without re-loading |

### D. Baseline and rejection
| # | What to check | Why it matters |
|---|---|---|
| D1 | Baseline = **mean of [-50 ms, 0 ms]** pre-stimulus | Pre-stimulus window is the reference; any other window is not comparable |
| D2 | Rejection criterion 1: baseline **std > 3 SD** (z-score across accepted trials) | Noisy baseline = unreliable % change |
| D3 | Rejection criterion 2: channel-appropriate amplitude hard limit | Prevents physiological artifacts from inflating scores |
| D4 | Rejected trials are **excluded from scoring** but **counted and reported** | Transparency about data loss |

### E. Scoring
| # | What to check | Why it matters |
|---|---|---|
| E1 | Score = **peak value in [20 ms, 100 ms]** minus baseline mean | This is the startle response window; any other window is a different measure |
| E2 | Trial type label applied correctly: `image_type` column OR subjective rule (arousal >= 7 OR valence <= 3) depending on `USE_SUBJECTIVE_TRIAL_TYPE` flag | Wrong labels = wrong condition grouping |
| E3 | Both Negative and Neutral computed; no implicit mixing | Four conditions must remain separable |

### F. Cache design
| # | What to check | Why it matters |
|---|---|---|
| F1 | **Two-layer cache**: `process_session()` writes a pickle (raw epochs); `apply_analysis_params()` runs every time from that pickle | Changing baseline/rejection params must not require re-loading MFF |
| F2 | `FORCE_RELOAD` flag bypasses pickle and re-runs from MFF | Required when filter or epoch window changes |
| F3 | Pickle stores raw (unscored) epochs; scoring happens outside the pickle | Single source of truth for raw data |

### G. Plot suite completeness
The EMG pipeline defines the **target plot suite** every channel should eventually reach. Flag what is missing as N/A (not a FAIL unless the user asks for completeness).

Required plots (check each exists or is planned):
- `plot_individual_trials()` — one PNG per trial; rejected = grey, Neg = red, Neu = blue
- `plot_trial_scores_per_subject()` — score vs trial number scatter
- `plot_group_boxplot_four_conditions()` — Eve-Neg / Eve-Neu / Mor-Neg / Mor-Neu with connecting lines
- `plot_group_ratio_boxplot()` — Neg/Neu ratio per subject, Eve vs Mor, with statistics
- `plot_group_average_timecourse()` — mean ± SEM epoch per condition
- `plot_subject_average_timecourse()` — per-subject mean epoch
- `plot_subject_psd()` — Welch PSD of raw signal
- `plot_group_session_comparison()` — group-level Eve vs Mor
- `plot_group_eve_mor_ratio()` — per-subject Evening / Morning ratio
- `plot_group_overall_eve_vs_mor()` — collapsed Neg+Neu: Eve vs Mor
- `plot_group_overall_ratio()` — overall Neg/Neu ratio
- `plot_group_overall_timecourse()` — grand mean timecourse
- `plot_stai_correlations()` — STAI-T vs score / subjective ratings
- `plot_subjective_negative_percentage()` — % trials rated Negative, Eve vs Mor

### H. Shared constants
| # | What to check |
|---|---|
| H1 | Color constants match exactly: NEG=#C0392B, NEU=#2980B9, REJ=#95A5A6, EVE=#E67E22, MOR=#8E44AD |
| H2 | STAI path = `/Volumes/My Passport/startle_raw/subjects.xlsx`, columns `ID` and `STAI-T` |
| H3 | Subject iteration is driven by folder scan (no hardcoded subject list) |

---

## Step 4 — What is ALLOWED to differ

These differences are **expected and correct**. Do not flag them as failures.

| Dimension | EMG reference | HR | Airflow |
|---|---|---|---|
| Filter type & frequencies | HP 28 Hz + rectify + LP 30 Hz | Channel-appropriate | Channel-appropriate |
| Whether filtering is applied | Yes (raw muscle signal) | Depends on signal type | Depends on signal type |
| Score units | µV above baseline | BPM / % change | % change / L·s⁻¹ |
| Channel name(s) | E238 (E241 for some subjects) | HR channel name | Airflow channel name |
| Per-subject channel overrides | SUBJECT_CHANNEL_OVERRIDES dict | Same pattern, different channels | Same pattern, different channels |
| Channel-specific preprocessing | Rectify (`abs()`) | R-peak detection (ECG only) | None expected |

If a target pipeline has **identical** filter frequencies to EMG (28 Hz / 30 Hz), that is suspicious — flag as WARN.

---

## Step 5 — Output format

Produce a structured report with three sections:

### CRITICAL FAILURES
Items marked FAIL from sections A–F. These break experimental validity.
For each: `[FAIL] <item ID> — <what the target does> vs <what EMG does> — <file:line>`

### WARNINGS
Items marked WARN. These need user confirmation before proceeding.
For each: `[WARN] <item ID> — <what differs> — <question for user>`

### GAPS (N/A)
Features not yet implemented in the target.
For each: `[N/A] <item ID> — <what is missing>`

### SUMMARY
One paragraph: overall assessment, whether HR/Airflow results would be directly comparable to EMG results given current state, and top priority fix if any FAILs exist.

---

## Important reminders

- **Do not flag channel-specific differences as failures.** The filter cutoffs, score units, and channel names will differ — that is correct.
- **Do flag structural differences.** If the baseline window is different, the epoch cutting is different, or the event alignment logic is different, those are CRITICAL failures even if the numbers look reasonable.
- **A pipeline that is partially complete is not a failure.** Missing plots are N/A. A wrong baseline window is a FAIL.
- **Assume the user has not read the EMG script recently.** Quote the relevant EMG line(s) when you report a deviation so they can see exactly what the reference does.
