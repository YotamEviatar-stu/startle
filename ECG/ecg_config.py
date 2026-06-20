"""
ECG Signal Processing Configuration
=====================================
All tunable parameters for the ECG pipeline.
Parameters marked "← change freely" can be updated without reprocessing MFF files.
Parameters marked "← requires FORCE_RELOAD_ECG = True" invalidate the cache.
"""

import os

# ── Data paths ────────────────────────────────────────────────────────────────
RAW_DATA_DIR = r"/Volumes/My Passport/startle_raw"
OUTPUT_DIR   = r"/Users/yotameviatar/Desktop/ecg_output"

# ── Trial type classification ─────────────────────────────────────────────────
USE_SUBJECTIVE_TRIAL_TYPE = False

if USE_SUBJECTIVE_TRIAL_TYPE:
    PLOT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "dynamic_rejection")
else:
    PLOT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "no_subjective")

# ── ECG channel configuration ─────────────────────────────────────────────────
# E256 is the vertex electrode on EGI hardware — strongest cardiac signal.
# The generic "ECG" channel label does not exist on these recordings.
ECG_CHANNEL = "E256"
SUBJECT_CHANNEL_OVERRIDES: dict = {}   # e.g. {"NB03": "E123"}

# ── Session mapping ───────────────────────────────────────────────────────────
SESSION_MAP = {
    "mor": {"label": "Morning", "csv_suffix": "_2"},
    "eve": {"label": "Evening", "csv_suffix": "_1"},
}

# ── DIN trigger codes ─────────────────────────────────────────────────────────
TRIGGER_SESSION_START = 101   # D101: task software signals paradigm start
TRIGGER_SESSION_END   = 124   # D124: paradigm ends; recording cropped to [101, 124]
TRIGGER_STARTLE       = 110   # D110: startle probe delivery; t=0 for all epochs

# ── Signal filtering ──────────────────────────────────────────────────────────
HP_FILTER_ORDER = 3
HP_CUTOFF_HZ    = 0.5    # Hz. Removes DC offset / slow baseline wander while
                          # preserving P/QRS/T morphology for R-peak detection.
                          # ← requires FORCE_RELOAD_ECG = True to change

LP_FILTER_ORDER = 3
LP_CUTOFF_HZ    = 40.0   # Hz. QRS energy is mostly below 40 Hz; removes EMG noise.
                          # No rectification — ECG waveform must be preserved.
                          # ← requires FORCE_RELOAD_ECG = True to change

APPLY_LP_FILTER = True

# ── R-peak detection ──────────────────────────────────────────────────────────
RPEAK_MIN_DISTANCE_MS = 500   # ms. Max physiological HR ~120 bpm = 500 ms/beat.
                               # Prevents T-wave / P-wave from being detected as
                               # a second R-peak within the same heartbeat.
                               # ← requires FORCE_RELOAD_ECG = True to change

RPEAK_PROMINENCE_FRAC = 0.10  # Fraction of signal peak-to-peak amplitude used as
                                # the minimum peak prominence. 10% is enough to
                                # suppress noise while passing every genuine R-peak.
                                # Auto-handles both normal and inverted ECG polarity:
                                # detector tries positive and negative sides and keeps
                                # whichever yields more peaks.
                                # ← requires FORCE_RELOAD_ECG = True to change

# ── Epoch window (wide — stored in cache) ─────────────────────────────────────
WIDE_TMIN = -2.000   # seconds. 2 s pre-trigger = 2-3 heartbeats at resting HR.
                      # ← requires FORCE_RELOAD_ECG = True to change

WIDE_TMAX = +4.000   # seconds. Cardiac defense response unfolds over 4-6 s.
                      # ← requires FORCE_RELOAD_ECG = True to change

# ── Analysis window (not cached — change freely) ──────────────────────────────
ANAL_TMIN = -2.000   # seconds ← change freely
ANAL_TMAX = +4.000   # seconds ← change freely

# ── Baseline HR window ────────────────────────────────────────────────────────
BASELINE_TMIN = -2.000   # seconds — start of baseline = start of epoch
BASELINE_TMAX = -0.500   # seconds — ends 500 ms before startle to avoid anticipatory
                          # autonomic changes. ← change freely

# ── HR change score window ────────────────────────────────────────────────────
SCORE_TMIN = 0.000   # seconds ← change freely
SCORE_TMAX = 4.000   # seconds ← change freely

# ── HR timecourse interpolation ───────────────────────────────────────────────
HR_INTERP_FS = 4.0   # Hz. Matches ecg_session_trend.py; 4 Hz = 250 ms resolution
                      # is more than enough for cardiac dynamics (responses unfold
                      # over seconds). ← change freely

# ── Rejection thresholds (not cached — change freely) ─────────────────────────
HR_MIN_BPM = 30    # bpm. Below this = R-peak detection failure or artifact.
HR_MAX_BPM = 120   # bpm. Above this = likely detecting noise, not R-peaks.

HR_Z_SCORE_THRESHOLD = 3.0   # Reject if baseline HR > 3 SD from session median.

MIN_RPEAKS_BASELINE = 2   # Need ≥2 R-peaks in baseline window to compute ≥1 IBI.
MIN_RPEAKS_EPOCH    = 3   # Need ≥3 R-peaks in full epoch for reliable interpolation.

# ── Output options ────────────────────────────────────────────────────────────
PLOT_DPI         = 150
VERBOSE          = True
FORCE_RELOAD_ECG = True    # Set True after changing any filter or R-peak parameter.

# ── Visualization colors ──────────────────────────────────────────────────────
NEG_COLOR  = "#C0392B"   # deep red   — Negative trials
NEU_COLOR  = "#2980B9"   # deep blue  — Neutral trials
REJ_COLOR  = "#95A5A6"   # grey       — Rejected trials
EVE_COLOR  = "#E67E22"   # orange     — Evening session
MOR_COLOR  = "#8E44AD"   # purple     — Morning session
UP_COLOR   = "#27AE60"   # green      — score increased Eve→Mor
DOWN_COLOR = "#E74C3C"   # red        — score decreased Eve→Mor

# ── STAI-T spreadsheet ────────────────────────────────────────────────────────
SUBJECTS_XLSX = r"/Users/yotameviatar/Desktop/data/subjects.xlsx"
