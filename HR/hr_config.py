"""
HR Pipeline Configuration
===================================
All tunable parameters. Edit here — do not touch processor or main.

Unlike the ECG pipeline, there is no cache-invalidation distinction:
HR (SpO2-Pulse channel) needs no filtering, so every parameter
can be changed freely without reprocessing the MFF files.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DATA_DIR = r"/Volumes/My Passport/startle_raw"
OUTPUT_DIR   = r"/Users/yotameviatar/Desktop/spo2_output"

# ── Subject filter ────────────────────────────────────────────────────────────
SUBJECT_FILTER = []   # [] = run all subjects

# ── Channel ───────────────────────────────────────────────────────────────────
# SpO2-Pulse is the pulse oximeter's processed pulse rate output in BPM.
# It is NOT a raw PPG waveform — filtering would destroy the BPM values.
HR_CHANNEL = "SpO2-Pulse"

# ── Session mapping ───────────────────────────────────────────────────────────
SESSION_MAP = {
    "mor": {"label": "Morning", "csv_suffix": "_2"},
    "eve": {"label": "Evening", "csv_suffix": "_1"},
}

# ── DIN trigger codes ─────────────────────────────────────────────────────────
TRIGGER_SESSION_START = 101   # D101: paradigm start; recording cropped here
TRIGGER_SESSION_END   = 124   # D124: paradigm end
TRIGGER_STARTLE       = 110   # D110: startle probe — t=0 for all epochs

# ── Epoch windows (seconds, relative to D110) ─────────────────────────────────
WIDE_TMIN = -10.0   # full epoch cut window — stored in cache
WIDE_TMAX =  15.0
ANAL_TMIN =  -5.0   # analysis / plot window (subset of WIDE) — change freely
ANAL_TMAX =  10.0

# ── Baseline ──────────────────────────────────────────────────────────────────
# Moving baseline: each trial uses its own pre-trigger window, so slow HR
# drift across the session is automatically corrected per event.
BASELINE_TMIN   = -5.0   # s — start of baseline window
BASELINE_TMAX   =  0.0   # s — end of baseline window (= trigger)
BASELINE_METHOD = "mean"  # "mean" or "median"

# ── Scoring ───────────────────────────────────────────────────────────────────
SCORE_METHOD = "max_minus_baseline"   # peak HR in window minus baseline mean
SCORE_TMIN   =  0.0   # s — post-startle window for peak HR search
SCORE_TMAX   =  6.0   # s

# ── Rejection ─────────────────────────────────────────────────────────────────
HR_MIN_BPM = 30    # reject if baseline mean below this (artifact / probe off)
HR_MAX_BPM = 200   # reject if baseline mean above this (artifact)
# Z-score on baseline variability: if pulse rate jumps around in the baseline
# window (high std relative to other trials), the probe was likely unstable.
HR_Z_SCORE_THRESHOLD = 3.0   # session-normalised std of baseline; None = disabled
SCORE_MAX_PCT = 50.0         # reject trial if |score| exceeds this % change; None = disabled

# ── Trial classification ──────────────────────────────────────────────────────
USE_SUBJECTIVE_TRIAL_TYPE = False
# True  = label by ratings (arousal >= 7 or valence <= 3 -> Negative)
# False = label by image_type column in CSV

# ── Cache ─────────────────────────────────────────────────────────────────────
FORCE_RELOAD = False   # True = ignore pickle, re-load all MFF files

# ── Output ────────────────────────────────────────────────────────────────────
VERBOSE  = True
PLOT_DPI = 150

# ── External data ─────────────────────────────────────────────────────────────
SUBJECTS_XLSX = r"/Volumes/My Passport/startle_raw/subjects.xlsx"

# ── Colors ────────────────────────────────────────────────────────────────────
NEG_COLOR  = "#C0392B"   # deep red   — Negative
NEU_COLOR  = "#2980B9"   # deep blue  — Neutral
REJ_COLOR  = "#95A5A6"   # grey       — Rejected
EVE_COLOR  = "#E67E22"   # orange     — Evening
MOR_COLOR  = "#8E44AD"   # purple     — Morning
UP_COLOR   = "#27AE60"   # green      — score increased
DOWN_COLOR = "#E74C3C"   # red        — score decreased
