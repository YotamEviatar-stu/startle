"""
HR Pipeline Configuration
===================================
All tunable parameters. Edit here — do not touch processor or main.

Two-layer cache boundary:
  Layer 1 (requires FORCE_RELOAD): HR_CHANNEL, HR_CACHE_SFREQ, trigger codes.
  Layer 2 (free to tune anytime):  everything else — WIDE_TMIN/TMAX, BASELINE_*,
    SCORE_*, HR_MIN/MAX_BPM, HR_Z_SCORE_THRESHOLD, FLAT_*, SPIKE_*, SCORE_MAX/MIN_PCT.
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

# ── Cache sampling rate (Layer 1) ─────────────────────────────────────────────
# The raw session signal is downsampled and stored at this rate.
# 10 Hz: Nyquist = 5 Hz, well above the 0.5 Hz HR content.
# Set to None to store at native MFF rate (larger cache, usually unnecessary).
# Changing this requires FORCE_RELOAD. Everything else is Layer 2.
HR_CACHE_SFREQ = 10.0   # Hz

# ── Epoch windows (Layer 2 — all free to change without reload) ───────────────
WIDE_TMIN = -10.0   # full epoch cut window (applied in Layer 2 from raw session)
WIDE_TMAX =  20.0
ANAL_TMIN =  -5.0   # analysis / plot window (subset of WIDE)
ANAL_TMAX =  10.0

# ── Baseline ──────────────────────────────────────────────────────────────────
# Moving baseline: each trial uses its own pre-trigger window, so slow HR
# drift across the session is automatically corrected per event.
BASELINE_TMIN   = -6.0     # s — start of baseline window
BASELINE_TMAX   = -1.0     # s — excludes SpO2 sensor lag (~1s hardware averaging delay)
BASELINE_METHOD = "median"    # "mean" or "median"

# ── Scoring ───────────────────────────────────────────────────────────────────
# max_minus_baseline: peak HR in score window minus baseline (captures brief tachycardia)
# mean_minus_baseline: mean HR in score window minus baseline (less sensitive to spikes)
SCORE_METHOD = "max_minus_baseline"
SCORE_TMIN   =  2.0   # s — post-startle window start
SCORE_TMAX   =  5.0   # s — post-startle window end

# ── Rejection ─────────────────────────────────────────────────────────────────
HR_MIN_BPM = 30    # reject if baseline mean below this (artifact / probe off)
HR_MAX_BPM = 200   # reject if baseline mean above this (artifact)
# Z-score on baseline variability: if pulse rate jumps around in the baseline
# window (high std relative to other trials), the probe was likely unstable.
HR_Z_SCORE_THRESHOLD = 3.0   # session-normalised std of baseline; None = disabled
SCORE_MAX_PCT = 30.0         # reject trial if |score| exceeds this % change; None = disabled
SCORE_MIN_PCT = None         # never reject by direction of effect — data quality only

# ── Artifact cleaning (flat plateaus + spikes) ────────────────────────────────
# SpO2-Pulse updates every ~2 s. Freezes longer than FLAT_MIN_SEC = probe dropout.
# Spikes are brief excursions >> SPIKE_THRESH_BPM above the local 5-s median.
# Both are linearly interpolated before any rejection gate runs.
# Optional lowpass filter on epoch (applied after artifact cleaning, before scoring).
# Smooths the BPM staircase so peak reflects sustained elevation, not a single high step.
# None = no filter (raw staircase); 0.3 Hz is a reasonable starting point.
SIGNAL_LOWPASS_HZ = None   # Hz; set to e.g. 0.3 to enable

FLAT_MIN_SEC      = 12.0    # seconds frozen = dropout (normal blocks ≤ 7s; dropouts = 8s+)
SPIKE_THRESH_BPM  = 20.0   # BPM deviation from local median = impulse artifact

# ── Artifact coverage gates ───────────────────────────────────────────────────
# Applied independently to baseline and score windows after interpolation.
# A window is rejected if EITHER threshold is exceeded:
#   total      — fraction of window that was artifact (many small hits)
#   contiguous — largest single artifact run / window length (one long gap)
FLAT_BASELINE_TOTAL_MAX      = 0.50   # >50% total artifact in baseline → reject
FLAT_BASELINE_CONTIGUOUS_MAX = 0.40   # >40% single gap in baseline   → reject
FLAT_SCORE_TOTAL_MAX         = 0.50   # >50% total artifact in score   → reject
FLAT_SCORE_CONTIGUOUS_MAX    = 0.40   # >40% single gap in score       → reject

# Dominant-flat gate: reject if any single constant-BPM block covers >this
# fraction of the score window, even if shorter than FLAT_MIN_SEC.
# Catches trials where BPM never updated during the response window.
SCORE_FLAT_DOMINANT_MAX      = 0.85   # fraction; None = disabled

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
