# airflow_config.py — all tunable parameters for the Airflow pipeline.

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DATA_DIR = r"/Volumes/My Passport/startle_raw"
OUTPUT_DIR   = r"/Users/yotameviatar/Desktop/airflow_output"

# ── Subject filter ─────────────────────────────────────────────────────────────
SUBJECT_FILTER = []

# ── Channel ────────────────────────────────────────────────────────────────────
AIRFLOW_CHANNEL = "Airflow"

# ── DIN trigger codes ──────────────────────────────────────────────────────────
TRIGGER_SESSION_START = 101
TRIGGER_SESSION_END   = 124
TRIGGER_STARTLE       = 110

# ── Session map ────────────────────────────────────────────────────────────────
SESSION_MAP = {
    "mor": {"label": "Morning", "csv_suffix": "_2"},
    "eve": {"label": "Evening", "csv_suffix": "_1"},
}

# ── Cache sampling rate (Layer 1) ─────────────────────────────────────────────
# The raw signal is anti-aliased (Kaiser window) and stored at this rate.
# 25 Hz: Nyquist = 12.5 Hz, well above the 3 Hz respiratory content.
# Changing this requires FORCE_RELOAD. Everything else is Layer 2.
CACHE_SFREQ = 25.0   # Hz

# ── NeuroKit2 cleaning method (Layer 2 — free to change) ──────────────────────
# khodadad2018: Butterworth bandpass 0.05–3 Hz, 2nd order.
# Running on the full session signal avoids filter edge effects.
RSP_CLEAN_METHOD = "khodadad2018"

# ── Epoch windows (Layer 2 — all free to change) ──────────────────────────────
WIDE_TMIN, WIDE_TMAX         = -6.0, 12.0   # epoch cut window applied in Layer 2
BASELINE_TMIN, BASELINE_TMAX = -5.0,  0.0   # pre-stimulus baseline (≥1 full breath cycle)
RESPONSE_TMIN, RESPONSE_TMAX =  0.5,  1.5   # startle gasp: begins at t=0, resolves ~3 s
ANAL_TMIN,     ANAL_TMAX     = -5.0, 10.0   # display/plot window

# ── Scoring ────────────────────────────────────────────────────────────────────
# Score = max absolute excursion from baseline in response window,
#         normalised by session-median RSP_Amplitude (typical breath size).
# Interpretation: 1.0 = response as large as a typical breath; 2.0 = twice as large.
# Stable: denominator is always positive (session-level statistic, never per-trial).
PERFORM_SCORING = True
SCORING_METHOD  = "peak_excursion_normalized"   # see apply_analysis_params()

# ── Classification ─────────────────────────────────────────────────────────────
USE_SUBJECTIVE_TRIAL_TYPE = False   # False = objective image_type column (must match HR and EMG)

# ── Rejection gates ────────────────────────────────────────────────────────────
# 1. Baseline noise (z-score on baseline std across session)
AIRFLOW_Z_SCORE_THRESHOLD     = 2.0   # None = disabled

# 2. Amplitude spike — robust z-score (median/MAD) on max|epoch|
AIRFLOW_AMPLITUDE_Z_THRESHOLD = 3.0   # None = disabled

# 3. Flat signal — baseline std < ratio × session median std
AIRFLOW_MIN_STD_RATIO         = 0.1   # None = disabled

# 4. Physiological RSP rate ceiling — NK2 RSP_Rate anywhere in epoch > this → artefact
#    Normal: 12–20 bpm at rest, ≤40 bpm under stress. 40 bpm = 1.5 s/breath (floor).
#    Anything faster is a movement artefact or cough mis-detected as a breath.
RSP_RATE_ARTIFACT_THRESHOLD   = 40    # bpm; None = disabled

# 5. Score ceiling — reject if normalised score exceeds this
#    Score ≥ 5 means the response was 5× the typical breath amplitude → artefact.
AIRFLOW_SCORE_MAX             = 5.0   # None = disabled

# ── External data ──────────────────────────────────────────────────────────────
SUBJECTS_XLSX = r"/Volumes/My Passport/startle_raw/subjects.xlsx"

# ── Cache ──────────────────────────────────────────────────────────────────────
FORCE_RELOAD = False

# ── Output ─────────────────────────────────────────────────────────────────────
VERBOSE  = True
PLOT_DPI = 150

# ── Colours ────────────────────────────────────────────────────────────────────
NEG_COLOR  = "#C0392B"
NEU_COLOR  = "#2980B9"
REJ_COLOR  = "#95A5A6"
EVE_COLOR  = "#E67E22"
MOR_COLOR  = "#8E44AD"
UP_COLOR   = "#27AE60"
DOWN_COLOR = "#E74C3C"
