# airflow_config.py — all tunable parameters for the Airflow gasp-ratio pipeline.

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DATA_DIR = r"/Volumes/My Passport/startle_raw"
OUTPUT_DIR   = r"/Users/yotameviatar/Desktop/airflow_output"

# ── Subject filter ─────────────────────────────────────────────────────────────
# Set to [] to process every subject found in RAW_DATA_DIR.
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

# ── Epoch windows (seconds relative to D110) ───────────────────────────────────
WIDE_TMIN, WIDE_TMAX         = -6.0, 15.0   # full extraction window (extra 1 s buffer before baseline)
BASELINE_TMIN, BASELINE_TMAX = -5.0,  0.0   # pre-stimulus baseline
RESPONSE_TMIN, RESPONSE_TMAX =  0.5,  5.0   # gasp detection window
ANAL_TMIN,     ANAL_TMAX     = -5.0, 10.0   # trimmed window stored per trial

# ── Scoring ────────────────────────────────────────────────────────────────────
PERFORM_SCORING = True
SCORING_METHOD  = "gasp_ratio"   # only supported method

# ── Classification ─────────────────────────────────────────────────────────────
# True  = subjective  (arousal >= 7 or valence <= 3 → Negative)
# False = objective   (from CSV "image_type" column)
USE_SUBJECTIVE_TRIAL_TYPE = True

# ── Rejection ──────────────────────────────────────────────────────────────────
AIRFLOW_Z_SCORE_THRESHOLD = 3.0   # session-normalised baseline std z-score; None = disabled
AIRFLOW_SCORE_MAX         = 10.0  # reject trial if gasp_ratio exceeds this; None = disabled

# ── External data ──────────────────────────────────────────────────────────────
SUBJECTS_XLSX = r"/Volumes/My Passport/startle_raw/subjects.xlsx"

# ── Cache ──────────────────────────────────────────────────────────────────────
FORCE_RELOAD = False   # True = ignore pickle, re-load all MFF files

# ── Output ─────────────────────────────────────────────────────────────────────
VERBOSE  = True
PLOT_DPI = 150

# ── Colours ────────────────────────────────────────────────────────────────────
NEG_COLOR  = "#C0392B"   # deep red   — Negative
NEU_COLOR  = "#2980B9"   # deep blue  — Neutral
REJ_COLOR  = "#95A5A6"   # grey       — Rejected
EVE_COLOR  = "#E67E22"   # orange     — Evening
MOR_COLOR  = "#8E44AD"   # purple     — Morning
UP_COLOR   = "#27AE60"   # green      — score increased
DOWN_COLOR = "#E74C3C"   # red        — score decreased
