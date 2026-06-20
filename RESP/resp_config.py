# resp_config.py — all tunable parameters for the RESP pipeline.
#
# CACHE boundary: changing RESP_CHANNELS, WIDE_TMIN, or WIDE_TMAX requires
# setting FORCE_PREPROCESSING = True in resp_main.py to rebuild the MFF cache.
#
# Everything else (filters, baseline, score window, rejection thresholds) is
# applied live on every run — edit here and re-run without touching the MFF.

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DATA_DIR = r"/Volumes/My Passport/startle_raw"
OUTPUT_DIR   = r"/Users/yotameviatar/Desktop/startle_output/RESP"

# ── Channels  [CACHE] ──────────────────────────────────────────────────────────
# Channels extracted from every MFF.  Channels absent in a given file are
# silently skipped for that session.
RESP_CHANNELS = ["Airflow", "Chest_belt", "Abdomen_belt", "Pleth"]

# Channel used for scalar scoring in group boxplots, ratio plots, and STAI
# correlations.  Timecourse plots always show all available channels.
PRIMARY_CHANNEL = "Airflow"

# ── Subject filter ─────────────────────────────────────────────────────────────
# Set to [] to process every subject found in RAW_DATA_DIR.
SUBJECT_FILTER: list = []

# ── Classification ─────────────────────────────────────────────────────────────
# True  = subjective  (arousal >= 7 or valence <= 3  -> Negative)
# False = objective   (from CSV "image_type" column)
USE_SUBJECTIVE_TRIAL_TYPE = False

# ── DIN trigger codes ──────────────────────────────────────────────────────────
TRIGGER_SESSION_START = 101
TRIGGER_SESSION_END   = 124
TRIGGER_STARTLE       = 110

# ── Session map ────────────────────────────────────────────────────────────────
SESSION_MAP = {
    "mor": {"label": "Morning", "csv_suffix": "_2"},
    "eve": {"label": "Evening", "csv_suffix": "_1"},
}

# ── Wide epoch window  [CACHE] ─────────────────────────────────────────────────
# Timing reference: the startle (D110) fires ~3s into a ~6s picture, followed
# by ~6s fixation → the next picture onset is ~9s after D110 (min ~7s).
# Keep WIDE_TMAX ≤ 6s to stay within the current trial's fixation window.
WIDE_TMIN = -5.0    # seconds  — must be ≤ baseline_tmin
WIDE_TMAX =  6.0    # seconds  — safe upper bound before next trial onset

# ── Default analysis parameters (no cache reload needed) ───────────────────────
DEFAULT_PARAMS = {
    # Butterworth bandpass applied to the raw epoch before scoring.
    # Set hp_cutoff or lp_cutoff to None to disable that filter stage.
    "hp_cutoff":    0.05,   # Hz — removes slow baseline drift (periods > 20 s)
    "lp_cutoff":    1.0,    # Hz — removes cardiac / HF noise, isolates respiration
    "filter_order": 2,

    # Updating baseline: mean of the ~4 s before startle onset.
    # At a typical breathing rate of 15 breaths/min (period ≈ 4 s), this window
    # spans roughly one full respiratory cycle, averaging out the oscillation to
    # the subject's current midline — so each trial's baseline tracks where
    # respiration is at THAT moment in the session rather than a fixed reference.
    "baseline_tmin": -4.0,  # seconds  — cover ≥1 breath cycle
    "baseline_tmax":  0.0,

    # Analysis (trimmed) window stored on each trial.
    "anal_tmin": -4.0,
    "anal_tmax":  5.0,

    # Score = mean(score_signal in score window) - baseline_mean.
    # Positive score = amplitude increase; negative = suppression.
    "score_tmin": 0.5,
    "score_tmax": 4.0,

    # Which NeuroKit2-derived signal to use for baseline and scoring.
    #   "clean"     → RSP_Clean  (filtered waveform, same timescale as raw)
    #   "amplitude" → RSP_Amplitude (tidal volume proxy, slowly varying)
    #   "rate"      → RSP_Rate (breaths/min, slowly varying)
    # Visualisation always uses RSP_Clean regardless of this setting.
    "score_signal": "amplitude",

    # Rejection: z-score of trial's baseline std relative to session distribution.
    # High baseline std indicates movement / electrode artifact.
    "z_score_threshold": 3.0,
    # Hard amplitude limit on the filtered baseline epoch (None = disabled).
    # Use native signal units (volts from MNE).
    "absolute_max": None,
}

# ── Per-subject parameter overrides ───────────────────────────────────────────
# Any key present in DEFAULT_PARAMS can be overridden per subject.
# Changes here take effect on the next run with no MFF reload.
# Example:
#   "DA01": {"lp_cutoff": 0.5, "baseline_tmin": -3.0},
#   "NB03": {"z_score_threshold": 2.5, "score_tmax": 4.0},
SUBJECT_PARAMS: dict = {
    # "DA01": {"lp_cutoff": 0.5},
}

# ── Colours ────────────────────────────────────────────────────────────────────
NEG_COLOR  = "#C0392B"
NEU_COLOR  = "#2980B9"
REJ_COLOR  = "#95A5A6"
EVE_COLOR  = "#E67E22"
MOR_COLOR  = "#8E44AD"
UP_COLOR   = "#27AE60"
DOWN_COLOR = "#E74C3C"

CHANNEL_COLORS = {
    "Airflow":      "#2980b9",
    "Chest_belt":   "#27ae60",
    "Abdomen_belt": "#8e44ad",
    "Pleth":        "#d35400",
}

# ── External data ──────────────────────────────────────────────────────────────
SUBJECTS_XLSX = r"/Volumes/My Passport/startle_raw/subjects.xlsx"
