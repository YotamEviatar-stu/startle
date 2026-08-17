RAW_DATA_DIR = r"/Volumes/My Passport/startle_raw"
OUTPUT_DIR   = r"/Users/yotameviatar/Desktop/untitled folder/airflow_output"

SUBJECT_FILTER = []

# Unresolved placeholder, not an established rejection rule -- chosen by eyeballing
# one outlier scan and applied inconsistently across similarly-contaminated sessions.
SUBJECTS_EXCLUDE = ["DA01", "ES29/eve", "EV15/eve", "LG07/mor", "MG14/mor", "MS13/eve",
                    "NB03", "YL26"]

MANUAL_BAD_SPANS = {"EV15/mor": [(253.001, 621.416)]}

AIRFLOW_CHANNEL = "Airflow"

SESSION_MAP = {
    "mor": {"label": "Morning", "csv_suffix": "_2"},
    "eve": {"label": "Evening", "csv_suffix": "_1"},
}

CACHE_SFREQ = 25.0

AIRFLOW_LOWPASS  = 70.0

RSP_CLEAN_METHOD         = "khodadad2018"
RSP_PEAK_METHOD_CLEANING = "khodadad2018"

PERFORM_SCORING = True
SCORING_METHOD  = "glm_deconvolution"   # "glm_deconvolution" | "peak_excursion_normalized"

USE_SUBJECTIVE_TRIAL_TYPE = True


AIRFLOW_Z_SCORE_THRESHOLD = 3.0
AIRFLOW_MIN_STD_RATIO = 0.1
RSP_RATE_ARTIFACT_THRESHOLD = 40
AIRFLOW_SCORE_MAX = 3.0
AIRFLOW_POST_MIN_STD_RATIO = 0.1

AIRFLOW_SHAPE_REJECTION_ENABLE = True
AIRFLOW_SHAPE_SD_THRESHOLD     = 3.0
AIRFLOW_SHAPE_WINDOW_MIN       = -5.0
AIRFLOW_SHAPE_WINDOW_MAX       = 10.0


GLM_DESPIKE_ENABLED     = False
GLM_DESPIKE_K           = 50.0
GLM_DESPIKE_MAX_RUN_SEC = 1.0

GLM_TARGET_SFREQ = 10.0
GLM_ZSCORE_RAW   = True

GLM_CYCLE_BANDPASS = (0.01, 0.6)
GLM_MEDIAN_WIN_SEC = 1.0
GLM_REFRACTORY_SEC = 1.0

GLM_FINAL_HP = {"RP": 0.01, "RA": 0.001, "RFR": 0.001}
GLM_FINAL_LP = 1.0

GLM_RF_PARAMS = {
    "RP":  (4.20, 1.65),
    "RA":  (8.07, 3.74),
    "RFR": (6.00, 3.23),
}
GLM_USE_DERIVATIVE = {"RP": False, "RA": True, "RFR": True}

GLM_PRE_FIXATION_SEC = 15.0
GLM_POST_CODE_SEC    = 15.0

ANAL_PRE_BASELINE_SEC = 60.0


QC_ENABLED            = False
QC_BREATH_BAND        = (0.01, 0.6)

CYCLE_REJECTION_ENABLED = True

CYCLE_LOGRA_SCALE = 0.2378
CYCLE_LOGRA_K     = 6.0

CYCLE_RP_MAX = 7.0

CYCLE_LOSTLOCK_MIN_PEAKS = 3

CYCLE_MIN_REF_CYCLES = 30

CYCLE_RECOVERY_RULE = True
CYCLE_GAPFILL_RULE  = True

TRIAL_MIN_VALID_FRAC = 0.60


GLM_PRIMARY_METRIC = "RA"   # "RA" | "RFR" | "RP" | "composite"


GLM_ESTIMATION = "pooled_session"   # "per_trial" | "pooled_session"

GLM_CONDITION_FIELD = "none"   # "none" (one beta/session) | "label" (Negative/Neutral) | "has_sound" (Sound/No-sound)

SUBJECTS_XLSX = r"/Volumes/My Passport/startle_raw/subjects.xlsx"

FORCE_RELOAD = False

VERBOSE  = True
PLOT_DPI = 150

NEG_COLOR  = "#C0392B"
NEU_COLOR  = "#2980B9"
REJ_COLOR  = "#00000000"
EVE_COLOR  = "#E67E22"
MOR_COLOR  = "#8E44AD"
UP_COLOR   = "#27AE60"
DOWN_COLOR = "#E74C3C"

GLM_ZSCORE_METRIC = {"RP": False, "RA": True, "RFR": True}
