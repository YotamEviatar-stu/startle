RAW_DATA_DIR = r"/Volumes/My Passport/startle_raw"
OUTPUT_DIR   = r"/Users/yotameviatar/Desktop/untitled folder/airflow_output"

SUBJECT_FILTER = []

# Unresolved placeholder, not an established rejection rule -- chosen by eyeballing
# one outlier scan and applied inconsistently across similarly-contaminated sessions.
SUBJECTS_EXCLUDE = ["DA01", "ES29/eve", "MG14/mor", "MS13/eve", "NB03", "YL26"]

AIRFLOW_CHANNEL = "Airflow"

SESSION_MAP = {
    "mor": {"label": "Morning", "csv_suffix": "_2"},
    "eve": {"label": "Evening", "csv_suffix": "_1"},
}

CACHE_SFREQ = 25.0

AIRFLOW_HIGHPASS = 0.01
AIRFLOW_LOWPASS  = 70.0

RSP_CLEAN_METHOD         = "khodadad2018"
RSP_PEAK_METHOD_CLEANING = "khodadad2018"

ANAL_TMIN, ANAL_TMAX = -5.0, 86.0

PERFORM_SCORING = True
SCORING_METHOD  = "glm_deconvolution"   # "glm_deconvolution" | "peak_excursion_normalized"

USE_SUBJECTIVE_TRIAL_TYPE = True


AIRFLOW_Z_SCORE_THRESHOLD = 3.0
AIRFLOW_AMPLITUDE_Z_THRESHOLD = None
AIRFLOW_MIN_STD_RATIO = 0.1
RSP_RATE_ARTIFACT_THRESHOLD = 40
AIRFLOW_SCORE_MAX = 3.0
AIRFLOW_POST_MIN_STD_RATIO = 0.1

AIRFLOW_SHAPE_REJECTION_ENABLE = True
AIRFLOW_SHAPE_SD_THRESHOLD     = 3.0
AIRFLOW_SHAPE_WINDOW_MIN       = -5.0
AIRFLOW_SHAPE_WINDOW_MAX       = 10.0


GLM_DESPIKE_ENABLED     = True
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

GLM_Z_SCORE_THRESHOLD       = 3.0
GLM_MIN_STD_RATIO           = 0.1
GLM_POST_MIN_STD_RATIO      = 0.1
GLM_RATE_ARTIFACT_THRESHOLD = 40

GLM_MIN_RATE_THRESHOLD = 5
GLM_MIN_CYCLES_IN_WINDOW = 2

GLM_SCORE_MAX = None


GLM_ARTIFACT_METHOD = "manual_exclude"   # "manual_exclude" | "hampel_drop_cycles" | "hampel_reject_trials"
GLM_ARTIFACT_K      = 3.5

GLM_PRE_FIXATION_SEC = 15.0
GLM_POST_CODE_SEC    = 15.0


QC_ENABLED            = True
QC_BREATH_BAND        = (0.01, 0.6)
QC_EXCURSION_K        = 5.0
QC_EXCURSION_WIN_SEC  = 1.0
QC_MERGE_GAP_SEC      = 0.2
QC_MAX_RUN_SEC        = 1.0
QC_FLAT_SESSION_RATIO = 0.10
QC_ROBUST_GATES       = True
QC_MIN_KNOTS          = 5
QC_MIN_VALID_FRACTION = 0.5
QC_MAX_BREATH_RATIO   = 5.0


GLM_PRIMARY_METRIC = "RA"   # "RA" | "RFR" | "RP" | "composite"


GLM_ESTIMATION = "pooled_session"   # "per_trial" | "pooled_session"

GLM_CONDITION_FIELD = "label"   # "label" (Negative/Neutral) | "has_sound" (Sound/No-sound)

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
