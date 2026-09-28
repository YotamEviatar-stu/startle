RAW_DATA_DIR = r"/Volumes/My Passport/startle_raw"
TRIGGER_CACHE = r"/Users/yotameviatar/airflow_output/_cache/airflow_cache.pkl"

SESSION_MAP = {
    "mor": {"label": "Morning", "csv_suffix": "_2"},
    "eve": {"label": "Evening", "csv_suffix": "_1"},
}

SUBJECTS_EXCLUDE = ["DA01", "ES29/eve", "ES32/mor", "EV15", "LG07/mor", "MG14/mor", "MS13/eve",
                    "NB03", "OA30/mor", "YL26/mor", "YR08/mor"]

MANUAL_BAD_SPANS = {"EV15/mor": [(253.001, 621.416)],
                    "AB22/mor": [(867.0, 873.4)]}

MANUAL_TRIAL_REJECT = {"YR08/eve": {9}, "AG05/mor": {11}, "AH19/eve": {10}, "OA30/eve": {4}}

MANUAL_TRIAL_ACCEPT = {"AG05/mor": {26}}

MANUAL_BASELINE_SPANS = {
    "SH25/eve": {"pre": (832.341, 892.561), "post": (1581.73, 1641.94)},
    "YR08/eve": {"pre": (319.78, 380.0), "post": None},
    "AH19/mor": {"post": None},
    "ER23/eve": {"pre": (480.0, 580.0), "post": None},
    "ER23/mor": {"pre": (110.0, 180.0), "post": (900.0, 940.0)},
    "ES29/mor": {"pre": (325.0, 400.0)},
    "LG07/eve": {"pre": (120.0, 200.0), "post": (869.536, 940.0)},
    "LO21/mor": {"pre": (150.0, 205.0), "post": (945.184, 1020.0)},
    "MG14/eve": {"pre": (59.0, 120.0)},
    "ML28/mor": {"post": (1020.0, 1078.968)},
    "MS18/eve": {"pre": (150.0, 220.0), "post": (861.342, 930.0)},
    "YL26/eve": {"pre": (340.731, 420.0), "post": (1087.311, 1137.0)},
}
