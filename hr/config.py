import os

RAW_DATA_DIR = r"/Volumes/My Passport/startle_raw"
DS_DATA_DIR = r"E:\startle_data"
DS_FS = 250

SESSIONS = ("eve", "mor")

SUBJECT_ID = {"SH25": "ST25"}

SUBJECTS_EXCLUDE = []

MANUAL_BAD_SPANS = {}

MANUAL_REST_SPANS = {
    "ST25/eve": {"pre": (832.341, 892.561), "post": (1581.73, 1641.94)},
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

MATLAB_BIN = "/Applications/MATLAB_R2026a.app/bin/matlab"
FIELDTRIP_DIR = os.path.expanduser("~/Documents/MATLAB/fieldtrip")
SASICA_DIR = os.path.expanduser("~/code/tools/SASICA")
HEART_FUNCTIONS_DIR = os.path.expanduser("~/code/tools/heart_functions")
