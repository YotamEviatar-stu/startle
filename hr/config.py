import os

RAW_DATA_DIR = r"F:\startle_raw"
DS_DATA_DIR = r"E:\startle_data"
DS_FS = 250

SESSIONS = ("eve", "mor")

DATA_250_DIR = r"C:\startle_data"

SESSION_FILE_ALIASES = {"DA01/eve": "task", "AK12/eve": "eve2"}

SUBJECT_ID = {"SH25": "ST25"}

SUBJECTS_EXCLUDE = []

MATLAB_BIN = "/Applications/MATLAB_R2026a.app/bin/matlab"
FIELDTRIP_DIR = os.path.expanduser("~/Documents/MATLAB/fieldtrip")
SASICA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SASICA")
HEART_FUNCTIONS_DIR = os.path.expanduser("~/code/tools/heart_functions")
