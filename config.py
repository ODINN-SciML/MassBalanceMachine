from typing import List

SEED: int = 30
BASE_URL: str = "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5_w_data/"

TEST_SIZE: float = 0.3
N_SPLITS: int = 5

# Default:
META_DATA: List[str] = [
    "RGIId", "POINT_ID", "ID", "N_MONTHS", "MONTHS"
]
NOT_METADATA_NOT_FEATURES: List[str] = [
    "POINT_BALANCE", "YEAR", "POINT_LAT", "POINT_LON", "ALTITUDE_CLIMATE"
]
NUM_JOBS: int = -1
LOSS:str = 'MSE'

# # For CH
# META_DATA: List[str] = [
#     "RGIId", "POINT_ID", "ID", "N_MONTHS", "MONTHS", "PERIOD", "GLACIER"
# ]
# NOT_METADATA_NOT_FEATURES: List[str] = [
#     "POINT_BALANCE", "YEAR", "POINT_LAT", "POINT_LON"
# ]
# NUM_JOBS: int = 25 
# LOSS:str = 'MSE'

def add_column(column_name):
    global META_DATA
    if column_name not in META_DATA:
        META_DATA.append(column_name)

def remove_column(column_name):
    global META_DATA
    if column_name in META_DATA:
        META_DATA.remove(column_name)

month_abbr_hydr = {
    'sep': 1,
    'oct': 2,
    'nov': 3,
    'dec': 4,
    'jan': 5,
    'feb': 6,
    'mar': 7,
    'apr': 8,
    'may': 9,
    'jun': 10,
    'jul': 11,
    'aug': 12,
    #'sep_': 13,
}