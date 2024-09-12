from typing import List

SEED: int = 30
BASE_URL: str = "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5_w_data/"
# For CH
META_DATA: List[str] = [
    "RGIId", "POINT_ID", "ID", "N_MONTHS", "MONTHS", "PERIOD", "GLACIER"
]
NOT_METADATA_NOT_FEATURES: List[str] = [
    "POINT_BALANCE", "YEAR", "POINT_LAT", "POINT_LON", 'ALTITUDE_CLIMATE', "POINT_ELEVATION"
]
# For Iceland
# META_DATA: List[str] = [
#     "RGIId", "POINT_ID", "ID", "N_MONTHS", "MONTHS"
# ]
# NOT_METADATA_NOT_FEATURES: List[str] = [
#     "POINT_BALANCE", "YEAR", "POINT_LAT", "POINT_LON", 'ALTITUDE_CLIMATE'
# ]


TEST_SIZE: float = 0.3
N_SPLITS: int = 5
#NUM_JOBS: int = -1
NUM_JOBS: int = 28 # For CH
LOSS: str = 'RMSE' # For now only allows RMSE and MSE


def add_column(column_name):
    global META_DATA
    if column_name not in META_DATA:
        META_DATA.append(column_name)

def remove_column(column_name):
    global META_DATA
    if column_name in META_DATA:
        META_DATA.remove(column_name)
