from typing import List

SEED: int = 30
BASE_URL: str = "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5_w_data/"
META_DATA: List[str] = ["RGIId", "POINT_ID", "ID", "N_MONTHS", "MONTHS"]
TEST_SIZE: float = 0.3
N_SPLITS: int = 5
NUM_JOBS: int = -1


def add_column(column_name):
    global META_DATA
    if column_name not in META_DATA:
        META_DATA.append(column_name)


def remove_column(column_name):
    global META_DATA
    if column_name in META_DATA:
        META_DATA.remove(column_name)

