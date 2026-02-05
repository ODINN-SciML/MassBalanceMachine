import pandas as pd

from regions.French_Alps.scripts.config_FR import *


def get_stakes_data_FR(cfg):
    data_FR = pd.read_csv(
        cfg.dataPath + path_PMB_GLACIOCLIM_csv + "FR_wgms_dataset_all.csv"
    )

    # Remove summer season if exists
    data_FR = data_FR[data_FR.PERIOD != "summer"]

    # Drop unused columns
    # cols_to_drop_FR = ['DATA_MODIFICATION', 'GLACIER_ZONE']
    # data_FR = data_FR.drop(columns=cols_to_drop_FR)

    # Rename GLACIER column using FR_gl_name dictionary
    data_FR["GLACIER"] = data_FR["GLACIER"].map(FR_gl_name)
    return data_FR
