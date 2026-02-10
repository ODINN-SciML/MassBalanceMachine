import pandas as pd

from regions.Iceland_mb.scripts.config_ICE import *


def get_stakes_data_ICE(cfg):
    data_ICE = pd.read_csv(
        cfg.dataPath + path_PMB_WGMS_csv + "ICE_wgms_dataset_all.csv"
    )

    # Remove summer season if exists
    data_ICE = data_ICE[data_ICE.PERIOD != "summer"]

    # Drop unused columns
    # cols_to_drop_NOR = ['DATA_MODIFICATION', 'GLACIER_ZONE']
    # data_ICE = data_ICE.drop(columns=cols_to_drop_NOR)

    # Rename GLACIER column using NOR_gl_name dictionary
    # data_ICE["GLACIER"] = data_ICE["GLACIER"].map(NOR_gl_name)
    return data_ICE
