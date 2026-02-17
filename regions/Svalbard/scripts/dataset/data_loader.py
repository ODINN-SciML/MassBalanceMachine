import pandas as pd

from regions.Svalbard.scripts.config_SVA import *


def get_stakes_data_SVA(cfg):
    data_SVA = pd.read_csv(
        cfg.dataPath + path_PMB_WGMS_csv + "SVA_wgms_dataset_all.csv"
    )

    # Remove summer season if exists
    data_SVA = data_SVA[data_SVA.PERIOD != "summer"]

    # Drop unused columns
    # cols_to_drop_NOR = ['DATA_MODIFICATION', 'GLACIER_ZONE']
    # data_SVA = data_SVA.drop(columns=cols_to_drop_NOR)

    # Rename GLACIER column using NOR_gl_name dictionary
    # data_SVA["GLACIER"] = data_SVA["GLACIER"].map(NOR_gl_name)
    return data_SVA
