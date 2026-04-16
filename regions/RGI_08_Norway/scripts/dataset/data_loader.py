import pandas as pd

from regions.Norway_mb.scripts.config_NOR import *


def get_stakes_data_NOR(cfg):
    data_NOR = pd.read_csv(
        cfg.dataPath + path_PMB_WGMS_csv + "NOR_wgms_dataset_all.csv"
    )

    # Remove summer season if exists
    data_NOR = data_NOR[data_NOR.PERIOD != "summer"]

    # Drop unused columns
    # cols_to_drop_NOR = ['DATA_MODIFICATION', 'GLACIER_ZONE']
    # data_NOR = data_NOR.drop(columns=cols_to_drop_NOR)

    # Rename GLACIER column using NOR_gl_name dictionary
    # data_NOR["GLACIER"] = data_NOR["GLACIER"].map(NOR_gl_name)
    return data_NOR
