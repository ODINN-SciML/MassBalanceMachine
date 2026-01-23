import os, sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../")
)  # Add root of repo to import MBM

import logging
import pandas as pd
import massbalancemachine as mbm
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    train_test_split,
    GroupShuffleSplit,
)
import geopandas as gpd
import xarray as xr
import numpy as np
from tqdm.notebook import tqdm

from regions.Switzerland.scripts.config_CH import *


def correct_for_biggest_grid(df, group_columns, value_column="value"):
    """
    Assign the most frequent value in the specified column to all rows in each group
    if there are more than one unique value in the column within the group.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        group_columns (list): The columns to group by (e.g., YEAR, MONTHS).
        value_column (str): The name of the column to check and replace.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """

    def process_group(group):
        # Check if the column has more than one unique value in the group
        if group[value_column].nunique() > 1:
            # Find the most frequent value
            most_frequent_value = group[value_column].mode()[0]
            # Replace all values with the most frequent value
            group[value_column] = most_frequent_value
        return group

    # Apply the function to each group
    return df.groupby(group_columns).apply(process_group).reset_index(drop=True)


def correct_vars_grid(
    df_grid_monthly, c_prec=1.434, t_off=0.617, temp_grad=-6.5 / 1000, dpdz=1.5 / 10000
):
    # Correct climate grids:
    for voi in [
        "t2m",
        "tp",
        "slhf",
        "sshf",
        "ssrd",
        "fal",
        "str",
        "u10",
        "v10",
        "ALTITUDE_CLIMATE",
    ]:
        df_grid_monthly = correct_for_biggest_grid(
            df_grid_monthly, group_columns=["YEAR", "MONTHS"], value_column=voi
        )

    # New elevation difference with corrected altitude climate (same for all cells of big glacier):
    df_grid_monthly["ELEVATION_DIFFERENCE"] = (
        df_grid_monthly["POINT_ELEVATION"] - df_grid_monthly["ALTITUDE_CLIMATE"]
    )

    # Apply T & P correction
    df_grid_monthly["t2m_corr"] = df_grid_monthly["t2m"] + (
        df_grid_monthly["ELEVATION_DIFFERENCE"] * temp_grad
    )
    df_grid_monthly["tp_corr"] = df_grid_monthly["tp"] * c_prec
    df_grid_monthly["t2m_corr"] += t_off

    # Apply elevation correction factor
    df_grid_monthly["tp_corr"] += df_grid_monthly["tp_corr"] * (
        df_grid_monthly["ELEVATION_DIFFERENCE"] * dpdz
    )

    return df_grid_monthly
