import os, sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)),
                 '../../../'))  # Add root of repo to import MBM

import logging
import pandas as pd
import massbalancemachine as mbm
from sklearn.model_selection import GroupKFold, KFold, train_test_split, GroupShuffleSplit
import geopandas as gpd
import xarray as xr
import numpy as np
from tqdm.notebook import tqdm

from regions.Switzerland.scripts.config_CH import *

def getDfAggregatePred(test_set, y_pred_agg, all_columns):
    # Aggregate predictions to annual or winter:
    df_pred = test_set['df_X'][all_columns].copy()
    df_pred['target'] = test_set['y']
    grouped_ids = df_pred.groupby('ID').agg({
        'target': 'mean',
        'YEAR': 'first',
        'POINT_ID': 'first'
    })
    grouped_ids['pred'] = y_pred_agg
    grouped_ids['PERIOD'] = test_set['df_X'][all_columns].groupby(
        'ID')['PERIOD'].first()
    grouped_ids['GLACIER'] = grouped_ids['POINT_ID'].apply(
        lambda x: x.split('_')[0])

    return grouped_ids


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
    return df.groupby(group_columns).apply(process_group).reset_index(
        drop=True)


def correct_vars_grid(df_grid_monthly,
                      c_prec=1.434,
                      t_off=0.617,
                      temp_grad=-6.5 / 1000,
                      dpdz=1.5 / 10000):
    # Correct climate grids:
    for voi in [
            't2m', 'tp', 'slhf', 'sshf', 'ssrd', 'fal', 'str', 'u10', 'v10',
            'ALTITUDE_CLIMATE'
    ]:
        df_grid_monthly = correct_for_biggest_grid(
            df_grid_monthly,
            group_columns=["YEAR", "MONTHS"],
            value_column=voi)

    # New elevation difference with corrected altitude climate (same for all cells of big glacier):
    df_grid_monthly['ELEVATION_DIFFERENCE'] = df_grid_monthly[
        "POINT_ELEVATION"] - df_grid_monthly["ALTITUDE_CLIMATE"]

    # Apply T & P correction
    df_grid_monthly['t2m_corr'] = df_grid_monthly['t2m'] + (
        df_grid_monthly['ELEVATION_DIFFERENCE'] * temp_grad)
    df_grid_monthly['tp_corr'] = df_grid_monthly['tp'] * c_prec
    df_grid_monthly['t2m_corr'] += t_off

    # Apply elevation correction factor
    df_grid_monthly['tp_corr'] += df_grid_monthly['tp_corr'] * (
        df_grid_monthly['ELEVATION_DIFFERENCE'] * dpdz)

    return df_grid_monthly


def has_geodetic_input(cfg, glacier_name, periods_per_glacier):
    """
    Returns a boolean indicating whether a glacier has valid geodetic data or not.
    For the data to be valid, no year should miss between between the minimum and
    the maximum of all the geodetic periods.
    """

    # Get the minimum and maximum geodetic years for the glacier
    min_geod_y, max_geod_y = np.min(periods_per_glacier[glacier_name]), np.max(
        periods_per_glacier[glacier_name])

    for year in range(min_geod_y, max_geod_y + 1):
        # Check that the glacier grid file exists
        file_name = f"{glacier_name}_grid_{year}.parquet"
        file_path = os.path.join(cfg.dataPath, path_glacier_grid_glamos,
                                 glacier_name, file_name)

        if not os.path.exists(file_path):
            return False
    return True


def create_geodetic_input(cfg,
                          glacier_name,
                          periods_per_glacier,
                          to_seasonal=False):
    """
    Creates a geodetic input array for MBM for a given glacier.

    Parameters:
    - cfg (mbm.Config): Configuration instance used to retrieve the path where to store data on disk.
    - glacier_name (str): Name of the glacier.
    - periods_per_glacier (dict): Dictionary mapping glacier names to geodetic year ranges.

    Returns:
    - pd.DataFrame: Processed geodetic input dataframe.
    """

    # Get the minimum and maximum geodetic years for the glacier
    min_geod_y, max_geod_y = np.min(periods_per_glacier[glacier_name]), np.max(
        periods_per_glacier[glacier_name])

    df_X_geod = pd.DataFrame()
    # Assemble the blocs per year
    for year in range(min_geod_y, max_geod_y + 1):
        # Read the glacier grid file (monthly)
        file_name = f"{glacier_name}_grid_{year}.parquet"
        file_path = os.path.join(cfg.dataPath, path_glacier_grid_glamos,
                                 glacier_name, file_name)

        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found, skipping...")
            continue

        # Load parquet input glacier grid file in monthly format (pre-processed)
        df_grid_monthly = pd.read_parquet(file_path)
        df_grid_monthly.drop_duplicates(inplace=True)

        # Optional: transform to seasonal frequency
        if to_seasonal:
            df_grid_seas = transform_df_to_seasonal(df_grid_monthly)
            df_grid = df_grid_seas
        else:
            df_grid = df_grid_monthly

        # Add GLWD_ID (unique glacier-wide ID corresponding to the year)
        df_grid['GLWD_ID'] = mbm.data_processing.utils.get_hash(
            f"{glacier_name}_{year}")

        # ID is not unique anymore (because of the way the monthly grids were pre-processed),
        # so recompute them:
        if 'ID' in df_grid.columns:
            df_grid['ID'] = df_grid.apply(lambda x: mbm.data_processing.utils.
                                          get_hash(f"{x.ID}_{x.YEAR}"),
                                          axis=1)
        else:
            print(
                f"Warning: 'ID' column missing in {file_name}, skipping ID modification."
            )

        # Append to the final dataframe
        df_X_geod = pd.concat([df_X_geod, df_grid], ignore_index=True)

    return df_X_geod


def transform_df_to_seasonal(data_monthly):
    # Aggregate to seasonal MB:
    months_winter = ['oct', 'nov', 'dec', 'jan', 'feb', 'mar']
    months_summer = ['apr', 'may', 'jun', 'jul', 'aug', 'sep']

    data_monthly['SEASON'] = np.where(
        data_monthly['MONTHS'].isin(months_winter), 'winter', 'summer')

    numerical_cols = [
        'YEAR',
        'POINT_LON',
        'POINT_LAT',
        'POINT_BALANCE',
        'ALTITUDE_CLIMATE',
        'ELEVATION_DIFFERENCE',
        'POINT_ELEVATION',
        'N_MONTHS',
        'aspect_sgi',
        'slope_sgi',
        'hugonnet_dhdt',
        'consensus_ice_thickness',
        'millan_v',
        't2m',
        'tp',
        'slhf',
        'sshf',
        'ssrd',
        'fal',
        'str',
        'u10',
        'v10',
        'pcsr',
    ]

    # All other non-numerical, non-grouping columns are assumed categorical
    exclude_cols = set(numerical_cols + ['ID', 'MONTHS', 'SEASON'])
    categorical_cols = [
        col for col in data_monthly.columns if col not in exclude_cols
    ]

    # Aggregate numerical
    data_seas_num = data_monthly.groupby(
        ['ID', 'SEASON'], as_index=False)[numerical_cols].mean()

    # Get one row per group for categoricals (assumes values per group are consistent)
    data_seas_cat = (data_monthly[['ID', 'SEASON'] +
                                  categorical_cols].drop_duplicates(
                                      subset=['ID', 'SEASON']))

    # Merge numerical + categorical
    data_seas = pd.merge(data_seas_num,
                         data_seas_cat,
                         on=['ID', 'SEASON'],
                         how='inner')

    # Add MONTHS list back in
    season_months = {'winter': months_winter, 'summer': months_summer}
    data_seas['MONTHS'] = data_seas['SEASON'].map(season_months)

    return data_seas
