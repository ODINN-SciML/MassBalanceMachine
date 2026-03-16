import pandas as pd
import numpy as np
import pyproj
import xarray as xr
import geopandas as gpd
import math
from itertools import combinations
from geopy.distance import geodesic
from shapely.geometry import Point
from tqdm import tqdm
import logging
from oggm import utils, workflow, tasks
from oggm import cfg as oggmCfg
from scipy.spatial.distance import cdist

from scripts.config_NOR import *
from scripts.helpers import *

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# --- Preprocess --- #


def fill_missing_dates(df):
    """
    The first measurement date is missing for some glaciers.
    Fill missing prev_yr_min_date values with October 1st assumed hydr. year
    start date. based on curr_yr_min_date.
    """
    # Create a copy to avoid modifying the original dataframe
    df_filled = df.copy()

    # Find rows with missing prev_yr_min_date
    missing_mask = df_filled["prev_yr_min_date"].isna()

    if missing_mask.sum() > 0:
        print(f"Filling {missing_mask.sum()} missing previous year dates...")

        # For each row with missing prev_yr_min_date
        for idx in df_filled[missing_mask].index:
            if pd.notna(df_filled.loc[idx, "curr_yr_max_date"]):
                # Extract date components
                try:
                    date_str = df_filled.loc[idx, "curr_yr_max_date"]
                    date_obj = pd.to_datetime(date_str, format="%d.%m.%Y")

                    # Get previous year
                    prev_year = date_obj.year - 1

                    # Create October 1st date for previous year
                    new_date = f"01.10.{prev_year}"

                    # Fill the value
                    df_filled.loc[idx, "prev_yr_min_date"] = new_date
                    df_filled.loc[idx, "DATA_MODIFICATION"] = (
                        "Filled missing FROM_DATE with October 1st of previous year"
                    )
                except:
                    print(
                        f"Warning: Could not process date '{date_str}' at index {idx}"
                    )
            else:
                print("no curr_yr_max_date entry to extract the year from")

    return df_filled


def split_stake_measurements(df_stakes):
    """
    Split stake measurements into separate annual and winter records.
    Only includes measurements where the mass balance value is not NaN.

    Args:
        df_stakes: DataFrame with combined stake measurement data

    Returns:
        DataFrame with separate rows for annual and winter measurements
    """

    # Create annual measurements dataframe - only where ba is not NaN
    annual_df = df_stakes[df_stakes["ba"].notna()].copy()
    annual_df["FROM_DATE"] = annual_df["prev_yr_min_date"]
    annual_df["TO_DATE"] = annual_df["curr_yr_min_date"]
    annual_df["POINT_BALANCE"] = annual_df["ba"]
    annual_df["PERIOD"] = "annual"
    annual_df["YEAR"] = pd.to_datetime(
        annual_df["curr_yr_max_date"], format="%d.%m.%Y"
    ).dt.year.astype(int)

    # Create winter measurements dataframe - only where bw is not NaN
    winter_df = df_stakes[df_stakes["bw"].notna()].copy()
    winter_df["FROM_DATE"] = winter_df["prev_yr_min_date"]
    winter_df["TO_DATE"] = winter_df["curr_yr_max_date"]
    winter_df["POINT_BALANCE"] = winter_df["bw"]
    winter_df["PERIOD"] = "winter"
    winter_df["YEAR"] = pd.to_datetime(
        winter_df["curr_yr_max_date"], format="%d.%m.%Y"
    ).dt.year.astype(int)
    """
    # Create summer measurements dataframe - only where bs is not NaN
    summer_df = df_stakes[df_stakes['bs'].notna()].copy()
    summer_df['FROM_DATE'] = summer_df['curr_yr_min_date']
    summer_df['TO_DATE'] = summer_df['curr_yr_max_date']
    summer_df['POINT_BALANCE'] = summer_df['bs']
    summer_df['PERIOD'] = 'summer'
    summer_df['YEAR'] = pd.to_datetime(summer_df['curr_yr_max_date'], format='%d.%m.%Y').dt.year.astype(int)
    """
    # Combine both dataframes
    combined_df = pd.concat(
        [annual_df, winter_df], ignore_index=True
    )  # Add, summer_df if needed

    # Select only necessary columns
    columns_to_drop = [
        "bw",
        "bs",
        "ba",
        "prev_yr_min_date",
        "curr_yr_max_date",
        "curr_yr_min_date",
    ]
    result_df = combined_df.drop(columns=columns_to_drop)

    return result_df


def check_period_consistency(df):
    """
    Checks if date ranges make sense for annual and winter periods:
    Returns dataframes with inconsistent periods
    """
    df_check = df.copy()

    # Convert dates to datetime objects
    df_check["FROM_DATE_DT"] = pd.to_datetime(df_check["FROM_DATE"], format="%Y%m%d")
    df_check["TO_DATE_DT"] = pd.to_datetime(df_check["TO_DATE"], format="%Y%m%d")

    df_check["MONTH_DIFF"] = (
        (df_check["TO_DATE_DT"].dt.year - df_check["FROM_DATE_DT"].dt.year) * 12
        + df_check["TO_DATE_DT"].dt.month
        - df_check["FROM_DATE_DT"].dt.month
    )

    # Identify inconsistent periods
    ## 9-15 and 4-9 excludes the normal varying range
    annual_inconsistent = df_check[
        (df_check["PERIOD"] == "annual")
        & ((df_check["MONTH_DIFF"] < 9) | (df_check["MONTH_DIFF"] > 15))
    ]

    winter_inconsistent = df_check[
        (df_check["PERIOD"] == "winter")
        & ((df_check["MONTH_DIFF"] < 4) | (df_check["MONTH_DIFF"] > 9))
    ]

    total_annual = len(df_check[df_check["PERIOD"] == "annual"])
    total_winter = len(df_check[df_check["PERIOD"] == "winter"])

    print(
        f"Annual periods: {len(annual_inconsistent)} out of {total_annual} ({len(annual_inconsistent)/total_annual*100:.1f}%) are inconsistent"
    )
    print(
        f"Winter periods: {len(winter_inconsistent)} out of {total_winter} ({len(winter_inconsistent)/total_winter*100:.1f}%) are inconsistent"
    )

    return annual_inconsistent, winter_inconsistent


def fix_january_to_october_dates(df, annual_inconsistent, winter_inconsistent):
    """
    For any date in the FROM_DATE or TO_DATE columns with month '01',
    change it to '10' for all inconsistent records
    """
    df_fixed = df.copy()

    # Get all indices to check
    all_indices = list(annual_inconsistent.index) + list(winter_inconsistent.index)

    # Count fixes
    fixes_made = 0

    for idx in all_indices:
        # Check and fix FROM_DATE
        from_date = df_fixed.loc[idx, "FROM_DATE"]
        if isinstance(from_date, str) and from_date[4:6] == "01":  # If month is January
            # Replace '01' with '10'
            df_fixed.loc[idx, "FROM_DATE"] = from_date[0:4] + "10" + from_date[6:]
            df_fixed.loc[idx, "DATA_MODIFICATION"] = (
                "Corrected FROM_DATE from January to October"
            )
            fixes_made += 1

        # Check and fix TO_DATE
        to_date = df_fixed.loc[idx, "TO_DATE"]
        if isinstance(to_date, str) and to_date[4:6] == "01":  # If month is January
            # Replace '01' with '10'
            df_fixed.loc[idx, "TO_DATE"] = to_date[0:4] + "10" + to_date[6:]
            df_fixed.loc[idx, "DATA_MODIFICATION"] = (
                "Corrected TO_DATE from January to October"
            )
            fixes_made += 1

    print(
        f"Made {fixes_made} fixes (01 → 10) across {len(all_indices)} inconsistent records"
    )
    return df_fixed


def flag_elevation_mismatch(df, threshold=400):
    """
    Flag rows where POINT_ELEVATION differs from DEM elevation ('topo')
    by more than a given threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'POINT_ELEVATION' and 'topo'.
    threshold : float, optional
        Maximum allowed absolute elevation difference (meters).
        Default is 400 m.

    Returns
    -------
    df_out : pandas.DataFrame
        Copy of input dataframe with:
        - 'elev_diff' : POINT_ELEVATION - topo
        - 'elev_mismatch' : True if abs(diff) > threshold
    mismatches : pandas.DataFrame
        Subset of rows where mismatch is True.
    """
    df_out = df.copy()

    df_out["elev_diff"] = df_out["POINT_ELEVATION"] - df_out["topo"]
    df_out["elev_mismatch"] = df_out["elev_diff"].abs() > threshold

    mismatches = df_out[df_out["elev_mismatch"]]

    print(
        f"{len(mismatches)} out of {len(df_out)} points "
        f"({len(mismatches)/len(df_out)*100:.2f}%) exceed ±{threshold} m elevation difference."
    )

    return df_out, mismatches
