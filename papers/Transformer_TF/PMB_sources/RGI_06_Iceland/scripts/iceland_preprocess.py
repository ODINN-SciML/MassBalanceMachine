import pandas as pd
from shapely.geometry import Point

from papers.Transformer_TF.PMB_sources.RGI_06_Iceland.scripts.config_ICE import *


# --- Preprocess --- #
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
    annual_df = df_stakes[df_stakes["ba_floating_date"].notna()].copy()
    annual_df["FROM_DATE"] = annual_df["d1"]
    annual_df["TO_DATE"] = annual_df["d3"]
    annual_df["POINT_BALANCE"] = annual_df["ba_floating_date"]
    annual_df["PERIOD"] = "annual"
    annual_df["YEAR"] = annual_df["yr"]

    # Create winter measurements dataframe - only where bw is not NaN
    winter_df = df_stakes[df_stakes["bw_floating_date"].notna()].copy()
    winter_df["FROM_DATE"] = winter_df["d1"]
    winter_df["TO_DATE"] = winter_df["d2"]
    winter_df["POINT_BALANCE"] = winter_df["bw_floating_date"]
    winter_df["PERIOD"] = "winter"
    winter_df["YEAR"] = winter_df["yr"]
    """
    # Create summer measurements dataframe - only where bs is not NaN
    summer_df = df_stakes[df_stakes['bs_floating_date'].notna()].copy()
    summer_df['FROM_DATE'] = summer_df['d2']
    summer_df['TO_DATE'] = summer_df['d3']
    summer_df['POINT_BALANCE'] = summer_df['bs_floating_date']
    summer_df['PERIOD'] = 'summer'
    summer_df['YEAR'] = annual_df['yr']
    """
    # Combine both dataframes
    combined_df = pd.concat(
        [annual_df, winter_df], ignore_index=True
    )  # Add ", summer_df" if needed

    # Select only necessary columns
    columns_to_drop = [
        "ba_floating_date",
        "ba_stratigraphic",
        "bs_floating_date",
        "bs_stratigraphic",
        "bw_floating_date",
        "bw_stratigraphic",
        "d1",
        "d2",
        "d3",
        "ds",
        "dw",
        "fall_elevation",
        "ice_melt_fall",
        "ice_melt_spring",
        "snow_melt_fall",
        "rhos",
        "rhow",
        "yr",
        "nswe_fall",
        "swes",
        "swew",
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

    # Calculate month difference
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
