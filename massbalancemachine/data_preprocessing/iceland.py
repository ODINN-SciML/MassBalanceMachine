import os, sys
import glob
import pandas as pd
import geopandas as gpd
import oggm.utils

from data_processing.product_utils import mbm_path
from data_processing.utils import get_rgi
from data_retrieval.iceland import download_all_stakes_data
from data_processing.Product import Product
from data_processing.Dataset import Dataset

iceland_path = os.path.join(mbm_path, ".data/stakes/iceland/")
raw_stakes_path = os.path.join(iceland_path, "stakes.csv")
processed_features_stakes_path = os.path.join(iceland_path, "processed.csv")


def raw_data():
    p = Product(raw_stakes_path)
    if not p.is_up_to_date():
        parse_raw_data()
        p.gen_chk()
    data = pd.read_csv(raw_stakes_path)
    return data


def parse_raw_data():
    # download_all_stakes_data()

    all_files = glob.glob(os.path.join(iceland_path, "*.csv"))

    # Initialize empty list to store dataframes
    dfs = []

    # Read each CSV file into a dataframe and append to list
    for file in all_files:
        df = pd.read_csv(file)
        dfs.append(df)

    # Concatenate all dataframes into one
    combined_df = pd.concat(dfs, ignore_index=True)

    # Print info
    print(
        f"Combined {len(all_files)} CSV files into one dataframe with {len(combined_df)} rows"
    )

    # Add data modification column to keep track of mannual changes
    combined_df["DATA_MODIFICATION"] = ""

    # display(combined_df.head(2))

    df_stakes_split = split_stake_measurements(combined_df)

    # Convert date columns to string in 'YYYYMMDD' format
    df_stakes_split["TO_DATE"] = pd.to_datetime(df_stakes_split["TO_DATE"]).dt.strftime(
        "%Y%m%d"
    )
    df_stakes_split["FROM_DATE"] = pd.to_datetime(
        df_stakes_split["FROM_DATE"]
    ).dt.strftime("%Y%m%d")

    # Change NaN year values to the year of the TO_DATE
    df_stakes_split.loc[df_stakes_split["YEAR"].isna(), "YEAR"] = (
        df_stakes_split.loc[df_stakes_split["YEAR"].isna(), "TO_DATE"]
        .astype(str)
        .str[:4]
        .astype(float)
    )
    # Some entries still need a correction because in the line above TO_DATE was NaN
    year_nan_mask = df_stakes_split["YEAR"].isna() & (
        df_stakes_split["PERIOD"] == "summer"
    )
    df_stakes_split.loc[year_nan_mask, "YEAR"] = (
        df_stakes_split.loc[year_nan_mask, "FROM_DATE"]
        .astype(str)
        .str[:4]
        .astype(float)
    )

    # Data modification column update
    date_nan_mask = (
        df_stakes_split["FROM_DATE"].isna() | df_stakes_split["TO_DATE"].isna()
    )
    df_stakes_split.loc[date_nan_mask, "DATA_MODIFICATION"] = (
        "Dates filled in according to hydrological year"
    )
    # Set FROM_DATE from NaN to 01 Oct of previous year
    df_stakes_split.loc[df_stakes_split["FROM_DATE"].isna(), "FROM_DATE"] = (
        df_stakes_split.loc[df_stakes_split["FROM_DATE"].isna(), "YEAR"].astype(int) - 1
    ).astype(str) + "1001"
    # Set TO_DATE from NaN to 30 Sept of the year (as only annual rows have NaN, no need for period distinction)
    df_stakes_split.loc[df_stakes_split["TO_DATE"].isna(), "TO_DATE"] = (
        df_stakes_split.loc[df_stakes_split["TO_DATE"].isna(), "YEAR"]
        .astype(int)
        .astype(str)
        + "0930"
    )
    df_stakes_split.YEAR = df_stakes_split.YEAR.astype(int)

    annual_inconsistent, winter_inconsistent, summer_inconsistent = (
        check_period_consistency(df_stakes_split)
    )

    # Display the inconsistent records
    if len(annual_inconsistent) > 0:
        print("\nInconsistent annual periods:")
        print(annual_inconsistent)

    if len(winter_inconsistent) > 0:
        print("\nInconsistent winter periods:")
        print(winter_inconsistent)

    if len(summer_inconsistent) > 0:
        print("\nInconsistent summer periods:")
        print(summer_inconsistent)

    # Stake `GL10a` is unreasonable (-2), probably wrong FROM_DATE year, change to year - 1
    df_stakes_split.loc[df_stakes_split["stake"] == "GL10a", "FROM_DATE"] = "19960825"
    df_stakes_split.loc[df_stakes_split["stake"] == "GL10a", "DATA_MODIFICATION"] = (
        "FROM_DATE year corrected from 1997 to 1996"
    )
    # Impossible to retrieve FROM_DATE and TO_DATE for stake `BB25`
    df_stakes_split.drop(
        df_stakes_split[df_stakes_split.stake == "BB25"].index, inplace=True
    )

    df_stakes_renamed = df_stakes_split.rename(
        columns={
            "lat": "POINT_LAT",
            "lon": "POINT_LON",
            "elevation": "POINT_ELEVATION",
            "stake": "POINT_ID",
        }
    )

    # NaN check
    # display(df_stakes_renamed[df_stakes_renamed.isna().any(axis=1)])

    # Remove all rows with any NaN values
    df_stakes_renamed = df_stakes_renamed.dropna()

    # Confirm removal - this should show 0 rows if all NaNs were removed
    print(
        f"Rows with NaN values after removal: {len(df_stakes_renamed[df_stakes_renamed.isna().any(axis=1)])}"
    )

    # Add RGI IDs through intersection
    df_stakes_renamed_rgiid = get_rgi(data=df_stakes_renamed, region=6)

    # display(df_stakes_renamed_rgiid[df_stakes_renamed_rgiid['RGIId'].isna()])
    # Remove (nine) stakes without RGIId, as they wont have OGGM data anyways
    df_stakes_renamed_rgiid = df_stakes_renamed_rgiid.dropna(subset=["RGIId"])

    df_stakes_renamed_rgiid.to_csv(raw_stakes_path, index=False)


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
    winter_df["YEAR"] = annual_df["yr"]

    # Create summer measurements dataframe - only where bs is not NaN
    summer_df = df_stakes[df_stakes["bs_floating_date"].notna()].copy()
    summer_df["FROM_DATE"] = summer_df["d2"]
    summer_df["TO_DATE"] = summer_df["d3"]
    summer_df["POINT_BALANCE"] = summer_df["bs_floating_date"]
    summer_df["PERIOD"] = "summer"
    summer_df["YEAR"] = annual_df["yr"]

    # Combine the three dataframes
    combined_df = pd.concat([annual_df, winter_df, summer_df], ignore_index=True)

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
    ## 9-15, 4-9 and 2-8 excludes the normal varying range
    annual_inconsistent = df_check[
        (df_check["PERIOD"] == "annual")
        & ((df_check["MONTH_DIFF"] < 9) | (df_check["MONTH_DIFF"] > 15))
    ]

    winter_inconsistent = df_check[
        (df_check["PERIOD"] == "winter")
        & ((df_check["MONTH_DIFF"] < 4) | (df_check["MONTH_DIFF"] > 9))
    ]

    summer_inconsistent = df_check[
        (df_check["PERIOD"] == "summer")
        & ((df_check["MONTH_DIFF"] < 2) | (df_check["MONTH_DIFF"] > 8))
    ]

    total_annual = len(df_check[df_check["PERIOD"] == "annual"])
    total_winter = len(df_check[df_check["PERIOD"] == "winter"])
    total_summer = len(df_check[df_check["PERIOD"] == "summer"])

    print(
        f"Annual periods: {len(annual_inconsistent)} out of {total_annual} ({len(annual_inconsistent)/total_annual*100:.1f}%) are inconsistent"
    )
    print(
        f"Winter periods: {len(winter_inconsistent)} out of {total_winter} ({len(winter_inconsistent)/total_winter*100:.1f}%) are inconsistent"
    )
    print(
        f"Summer periods: {len(summer_inconsistent)} out of {total_summer} ({len(summer_inconsistent)/total_summer*100:.1f}%) are inconsistent"
    )

    return annual_inconsistent, winter_inconsistent, summer_inconsistent


def build_monthly_data(data, cfg):

    dataset = Dataset(cfg, data=data, region_name="iceland", region_id=6)

    voi_topographical = ["aspect", "slope", "svf"]

    # Retrieve the topographical features for each stake measurement based on the latitude and longitude of the stake and add them to the dataset
    dataset.get_topo_features(vois=voi_topographical)

    df = dataset.data
    df["MONTH_START"] = [str(date)[4:6] for date in df.FROM_DATE]
    df["MONTH_END"] = [str(date)[4:6] for date in df.TO_DATE]
    # df.MONTH_START.unique(), df.MONTH_END.unique()

    dataset.get_climate_features()

    # Specify the short names of the climate variables available in the dataset
    vois_climate = ["t2m", "tp", "slhf", "sshf", "ssrd", "fal", "str"]

    # For each record, convert to a monthly time resolution
    dataset.convert_to_monthly(
        vois_climate=vois_climate, vois_topographical=voi_topographical
    )

    dataset.data.to_csv(processed_features_stakes_path, index=False)
