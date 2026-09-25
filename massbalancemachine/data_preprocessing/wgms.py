import os, sys
import glob
import pandas as pd
import geopandas as gpd

from data_processing.Dataset import Dataset
from data_processing.utils import get_rgi
from data_processing.wgms import wgms_folder
from data_processing.glacier_utils import get_region_name

processed_stakes_folder = os.path.join(wgms_folder, "processed")


def processed_features_stakes_path(rgi_region):
    if rgi_region is None:
        return os.path.join(processed_stakes_folder, "all.csv")
    else:
        assert isinstance(rgi_region, int)
        return os.path.join(processed_stakes_folder, f"region_{rgi_region}.csv")


def build_monthly_data(data, cfg, rgi_region=None):

    assert (
        rgi_region is not None
    ), "For the moment only one single RGI region can be used at a time with the WGMS data."

    data = get_rgi(data=data, region=rgi_region)

    # Drop measurements where no RGIId was found
    data = data[data.RGIId.notnull()]

    # Filter out measurements with NaN elevation
    data = data[data.POINT_ELEVATION.notnull()]

    # Drop measurements with unrealistic SMB
    data = data[(data.POINT_BALANCE > -200) & (data.POINT_BALANCE < 200)]

    # Drop measurements with inconsistent time period
    df_check = data.copy()

    # Convert dates to datetime objects
    df_check["FROM_DATE_DT"] = pd.to_datetime(df_check["FROM_DATE"], format="%Y%m%d")
    df_check["TO_DATE_DT"] = pd.to_datetime(df_check["TO_DATE"], format="%Y%m%d")

    # Calculate month difference
    df_check["MONTH_DIFF"] = (
        (df_check["TO_DATE_DT"].dt.year - df_check["FROM_DATE_DT"].dt.year) * 12
        + df_check["TO_DATE_DT"].dt.month
        - df_check["FROM_DATE_DT"].dt.month
    )

    # Filter out measurements that cannot be represented with padding (more than 12 padded months to add)
    data = df_check[df_check.MONTH_DIFF <= 24]

    # Filter out measurements with period too small
    data = data[data.MONTH_DIFF > 0]
    data = data[(data.TO_DATE_DT - data.FROM_DATE_DT).dt.days >= 30]

    # Filter out specific measurements with a time window that is too large and that prevent from correctly computing the padding
    # RGI60-11
    data = data[~((data.RGIId == "RGI60-11.01450") & (data.MONTH_DIFF > 15))]  # Aletsch
    data = data[
        ~((data.RGIId == "RGI60-11.01509") & (data.MONTH_DIFF == 24))
    ]  # Oberaar
    data = data[~((data.RGIId == "RGI60-11.00638") & (data.MONTH_DIFF == 19))]  # Pizol
    # RGI60-01
    data = data[
        ~((data.RGIId == "RGI60-01.12645") & (data.MONTH_DIFF == 24))
    ]  # East Yakutat
    data = data[
        ~((data.RGIId == "RGI60-01.23646") & (data.MONTH_DIFF == 24))
    ]  # West Yakutat

    # Filter out or correct measurements based on the MB values

    # Basodino: likely a decimal shift for the whole glacier, dividing by 10 gives a monotonic profile indistinguishable from 2017 and 2020
    mask_sel = (data.RGIId == "RGI60-11.01987") & (data.YEAR == 2018)
    data.loc[mask_sel, "POINT_BALANCE"] = data[mask_sel].POINT_BALANCE / 10

    # Stakes in the upper part of the glacier with a 45-year record and moderate accumulation in average but outliers in 2010
    data = data[
        ~(
            (data.RGIId == "RGI60-11.00897")
            & (data.YEAR == 2010)
            & (
                (data.WGMS_ID == "HJ")
                | (data.WGMS_ID == "SJ")
                | (data.WGMS_ID == "WJ")
                | (data.WGMS_ID == "BE")
                | (data.WGMS_ID == "HOI")
                | (data.WGMS_ID == "TE")
            )
        )
    ]  # Hintereisferner

    # Extreme accumulation but another stake at the same elevation gives a negative MB
    data = data[
        ~(
            (data.RGIId == "RGI60-11.00897")
            & (data.YEAR == 2008)
            & (data.WGMS_ID == "HE")
        )
    ]  # Hintereisferner

    # Malavalle: Inconsistent value for this stake across the years, and nearby stakes give a value 10 times smaller
    mask_sel = (
        (data.RGIId == "RGI60-11.00597") & (data.YEAR == 2023) & (data.WGMS_ID == "P21")
    )
    data.loc[mask_sel, "POINT_BALANCE"] = data[mask_sel].POINT_BALANCE / 10

    # Pendente: Sign flip
    mask_sel = (
        (data.RGIId == "RGI60-11.00603") & (data.YEAR == 2006) & (data.WGMS_ID == "P48")
    )
    data.loc[mask_sel, "POINT_BALANCE"] = -data[mask_sel].POINT_BALANCE

    # Plaine Morte: Annual measurements miss winter+summer by ~2m w.e., while the four other stakes of 2011 close exactly, probably a multi-year read
    data = data[
        ~(
            (data.RGIId == "RGI60-11.02072")
            & (data.YEAR == 2011)
            & (
                (data.WGMS_ID == "plm1-09")
                | (data.WGMS_ID == "plm2-09")
                | (data.WGMS_ID == "plm3-10")
                | (data.WGMS_ID == "plm4-10")
            )
        )
    ]

    # Correct metadata

    # Measurements labelled as annual but should be winter
    mask_sel = (
        (data.RGIId == "RGI60-11.03166")
        & ((data.YEAR == 2024) | (data.YEAR == 2025))
        & (data.PERIOD == "annual")
        & (data.MONTH_DIFF == 8)
    )
    data.loc[mask_sel, "PERIOD"] = "winter"  # Grand Etret

    # End window is inconsistent given the stake value, and the nearby stakes of the same year
    mask_sel = (
        (data.RGIId == "RGI60-11.00804")
        & (data.YEAR == 2025)
        & (data.WGMS_ID == "2408")
        & (data.PERIOD == "annual")
    )
    data.loc[mask_sel, "TO_DATE_DT"] = "2025-09-19"
    data.loc[mask_sel, "TO_DATE"] = "20250919"
    data.loc[mask_sel, "MONTH_DIFF"] = 12  # Silvretta

    # End window is inconsistent given the stake value, and the nearby stakes of the same year
    mask_sel = (data.RGIId == "RGI60-11.00006") & (data.YEAR == 2024)
    data.loc[mask_sel, "PERIOD"] = "summer"  # Hallstätter

    # Start date is a year early
    mask_sel = (
        (data.RGIId == "RGI60-11.01450")
        & (data.WGMS_ID == "al-1024")
        & (data.FROM_DATE == "20231115")
    )
    data.loc[mask_sel, "FROM_DATE_DT"] = "2024-11-15"
    data.loc[mask_sel, "FROM_DATE"] = "20241115"  # Aletsch

    # Elevation is too low
    mask_sel = (
        (data.RGIId == "RGI60-11.00597")
        & (data.WGMS_ID == "P04")
        & (data.POINT_ELEVATION == 1812)
    )
    data.loc[mask_sel, "POINT_ELEVATION"] = (
        data[mask_sel].POINT_ELEVATION + 1000
    )  # Malavalle

    # Elevation is too low
    mask_sel = (data.RGIId == "RGI60-11.00833") & (data.YEAR == 1960)
    data.loc[mask_sel, "POINT_ELEVATION"] = (
        data[mask_sel].POINT_ELEVATION + 500
    )  # Silvretta

    # Elevation is too low
    mask_sel = (
        (data.RGIId == "RGI60-11.00647")
        & (data.WGMS_ID == "22/10")
        & (data.YEAR == 2012)
    )
    data.loc[mask_sel, "POINT_ELEVATION"] = 2828  # Ries Occidentale

    # Coordinates correspond to another glacier
    mask_sel = (
        (data.RGIId == "RGI60-11.02674")
        & (data.YEAR == 2023)
        & (data.WGMS_ID == "s2-001")
    )
    data = data[~mask_sel]

    # Discard points before 1950 since ERA5 Land does not cover this period
    data = data[data.YEAR > 1950]

    # User does not need this
    data = data.drop(columns=["WGMS_ID"])

    region_name = get_region_name(rgi_region)
    dataset = Dataset(cfg, data=data, region_name=region_name, region_id=rgi_region)

    voi_topographical = ["aspect", "slope", "svf"]

    # Retrieve the topographical features for each stake measurement based on the latitude and longitude of the stake and add them to the dataset
    dataset.get_topo_features(vois=voi_topographical)

    df = dataset.data
    df["MONTH_START"] = [str(date)[4:6] for date in df.FROM_DATE]
    df["MONTH_END"] = [str(date)[4:6] for date in df.TO_DATE]
    # df.MONTH_START.unique(), df.MONTH_END.unique()

    dataset.get_climate_features()

    # Specify the short names of the climate variables available in the dataset
    vois_climate = [
        "t2m",
        "tp",
        "slhf",
        "sshf",
        "ssrd",
        "fal",
        "str",
        "u10",
        "v10",
        "tp_sum",
        "slhf_sum",
        "sshf_sum",
        "ssrd_sum",
        "str_sum",
    ]

    # For each record, convert to a monthly time resolution
    dataset.convert_to_monthly(
        vois_climate=vois_climate, vois_topographical=voi_topographical
    )

    os.makedirs(processed_stakes_folder, exist_ok=True)
    dataset.data.to_csv(processed_features_stakes_path(rgi_region), index=False)
