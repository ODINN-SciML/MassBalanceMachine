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
    vois_climate = ["t2m", "tp", "slhf", "sshf", "ssrd", "fal", "str"]

    # For each record, convert to a monthly time resolution
    dataset.convert_to_monthly(
        vois_climate=vois_climate, vois_topographical=voi_topographical
    )

    os.makedirs(processed_stakes_folder, exist_ok=True)
    dataset.data.to_csv(processed_features_stakes_path(rgi_region), index=False)
