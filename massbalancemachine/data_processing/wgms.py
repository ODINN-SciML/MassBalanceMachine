import os
import shutil
import urllib.request
import zipfile
import pandas as pd

from data_processing.product_utils import data_path

wgms_zip_file = "DOI-WGMS-FoG-2026-02-10.zip"

wgms_source_data_link = f"https://wgms.ch/downloads/{wgms_zip_file}"
local_path_wgms = f"{data_path}/WGMS/{wgms_zip_file}"

wgms_folder = f"{data_path}/WGMS/{wgms_zip_file.replace('.zip', '')}"


def _clean_extracted_wgms():
    if os.path.isdir(wgms_folder):
        shutil.rmtree(wgms_folder)


def check_and_download_wgms():
    os.makedirs(f"{data_path}/WGMS/", exist_ok=True)
    if not os.path.isdir(wgms_folder):
        if not os.path.isfile(local_path_wgms):
            print("Downloading data from WGMS website")
            urllib.request.urlretrieve(wgms_source_data_link, local_path_wgms)
        print("Unzipping WGMS archive")
        with zipfile.ZipFile(local_path_wgms, "r") as zip_ref:
            zip_ref.extractall(wgms_folder)


def load_wgms_data():
    """
    Load WGMS data and enrich mass balance data with rgi_region.

    Returns:
        pd.DataFrame: mass balance data with added 'rgi_region' column
    """
    check_and_download_wgms()

    point_mb_file = f"{wgms_folder}/data/mass_balance_point.csv"
    glacier_file = f"{wgms_folder}/data/glacier.csv"

    data_mb = pd.read_csv(point_mb_file)
    data_glacier = pd.read_csv(glacier_file)

    # Build mapping: id -> rgi_region (extract number before "_")
    mapping = data_glacier.assign(
        rgi_region=data_glacier["gtng_region"].str.split("_").str[0].astype(int)
    ).set_index("id")["rgi_region"]

    # Apply mapping to data_mb
    data_mb["rgi_region"] = data_mb["glacier_id"].map(mapping)

    return data_mb


def parse_wgms_format(data_mb):
    """
    Converts the WGMS point balance DataFrame to a dataframe ready to be used by MBM Data preparation notebook.

    Args:
        df_pb (pd.DataFrame): dataframe loaded by load_wgms_data "mass_balance_point.csv" from WGMS.
    Returns:
        pd.DataFrame
    """

    new_df = data_mb.drop(
        columns=[
            "country",
            "glacier_name",
            "original_id",
            "glacier_id",
            "time_system",
            "begin_date_unc",
            "end_date_unc",
            "balance_unc",
            "density",
            "density_unc",
            "method",
            "remarks",
        ]
    )
    new_df = new_df.rename(
        columns={
            "id": "ID",
            "year": "YEAR",
            "balance": "POINT_BALANCE",
            "latitude": "POINT_LAT",
            "longitude": "POINT_LON",
            "elevation": "POINT_ELEVATION",
            "begin_date": "FROM_DATE",
            "end_date": "TO_DATE",
            "balance_code": "PERIOD",
        },
    )
    assert new_df.ID.nunique() == new_df.shape[0], "It seems that ID are not unique"

    new_df["FROM_DATE"] = pd.to_datetime(new_df["FROM_DATE"]).dt.strftime("%Y%m%d")
    new_df["TO_DATE"] = pd.to_datetime(new_df["TO_DATE"]).dt.strftime("%Y%m%d")

    return new_df


def filter_dates(df):
    # Remove points for which the dates have a too large uncertainty
    threshold_date_uncertainty = 5
    filtered_df = df[
        (df.end_date_unc <= threshold_date_uncertainty)
        & (df.begin_date_unc <= threshold_date_uncertainty)
    ]

    return filtered_df


def load_processed_wgms(rgi_region=None):
    check_and_download_wgms()
    df = load_wgms_data()
    df = filter_dates(df)
    df = parse_wgms_format(df)
    if rgi_region is not None:
        df = df.loc[df.rgi_region == rgi_region]
    return df
