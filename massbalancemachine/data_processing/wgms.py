import os
import urllib.request
import zipfile
import pandas as pd

wgms_zip_file = "DOI-WGMS-FoG-2026-02-10.zip"
data_path = ".data"

wgms_source_data_link = f"https://wgms.ch/downloads/{wgms_zip_file}"
local_path_wgms = f"{data_path}/{wgms_zip_file}"

wgms_folder = f"{data_path}/{wgms_zip_file.replace('.zip', '')}"


def check_and_download_wgms():
    os.makedirs(data_path, exist_ok=True)
    if not os.path.isdir(wgms_folder):
        if not os.path.isfile(local_path_wgms):
            print("Downloading from WGMS website")
            urllib.request.urlretrieve(wgms_source_data_link, local_path_wgms)
        print("Unzipping file")
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
            "balance_code",
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
        },
    )
    new_df["FROM_DATE"] = pd.to_datetime(new_df["FROM_DATE"]).dt.strftime("%Y%m%d")
    new_df["TO_DATE"] = pd.to_datetime(new_df["FROM_DATE"]).dt.strftime("%Y%m%d")

    return new_df
