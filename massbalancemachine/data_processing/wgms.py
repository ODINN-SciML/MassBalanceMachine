import os
import urllib.request
import zipfile
import pandas as pd

wgms_zip_file = "DOI-WGMS-FoG-2025-02b.zip"
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
        with zipfile.ZipFile(local_path_wgms, 'r') as zip_ref:
            zip_ref.extractall(wgms_folder)

def load_wgms_data():
    check_and_download_wgms()
    point_mb_file = f"{wgms_folder}/data/mass_balance_point.csv"
    data = pd.read_csv(point_mb_file)
    return data
