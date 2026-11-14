import os
import io
import json
import zipfile
import urllib.request

from data_processing.product_utils import get_data_path

"""
This script downloads the glaciological point mass balance measurements of NVE for
mainland Norway, 1962-2021 (Elvehøy et al., 2025, https://doi.org/10.58059/sjse-6w92).
It is the published version of the stake dataset of Sjursen et al. (2025).

The dataset is archived in NVA (Nasjonalt vitenarkiv). Its file is not served at a
fixed address: the API returns a presigned S3 link valid for a short time, which is
requested again at every download.
"""

NVA_API = "https://api.nva.unit.no/publication"
NVA_PUBLICATION_ID = "01953ca70ea8-d0d31171-7b4d-4271-9d93-1291f4133b08"
NVA_FILE_ID = "bc47f559-21be-4c72-aee4-d8d256c078d1"
CSV_FILE = "glaciological_point_mass_balance_Norway.csv"


def nve_path():
    return os.path.join(get_data_path(), "stakes", "nve") + "/"


def point_mb_path():
    return os.path.join(nve_path(), CSV_FILE)


def download_point_mb():
    """Download the NVE point mass balance archive and extract its CSV file and README
    to `nve_path()`."""
    print("Downloading NVE point mass balance data")
    url_link = f"{NVA_API}/{NVA_PUBLICATION_ID}/filelink/{NVA_FILE_ID}"
    with urllib.request.urlopen(url_link, timeout=60) as response:
        presigned_url = json.load(response)["id"]
    with urllib.request.urlopen(presigned_url, timeout=300) as response:
        content = response.read()

    os.makedirs(nve_path(), exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for name in [CSV_FILE, "README.txt"]:
            with open(os.path.join(nve_path(), name), "wb") as f:
                f.write(z.read(name))
    print(f"Download complete: {point_mb_path()}")
