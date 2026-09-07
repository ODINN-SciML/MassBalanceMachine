from pathlib import Path
import os
import zenodo_get
import zipfile

from data_processing.product_utils import data_path

_code_to_folder = {
    "FVG": "ARPA_FVG",
    "LOM": "ARPA_Lombardia",
    "PIE": "ARPA_Piemonte",
    "EMR": "ARPAE",
    "LIG": "ARPAL",
    "SLV": "ARSO",
    "CZE": "CHMU",
    "CRO": "DHMZ",
    "GER": "DWD",
    "IED": "ECAD",
    "GBH": "GHCN",
    "FRA": "MeteoFrance",
    "TRE": "MeteoTrentino",
    "HUN": "OMSZ",
    "SUT": "Provincia_Autonoma_di_Bolzano",
    "MAR": "Regione_Marche",
    "TUS": "Regione_Toscana",
    "UMB": "Regione_Umbria",
    "VDA": "Regione_Valle_Aosta",
    "SVK": "SHMU",
    "AUT": "ZAMG",
}
_override_zip = {"ZAMG": "Geosphere"}

eear_dir = os.path.join(data_path, "AWS", "EEAR-Clim")


def _ensure_dataset():
    extract = False
    download = False
    for e in _code_to_folder.values():
        if not Path(os.path.join(eear_dir, e)).is_dir():
            extract = True
        ee = _override_zip.get(e, e)
        if not os.path.isfile(os.path.join(eear_dir, f"{ee}.zip")):
            download = True

    if extract:
        print("Downloading the EEAR-Clim dataset")
        if download:
            zenodo_get.download(
                record_or_doi="10.5281/zenodo.10951609",
                output_dir=eear_dir,
            )

        zipfiles = [
            f
            for f in os.listdir(eear_dir)
            if os.path.isfile(os.path.join(eear_dir, f)) and f.endswith(".zip")
        ]
        for f in zipfiles:
            if f == "scripts.zip":
                continue
            zip_path = os.path.join(eear_dir, f)
            print(f"Unzipping {f}")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(eear_dir)
