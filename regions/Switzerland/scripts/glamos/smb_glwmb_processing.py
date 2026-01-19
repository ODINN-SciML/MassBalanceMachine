import os
import re

from regions.Switzerland.scripts.config_CH import *
from regions.Switzerland.scripts.utils import *


def process_SMB_GLAMOS(cfg):
    """
    Convert GLAMOS SMB `.dat` files (obs and fix variants) into CSV format.

    This function scans `<dataPath>/<path_SMB_GLAMOS_raw>` for `.dat` files containing
    'obs' and 'fix' in the filename, converts them using `process_dat_file_glwmb`,
    and writes outputs into:
    - `<path_SMB_GLAMOS_csv>/obs/`
    - `<path_SMB_GLAMOS_csv>/fix/`

    Parameters
    ----------
    cfg : object
        Configuration object with attribute `dataPath`. Uses `path_SMB_GLAMOS_raw`
        and `path_SMB_GLAMOS_csv` constants from config.

    Returns
    -------
    None

    Side Effects
    ------------
    Empties output folders and writes CSV files.

    Raises
    ------
    FileNotFoundError
        If the raw folder does not exist.
    """

    # OBS:
    # Get all files with pmb (for winter and annual mb):
    glamosfiles_smb = []
    for file in os.listdir(cfg.dataPath + path_SMB_GLAMOS_raw):
        # check if current path is a file
        if (
            os.path.isfile(os.path.join(cfg.dataPath, path_SMB_GLAMOS_raw, file))
            and "obs" in file
        ):
            glamosfiles_smb.append(file)
    # print('Examples of index stake raw files:\n', glamosfiles_smb[:5])

    # Transform all files to csv
    emptyfolder(cfg.dataPath + path_SMB_GLAMOS_csv + "obs/")
    for file in glamosfiles_smb:
        fileName = re.split(".dat", file)[0]
        process_dat_file_glwd_mb(
            fileName,
            cfg.dataPath + path_SMB_GLAMOS_raw,
            cfg.dataPath + path_SMB_GLAMOS_csv + "obs/",
        )

    # Get all files with pmb (for winter and annual mb):
    glamosfiles_smb = []
    for file in os.listdir(cfg.dataPath + path_SMB_GLAMOS_raw):
        # check if current path is a file
        if (
            os.path.isfile(os.path.join(cfg.dataPath, path_SMB_GLAMOS_raw, file))
            and "fix" in file
        ):
            glamosfiles_smb.append(file)
    # print('Examples of index stake raw files:\n', glamosfiles_smb[:5])
    # Transform all files to csv
    emptyfolder(cfg.dataPath + path_SMB_GLAMOS_csv + "fix/")
    for file in glamosfiles_smb:
        fileName = re.split(".dat", file)[0]
        process_dat_file_glwd_mb(
            fileName,
            cfg.dataPath + path_SMB_GLAMOS_raw,
            cfg.dataPath + path_SMB_GLAMOS_csv + "fix/",
        )


def process_dat_file_glwd_mb(fileName, path_dat, path_csv):
    """
    Convert a GLWMB `.dat` file to CSV, keeping a fixed subset of columns and adding YEAR.

    The output keeps the first 12 fields of the data line (after parsing) and appends
    `YEAR`, extracted from the `date1` field (assumed YYYYMMDD).

    Parameters
    ----------
    fileName : str
        Base filename without extension.
    path_dat : str
        Directory containing the `.dat` file.
    path_csv : str
        Output directory for the `.csv` file.

    Returns
    -------
    None

    Side Effects
    ------------
    Writes a CSV file to disk.

    Raises
    ------
    FileNotFoundError
        If the input `.dat` file does not exist.
    """

    if not os.path.exists(path_csv):
        os.makedirs(path_csv)

    dat_path = os.path.join(path_dat, fileName + ".dat")
    csv_path = os.path.join(path_csv, fileName + ".csv")

    with (
        open(dat_path, "r", encoding="latin-1") as dat_file,
        open(csv_path, "w", newline="", encoding="latin-1") as csv_file,
    ):

        header_written = False

        for line in dat_file:
            line = line.strip()
            if not line:
                continue

            # HEADER
            if line.startswith("#"):
                if "MB_ID" in line and ";" in line and not header_written:
                    cols = [v.strip() for v in line.lstrip("#").split(";")]
                    cols = cols[:12]
                    cols.append("YEAR")
                    csv_file.write(",".join(cols) + "\n")
                    header_written = True
                continue

            # DATA
            parts = line.split()
            parts = [p.replace(",", "-") for p in parts]
            parts = parts[:12]

            if len(parts) >= 5:
                date1 = parts[4]
                year = date1[:4]
            else:
                year = ""

            parts.append(year)

            csv_file.write(",".join(parts) + "\n")
