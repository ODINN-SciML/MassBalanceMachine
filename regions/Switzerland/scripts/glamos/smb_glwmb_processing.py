import os
import re
from tqdm.notebook import tqdm
import xarray as xr
import pandas as pd
import warnings

from regions.Switzerland.scripts.config_CH import *
from regions.Switzerland.scripts.utils import *
from regions.Switzerland.scripts.geo_data import *


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


def get_GLAMOS_glwmb(glacier_name, cfg):
    """
    Load glacier-wide annual mass balance from GLAMOS grid files.

    For a given glacier, the function searches for available annual GLAMOS
    grid files (LV95 or LV03), loads them, converts them to WGS84 coordinates,
    and computes the glacier-wide mean annual mass balance for each year.

    Parameters
    ----------
    glacier_name : str
        Glacier identifier used in the GLAMOS directory structure.
    cfg : object
        Configuration object providing at least the attribute `dataPath`
        and used by GLAMOS helper functions.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by year with a single column 'GLAMOS Balance'
        containing glacier-wide annual mass balance values (m w.e.).
        If no valid files are found, the DataFrame may be empty.
    """
    years = get_glwd_glamos_years(cfg, glacier_name)
    if years == []:
        print(f"Warning: No GLAMOS data found for {glacier_name}.")

    def pick_ann_file(cfg, glacier_name, year):
        base = os.path.join(
            cfg.dataPath, path_distributed_MB_glamos, "GLAMOS", glacier_name
        )
        cand_lv95 = os.path.join(base, f"{year}_ann_fix_lv95.grid")
        cand_lv03 = os.path.join(base, f"{year}_ann_fix_lv03.grid")
        if os.path.exists(cand_lv95):
            return cand_lv95, "lv95"
        if os.path.exists(cand_lv03):
            return cand_lv03, "lv03"
        return None, None

    glamos_glwd_mb = []
    for year in years:
        grid_path_ann, proj = pick_ann_file(cfg, glacier_name, year)
        if grid_path_ann is None:
            warnings.warn(
                f"No ann file found for {glacier_name} {year} (lv95/lv03). Skipping."
            )
            continue

        metadata_ann, grid_data_ann = load_grid_file(grid_path_ann)
        ds_glamos_ann = convert_to_xarray_geodata(grid_data_ann, metadata_ann)

        if proj == "lv03":
            ds_glamos_wgs84_ann = transform_xarray_coords_lv03_to_wgs84(ds_glamos_ann)
        elif proj == "lv95":
            ds_glamos_wgs84_ann = transform_xarray_coords_lv95_to_wgs84(ds_glamos_ann)
        else:
            raise RuntimeError(f"Unknown projection for {grid_path_ann}")

        glamos_glwd_mb.append(float(ds_glamos_wgs84_ann.mean().values))

    df = pd.DataFrame({"GLAMOS Balance": glamos_glwd_mb, "YEAR": years})

    # set index years
    df.set_index("YEAR", inplace=True)
    return df


def get_glwd_glamos_years(cfg, glacier_name):
    """
    Retrieve available years of GLAMOS glacier-wide annual mass balance data.

    The function scans the GLAMOS directory for a given glacier and returns
    all years for which an annual mass balance grid file is available
    (either LV95 or LV03 projection).

    Parameters
    ----------
    cfg : object
        Configuration object providing at least the attribute `dataPath`.
    glacier_name : str
        Glacier identifier used in the GLAMOS directory structure.

    Returns
    -------
    list of int
        Sorted list of years for which GLAMOS annual grid files are available.
        Returns an empty list if the glacier folder does not exist or no files
        are found.
    """
    folder = os.path.join(
        cfg.dataPath, path_distributed_MB_glamos, "GLAMOS", glacier_name
    )
    if not os.path.exists(folder):
        print(f"Warning: Folder {folder} does not exist.")
        return []

    # Match: 2005_ann_fix_lv95.grid OR 2005_ann_fix_lv03.grid
    pattern = re.compile(r"^(\d{4})_ann_fix_lv(?:95|03)\.grid$")

    years = []
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            years.append(int(match.group(1)))  # Extract the year as an integer

    years = np.unique(years).tolist()
    years.sort()
    return years
