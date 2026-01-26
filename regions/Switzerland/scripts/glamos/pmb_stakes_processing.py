# --- standard library ---
import os
import re
import logging
import math

# --- third-party ---
import numpy as np
import pandas as pd
import xarray as xr
import pyproj
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial.distance import cdist
from tqdm import tqdm
from oggm import utils, workflow, tasks
from oggm import cfg as oggmCfg

# --- project/local ---
import massbalancemachine as mbm
from regions.Switzerland.scripts.config_CH import *
from regions.Switzerland.scripts.utils import *
from regions.Switzerland.scripts.geo_data.geodata import LV03_to_WGS84

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# ------------------- PMB stake ingestion and WGMS formatting ------------------- #


def process_pmb_dat_files(cfg):
    """
    Convert raw GLAMOS PMB stake `.dat` files (annual + winter) to CSV and apply Clariden split.

    This function:
    1) Ensures output folders for annual and winter CSVs exist.
    2) Converts all `.dat` files found in the configured raw directories to `.csv`
       using `dat_to_csv`.
    3) Splits the Clariden combined file into separate `claridenL` and `claridenU`
       annual and winter CSVs.
    4) Deletes the original combined Clariden CSVs.

    Parameters
    ----------
    cfg : object
        Configuration object with attribute `dataPath`. Relies on configured path
        constants (e.g., `path_PMB_GLAMOS_*`) imported from `config_CH`.

    Returns
    -------
    None

    Side Effects
    ------------
    Creates directories, writes CSV files, deletes Clariden combined CSV files.

    Raises
    ------
    FileNotFoundError
        If expected raw input directories/files do not exist.
    """

    # Clean CSV output folders if they exist
    # otherwise create them:
    if not os.path.exists(cfg.dataPath + path_PMB_GLAMOS_csv_a):
        os.makedirs(cfg.dataPath + path_PMB_GLAMOS_csv_a)
    if not os.path.exists(cfg.dataPath + path_PMB_GLAMOS_csv_w):
        os.makedirs(cfg.dataPath + path_PMB_GLAMOS_csv_w)

    # List .dat files
    glamosfiles_mb_a = [
        file
        for file in os.listdir(cfg.dataPath + path_PMB_GLAMOS_a_raw)
        if os.path.isfile(os.path.join(cfg.dataPath, path_PMB_GLAMOS_a_raw, file))
    ]
    glamosfiles_mb_w = [
        file
        for file in os.listdir(cfg.dataPath + path_PMB_GLAMOS_w_raw)
        if os.path.isfile(os.path.join(cfg.dataPath, path_PMB_GLAMOS_w_raw, file))
    ]

    # Convert .dat files to .csv
    for file in glamosfiles_mb_a:
        fileName = re.split(r"\.dat", file)[0]
        dat_to_csv(
            fileName,
            cfg.dataPath + path_PMB_GLAMOS_a_raw,
            cfg.dataPath + path_PMB_GLAMOS_csv_a,
        )

    for file in glamosfiles_mb_w:
        fileName = re.split(r"\.dat", file)[0]
        dat_to_csv(
            fileName,
            cfg.dataPath + path_PMB_GLAMOS_w_raw,
            cfg.dataPath + path_PMB_GLAMOS_csv_w,
        )

    # Handle Clariden split (annual)
    fileName = "clariden_annual.csv"
    clariden_csv_a = pd.read_csv(
        cfg.dataPath + path_PMB_GLAMOS_csv_a + fileName,
        sep=",",
        header=0,
        encoding="latin-1",
    )
    clariden_csv_a[clariden_csv_a["# name"] == "L"].to_csv(
        cfg.dataPath + path_PMB_GLAMOS_csv_a + "claridenL_annual.csv", index=False
    )
    clariden_csv_a[clariden_csv_a["# name"] == "U"].to_csv(
        cfg.dataPath + path_PMB_GLAMOS_csv_a + "claridenU_annual.csv", index=False
    )

    # Handle Clariden split (winter)
    fileName = "clariden_winter.csv"
    clariden_csv_w = pd.read_csv(
        cfg.dataPath + path_PMB_GLAMOS_csv_w + fileName,
        sep=",",
        header=0,
        encoding="latin-1",
    )
    clariden_csv_w[clariden_csv_w["# name"] == "L"].to_csv(
        cfg.dataPath + path_PMB_GLAMOS_csv_w + "claridenL_winter.csv", index=False
    )
    clariden_csv_w[clariden_csv_w["# name"] == "U"].to_csv(
        cfg.dataPath + path_PMB_GLAMOS_csv_w + "claridenU_winter.csv", index=False
    )

    # Remove original Clariden files
    os.remove(cfg.dataPath + path_PMB_GLAMOS_csv_a + "clariden_annual.csv")
    os.remove(cfg.dataPath + path_PMB_GLAMOS_csv_w + "clariden_winter.csv")


def dat_to_csv(fileName, path_dat, path_csv):
    """
    Converts a `.dat` file into a `.csv` file.

    This function processes a `.dat` file located in a specified directory,
    performs data cleaning and transformations, and saves the processed
    content as a `.csv` file in another specified directory.

    Parameters:
        fileName (str): The name of the file (without extension) to be converted.
        path_dat (str): The directory path where the `.dat` file is located.
        path_csv (str): The directory path where the output `.csv` file will be saved.

    File Format Assumptions:
        - The `.dat` file uses a semicolon (`;`) as a delimiter in the second row.
        - Rows after the third row use spaces as a delimiter.
        - Empty spaces and commas in the data are cleaned:
            - Commas in data values are replaced with hyphens (`-`).
            - Empty strings are removed from the row.

    Processing Steps:
        1. The second row of the `.dat` file is treated as the header.
           - Values are stripped of whitespace and joined with a comma (`,`).
        2. Rows after the third row are treated as data rows.
           - Values are stripped of whitespace.
           - Commas within the data values are replaced with hyphens (`-`).
           - Empty values are removed.
        3. The processed rows are written to the `.csv` file.

    Encoding:
        - The function uses `latin-1` encoding to handle file reading and writing.

    Example:
        Given a `.dat` file "example.dat" with content:
        ```
        Header information (ignored)
        ;Col1;Col2;Col3;
        Another header (ignored)
        Data start
        Value1 Value2 Value3
        Value4 Value5,Value6
        ```
        The resulting "example.csv" will contain:
        ```
        Col1,Col2,Col3
        Value1,Value2,Value3
        Value4,Value5-Value6
        ```

    Notes:
        - The function assumes the file structure outlined above and may not work with different formats.
        - Ensure the provided paths end with a directory separator (`/` or `\\`) based on the operating system.

    """
    # create path_csv if does not exist
    if not os.path.exists(path_csv):
        os.makedirs(path_csv)

    with open(path_dat + fileName + ".dat", "r", encoding="latin-1") as dat_file:
        with open(
            path_csv + fileName + ".csv", "w", newline="", encoding="latin-1"
        ) as csv_file:
            for num_rows, row in enumerate(dat_file):
                if num_rows == 1:
                    row = [value.strip() for value in row.split(";")]
                    csv_file.write(",".join(row) + "\n")
                if num_rows > 3:
                    row = [value.strip() for value in row.split(" ")]
                    # Replace commas if there are any, otherwise this will create a bug.
                    row = [value.replace(",", "-") for value in row]
                    # Remove empty spaces.
                    row = [i for i in row if i]
                    csv_file.write(",".join(row) + "\n")


def process_annual_stake_data(path_csv_folder):
    """
    Load and clean annual GLAMOS stake CSVs into a WGMS-like table.

    For each annual CSV in `path_csv_folder`, this function:
    - adds glacier/period fields
    - fixes/standardizes dates via `transform_dates`
    - converts LV03 coordinates to WGS84 via `LV03_to_WGS84`
    - builds the standard column set/order used downstream
    - filters out early years (<1950) and invalid measurement periods
    - filters by measurement type/quality

    Parameters
    ----------
    path_csv_folder : str
        Folder containing per-glacier annual CSV files (typically produced by `process_pmb_dat_files`).

    Returns
    -------
    pandas.DataFrame
        Cleaned annual stake dataset in WGMS-like column structure.

    Raises
    ------
    FileNotFoundError
        If `path_csv_folder` does not exist.
    KeyError
        If expected columns are missing from the input CSVs.
    """
    df_list = []

    # check path_csv_folder is not empty
    if not os.path.exists(path_csv_folder):
        raise FileNotFoundError(f"Path {path_csv_folder} does not exist.")

    for file in os.listdir(path_csv_folder):
        fileName = re.split(r"\.csv", file)[0]
        glacierName = re.split(r"_", fileName)[0]

        df = pd.read_csv(
            os.path.join(path_csv_folder, file), sep=",", header=0, encoding="latin-1"
        )
        df["glacier"] = glacierName
        df["period"] = "annual"

        df = transform_dates(df)
        df = df.drop_duplicates()
        df = LV03_to_WGS84(df)
        df_list.append(df)

    # Concatenate and process
    df_annual_raw = pd.concat(df_list, ignore_index=True)
    df_annual_raw["YEAR"] = pd.to_datetime(df_annual_raw["date1"]).dt.year
    df_annual_raw = df_annual_raw[df_annual_raw["YEAR"] >= 1950]

    # Rename and reorder columns
    columns_mapping = {
        "# name": "POINT_ID",
        "lat": "POINT_LAT",
        "lon": "POINT_LON",
        "height": "POINT_ELEVATION",
        "date0": "FROM_DATE",
        "date1": "TO_DATE",
        "mb_we": "POINT_BALANCE",
        "glacier": "GLACIER",
        "period": "PERIOD",
    }
    df_annual_raw.rename(columns=columns_mapping, inplace=True)

    columns_order = [
        "YEAR",
        "POINT_ID",
        "GLACIER",
        "FROM_DATE",
        "TO_DATE",
        "POINT_LAT",
        "POINT_LON",
        "POINT_ELEVATION",
        "POINT_BALANCE",
        "PERIOD",
        "date_fix0",
        "date_fix1",
        "time0",
        "time1",
        "date_quality",
        "position_quality",
        "mb_raw",
        "density",
        "density_quality",
        "measurement_quality",
        "measurement_type",
        "mb_error",
        "reading_error",
        "density_error",
        "error_evaluation_method",
        "source",
    ]
    df_annual_raw = df_annual_raw[columns_order]

    # Keep only 1-year measurement periods
    valid_date_mask = (
        pd.to_datetime(df_annual_raw["TO_DATE"], format="%Y%m%d").dt.year
        - pd.to_datetime(df_annual_raw["FROM_DATE"], format="%Y%m%d").dt.year
    ) == 1
    df_annual_raw = df_annual_raw[valid_date_mask]

    # Filter by measurement type and quality
    df_annual_raw = df_annual_raw[
        (df_annual_raw["measurement_type"] <= 2)
        & (df_annual_raw["measurement_quality"] == 1)
    ]

    df_annual_raw = df_annual_raw.drop_duplicates()

    return df_annual_raw


def process_winter_stake_data(df_annual_raw, path_input, path_output):
    """
    Load and clean winter GLAMOS stake CSVs and write per-glacier cleaned CSV files.

    Winter stake data are only processed for glaciers that also appear in `df_annual_raw`.
    This function:
    - creates/empties `path_output`
    - reads each `<glacier>_winter.csv`
    - fixes/standardizes dates via `transform_dates`
    - converts LV03 to WGS84 via `LV03_to_WGS84`
    - applies year/quality filters
    - writes `<glacier>_winter_all.csv` to `path_output`

    Parameters
    ----------
    df_annual_raw : pandas.DataFrame
        Cleaned annual dataset used to determine valid glaciers.
        Must contain a `GLACIER` column.
    path_input : str
        Folder containing raw winter stake CSVs.
    path_output : str
        Output folder where cleaned winter CSVs will be written.

    Returns
    -------
    None

    Side Effects
    ------------
    Creates/empties output folder and writes cleaned winter CSV files.

    Raises
    ------
    FileNotFoundError
        If `path_input` does not exist or required files are missing.
    KeyError
        If required columns are missing from winter inputs.
    """

    # Clear output folder if exists, otherwise create it
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    else:
        emptyfolder(path_output)

    # Identify glaciers with available winter data
    winter_glaciers = {
        re.split(r"_winter\.csv", f)[0]
        for f in os.listdir(path_input)
        if f.endswith("_winter.csv")
    }

    # Only keep glaciers that are also present in annual data
    annual_glaciers = set(df_annual_raw["GLACIER"].unique())
    glaciers_to_process = annual_glaciers.intersection(winter_glaciers)

    # Process each glacier
    for glacier in glaciers_to_process:
        file_path = os.path.join(path_input, f"{glacier}_winter.csv")
        df_winter = pd.read_csv(file_path, sep=",", header=0, encoding="latin-1")

        df_winter["period"] = "winter"
        df_winter["glacier"] = glacier

        # Date and coordinate transformation
        df_winter = transform_dates(df_winter).drop_duplicates()
        df_winter = LV03_to_WGS84(df_winter)

        # Add YEAR and filter
        df_winter["YEAR"] = pd.to_datetime(df_winter["date1"]).dt.year
        df_winter = df_winter[df_winter["YEAR"] >= 1950]

        # Rename and reorder columns
        columns_mapping = {
            "# name": "POINT_ID",
            "lat": "POINT_LAT",
            "lon": "POINT_LON",
            "height": "POINT_ELEVATION",
            "date0": "FROM_DATE",
            "date1": "TO_DATE",
            "mb_we": "POINT_BALANCE",
            "glacier": "GLACIER",
            "period": "PERIOD",
        }
        df_winter.rename(columns=columns_mapping, inplace=True)

        columns_order = [
            "YEAR",
            "POINT_ID",
            "GLACIER",
            "FROM_DATE",
            "TO_DATE",
            "POINT_LAT",
            "POINT_LON",
            "POINT_ELEVATION",
            "POINT_BALANCE",
            "PERIOD",
            "date_fix0",
            "date_fix1",
            "time0",
            "time1",
            "date_quality",
            "position_quality",
            "mb_raw",
            "density_quality",
            "measurement_quality",
            "measurement_type",
            "mb_error",
            "reading_error",
            "density",
            "density_error",
            "error_evaluation_method",
            "source",
        ]
        df_winter = df_winter[columns_order]

        # Filter valid measurements
        df_winter = df_winter[
            (df_winter["measurement_type"] <= 2)
            & (df_winter["measurement_quality"] == 1)
        ]

        # Save cleaned DataFrame
        output_file = os.path.join(path_output, f"{glacier}_winter_all.csv")
        df_winter.to_csv(output_file, index=False)


def assemble_all_stake_data(df_annual_raw, path_winter_clean, path_output):
    """
    Combine cleaned annual and winter stake datasets into a single table.

    This function:
    - loads all files matching `*_winter_all.csv` in `path_winter_clean`
    - concatenates them with `df_annual_raw`
    - applies `clean_winter_dates` to fix winter FROM/TO inconsistencies
    - removes the glacier 'pers' (treated as part of Morteratsch ensemble)

    Parameters
    ----------
    df_annual_raw : pandas.DataFrame
        Cleaned annual stake dataset.
    path_winter_clean : str
        Folder containing cleaned winter stake CSVs.
    path_output : str
        Reserved for compatibility; currently not used for writing in the shown implementation.

    Returns
    -------
    pandas.DataFrame
        Combined annual + winter stake dataset.

    Notes
    -----
    The current implementation does not write to `path_output`. If you want it
    to save, add an explicit `to_csv(...)` call.
    """

    # Collect all cleaned winter stake files
    files_stakes = [f for f in os.listdir(path_winter_clean) if "_winter_all" in f]

    winter_dataframes = []
    for file in files_stakes:
        df_winter = pd.read_csv(
            os.path.join(path_winter_clean, file), sep=",", header=0, encoding="latin-1"
        ).drop(columns="Unnamed: 0", errors="ignore")
        # Filter out empty or all-NA DataFrames
        if not df_winter.empty and not df_winter.isna().all(axis=None):
            winter_dataframes.append(df_winter)

    # Combine with annual data
    df_all_raw = pd.concat([df_annual_raw] + winter_dataframes, ignore_index=True)
    df_all_raw.reset_index(drop=True, inplace=True)

    # Clean winter date inconsistencies
    df_all_raw = clean_winter_dates(df_all_raw)

    # Remove Pers glacier (part of Morteratsch ensemble)
    df_all_raw = df_all_raw[df_all_raw["GLACIER"] != "pers"]

    return df_all_raw


# ---------------------------------------Point enrichment / feature joining ------------------------------------------- #


def add_rgi_ids_to_df(df_all_raw, glacier_outline_fname):
    """
    Attach RGIId to each stake point by intersecting with glacier outlines; fill gaps by nearest outline.

    Steps:
    1) Selects core WGMS-like columns from `df_all_raw`.
    2) Uses `mbm.data_processing.utils.get_rgi` to assign RGIId by intersection.
    3) For unmatched points, assigns the closest glacier outline polygon's RGIId.

    Parameters
    ----------
    df_all_raw : pandas.DataFrame
        Stake dataset with columns including POINT_LAT/POINT_LON and GLACIER metadata.
    glacier_outline_fname : str
        Path to a glacier outline file readable by GeoPandas (e.g., shapefile, geopackage).

    Returns
    -------
    pandas.DataFrame
        DataFrame with a populated `RGIId` column.

    Raises
    ------
    FileNotFoundError
        If `glacier_outline_fname` cannot be read.
    """
    # Select relevant columns
    df_pmb = df_all_raw[
        [
            "YEAR",
            "POINT_ID",
            "GLACIER",
            "FROM_DATE",
            "TO_DATE",
            "POINT_LAT",
            "POINT_LON",
            "POINT_ELEVATION",
            "POINT_BALANCE",
            "PERIOD",
        ]
    ].copy()

    # Load the glacier outlines
    glacier_outline = gpd.read_file(glacier_outline_fname)

    # Add RGI IDs through intersection
    df_pmb = mbm.data_processing.utils.get_rgi(
        data=df_pmb, glacier_outlines=glacier_outline
    )

    # Handle unmatched points
    no_match_df = df_pmb[df_pmb["RGIId"].isna()]
    geometry = [
        Point(lon, lat)
        for lon, lat in zip(no_match_df["POINT_LON"], no_match_df["POINT_LAT"])
    ]
    points_gdf = gpd.GeoDataFrame(
        no_match_df, geometry=geometry, crs=glacier_outline.crs
    )

    for index in tqdm(no_match_df.index, desc="Finding closest RGIId"):
        point = points_gdf.loc[index]["geometry"]
        polygon_index = glacier_outline.distance(point).sort_values().index[0]
        closest_rgi = glacier_outline.loc[polygon_index].RGIId
        df_pmb.at[index, "RGIId"] = closest_rgi

    return df_pmb


def merge_pmb_with_sgi_data(
    df_pmb,  # cleaned PMB DataFrame
    path_masked_grids,  # path to SGI grids
    voi=["masked_aspect", "masked_slope", "masked_elev"],
):
    """
    Enrich a PMB point dataset with SGI topographic variables by nearest-neighbor sampling.

    For each glacier present in both `df_pmb` and `path_masked_grids`, this function:
    - opens `<glacier>.zarr` (SGI masked topo dataset on lon/lat grid)
    - selects nearest grid cell for each stake point (POINT_LON, POINT_LAT)
    - assigns requested variables into the dataframe

    Parameters
    ----------
    df_pmb : pandas.DataFrame
        Stake dataset containing at least columns: `GLACIER`, `POINT_LON`, `POINT_LAT`.
    path_masked_grids : str
        Folder containing `<glacier>.zarr` datasets.
    voi : list of str, optional
        Variables of interest to sample from the SGI datasets.

    Returns
    -------
    pandas.DataFrame
        Copy of `df_pmb` filtered to glaciers with SGI grids and enriched with:
        - `aspect_sgi`, `slope_sgi`, `topo_sgi` (renamed outputs)

    Raises
    ------
    KeyError
        If required columns are missing.
    """

    # Get fully processed SGI glacier names
    sgi_glaciers = set(
        re.split(r".zarr", f)[0]
        for f in os.listdir(path_masked_grids)
        if f.endswith(".zarr")
    )

    # Filter DataFrame for glaciers with SGI grid only
    df_pmb_sgi = df_pmb[df_pmb.GLACIER.isin(sgi_glaciers)].copy()

    # Initialize empty columns for variables of interest
    for var in voi:
        df_pmb_sgi[var] = np.nan

    # Group rows by glacier name to process each glacier in bulk
    grouped = df_pmb_sgi.groupby("GLACIER")

    # Process each glacier
    for glacier_name, group in grouped:
        try:
            # Open the dataset for the current glacier
            file_path = os.path.join(path_masked_grids, f"{glacier_name}.zarr")
            ds_sgi = xr.open_dataset(file_path)

            # Transform coordinates for the group
            lon = group["POINT_LON"].values
            lat = group["POINT_LAT"].values

            # Select nearest values for all points in the group
            stake = ds_sgi.sel(
                lon=xr.DataArray(lon, dims="points"),
                lat=xr.DataArray(lat, dims="points"),
                method="nearest",
            )

            # Extract variables of interest and convert to a DataFrame
            stake_var = stake[voi].to_dataframe().reset_index()

            # Map extracted values back to the original DataFrame
            for var in voi:
                df_pmb_sgi.loc[group.index, var] = stake_var[var].values
        except FileNotFoundError:
            log.error(f"File not found for glacier: {glacier_name}")
            continue

    # Rename columns
    df_pmb_sgi.rename(
        columns={
            "masked_aspect": "aspect_sgi",
            "masked_slope": "slope_sgi",
            "masked_elev": "topo_sgi",
        },
        inplace=True,
    )

    return df_pmb_sgi


def merge_pmb_with_oggm_data(
    df_pmb,
    gdirs,
    rgi_region="11",
    rgi_version="6",
    variables_of_interest=None,
    verbose=True,
):
    """
    Enrich stake point data with OGGM gridded variables and a within-glacier flag.

    For each RGIId group, this function:
    - loads OGGM `gridded_data` for the matching glacier directory
    - transforms stake points from WGS84 to the OGGM projection
    - samples nearest grid cell values for selected variables
    - computes whether each point lies within the RGI glacier polygon (spatial join)
    - converts `aspect` and `slope` from radians to degrees

    Parameters
    ----------
    df_pmb : pandas.DataFrame
        Stake dataset with at least columns: `RGIId`, `POINT_LON`, `POINT_LAT`, `PERIOD`.
    gdirs : list
        List of OGGM GlacierDirectory objects.
    rgi_region : str, optional
        RGI region (default "11").
    rgi_version : str, optional
        RGI version passed to OGGM utilities (default "6").
    variables_of_interest : list of str, optional
        Variables to sample from OGGM gridded dataset.
    verbose : bool, optional
        If True, logs warnings and summary counts.

    Returns
    -------
    pandas.DataFrame
        Input dataframe with added columns for sampled OGGM variables and:
        - `within_glacier_shape` : bool

    Raises
    ------
    KeyError
        If required columns are missing.
    """
    if variables_of_interest is None:
        variables_of_interest = [
            "aspect",
            "slope",
            "topo",
            "hugonnet_dhdt",
            "consensus_ice_thickness",
            "millan_v",
        ]
        # other options: "millan_ice_thickness", "millan_vx", "millan_vy", "dis_from_border"

    # Load RGI shapefile
    path = utils.get_rgi_region_file(region=rgi_region, version=rgi_version)
    rgidf = gpd.read_file(path)

    # Initialize empty columns
    for var in variables_of_interest:
        df_pmb[var] = np.nan
    df_pmb["within_glacier_shape"] = False

    grouped = df_pmb.groupby("RGIId")

    for rgi_id, group in grouped:
        # Find corresponding glacier directory
        gdir = next((gd for gd in gdirs if gd.rgi_id == rgi_id), None)
        if gdir is None:
            if verbose:
                log.error(
                    f"Warning: No glacier directory for RGIId {rgi_id}, skipping..."
                )
            continue

        with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
            ds = ds.load()

        # Match RGI shape
        glacier_shape = rgidf[rgidf["RGIId"] == rgi_id]
        if glacier_shape.empty:
            if verbose:
                log.error(f"Warning: No shape found for RGIId {rgi_id}, skipping...")
            continue

        # Coordinate transformation from WGS84 to the projection of OGGM data
        transf = pyproj.Transformer.from_proj(
            pyproj.CRS.from_user_input("EPSG:4326"),
            pyproj.CRS.from_user_input(ds.pyproj_srs),
            always_xy=True,
        )
        lon, lat = group["POINT_LON"].values, group["POINT_LAT"].values
        x_stake, y_stake = transf.transform(lon, lat)

        # Create GeoDataFrame of points
        geometry = [Point(xy) for xy in zip(lon, lat)]
        points_rgi = gpd.GeoDataFrame(group, geometry=geometry, crs="EPSG:4326")

        # Intersect with glacier shape
        glacier_shape = glacier_shape.to_crs(points_rgi.crs)
        points_in_glacier = gpd.sjoin(
            points_rgi.loc[group.index], glacier_shape, predicate="within", how="inner"
        )

        # Get nearest OGGM grid data for points
        stake = ds.sel(
            x=xr.DataArray(x_stake, dims="points"),
            y=xr.DataArray(y_stake, dims="points"),
            method="nearest",
        )

        # # if all variables:
        # stake_var_df = stake[variables_of_interest].to_dataframe()
        # variables that actually exist in the dataset
        present_vars = [v for v in variables_of_interest if v in stake.data_vars]
        missing_vars = [v for v in variables_of_interest if v not in stake.data_vars]

        # warn globally
        if missing_vars:
            if verbose:
                log.warning(f"Missing variables for {rgi_id}: {missing_vars}")

        # extract only the existing variables
        stake_var_df = stake[present_vars].to_dataframe()

        # Assign to original DataFrame
        df_pmb.loc[group.index, present_vars] = stake_var_df[present_vars].values

        df_pmb.loc[points_in_glacier.index, "within_glacier_shape"] = True

    # Convert radians to degrees
    df_pmb["aspect"] = df_pmb["aspect"].apply(
        lambda x: math.degrees(x) if not pd.isna(x) else x
    )
    df_pmb["slope"] = df_pmb["slope"].apply(
        lambda x: math.degrees(x) if not pd.isna(x) else x
    )

    if verbose:
        log.info("-- Number of winter and annual samples:", len(df_pmb))
        log.info("-- Number of annual samples:", len(df_pmb[df_pmb.PERIOD == "annual"]))
        log.info("-- Number of winter samples:", len(df_pmb[df_pmb.PERIOD == "winter"]))

    return df_pmb
