# --- standard library ---
import os
import re
import tempfile
import shutil
import logging
from calendar import monthrange
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- third-party ---
import numpy as np
import pandas as pd
import xarray as xr
import pyproj
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial.distance import cdist
from tqdm import tqdm

# --- project/local ---
import massbalancemachine as mbm
from oggm import utils, workflow, tasks
from oggm import cfg as oggmCfg
from regions.Switzerland.scripts.wgs84_ch1903 import *
from regions.Switzerland.scripts.config_CH import *
from regions.Switzerland.scripts.helpers import *
from regions.Switzerland.scripts.geodata import (
    LV03toWGS84,
    xr_SGI_masked_topo,
    coarsenDS,
    get_rgi_sgi_ids,
    transformDates,
    load_grid_file,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


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


def clean_winter_dates(df_raw):
    # For some winter measurements the FROM_DATE is the same year as the TO_DATE (even same date)
    # Correct it by setting it to beginning of hydrological year:
    for index, row in df_raw.iterrows():
        if row["PERIOD"] == "winter":
            df_raw.loc[index, "FROM_DATE"] = (
                str(pd.to_datetime(row["TO_DATE"], format="%Y%m%d").year - 1) + "1001"
            )
    for i, row in df_raw.iterrows():
        if (
            pd.to_datetime(row["TO_DATE"], format="%Y%m%d").year
            - pd.to_datetime(row["FROM_DATE"], format="%Y%m%d").year
            != 1
        ):
            # throw error if not corrected
            raise ValueError(
                "Date correction failed:",
                row["GLACIER"],
                row["PERIOD"],
                row["FROM_DATE"],
                pd.to_datetime(row["FROM_DATE"], format="%Y%m%d").year,
                row["TO_DATE"],
                pd.to_datetime(row["TO_DATE"], format="%Y%m%d").year,
            )
    return df_raw


def check_multiple_rgi_ids(df):
    """
    Checks if any glacier is associated with more than one RGIId.
    """
    rgi_per_glacier = df.groupby("GLACIER")["RGIId"].nunique()
    glaciers_with_multiple_rgi = rgi_per_glacier[rgi_per_glacier > 1]
    if not glaciers_with_multiple_rgi.empty:
        return True
    else:
        return False


def clean_rgi_ids(df):
    """
    Cleans and preprocesses RGI IDs for specific glaciers based on predefined rules.
    """
    corrections = {
        # Format: 'GLACIER': {'valid_rgi': 'RGI60-XX.XXXXX', 'action': 'drop|replace'}
        "albigna": {"valid_rgi": "RGI60-11.02285", "action": "drop"},
        "adler": {"valid_rgi": "RGI60-11.02764", "action": "drop"},
        "allalin": {"valid_rgi": "RGI60-11.02704", "action": "drop"},
        "basodino": {"valid_rgi": "RGI60-11.01987", "action": "drop"},
        "blauschnee": {"action": "remove_glacier"},
        "corvatsch": {"valid_rgi": "RGI60-11.01962", "action": "drop"},
        "damma": {"valid_rgi": "RGI60-11.01246", "action": "drop"},
        "findelen": {"valid_rgi": "RGI60-11.02773", "action": "drop"},
        "hohlaub": {"valid_rgi": "RGI60-11.02679", "action": "drop"},
        "gries": {"valid_rgi": "RGI60-11.01876", "action": "drop"},
        "limmern": {"valid_rgi": "RGI60-11.00918", "action": "drop"},
        "ofental": {"action": "remove_glacier"},
        "orny": {"valid_rgi": "RGI60-11.02775", "action": "replace"},
        "otemma": {"valid_rgi": "RGI60-11.02801", "action": "drop"},
        "plattalva": {"valid_rgi": "RGI60-11.00892", "action": "replace"},
        "plainemorte": {"valid_rgi": "RGI60-11.02072", "action": "drop"},
        "rhone": {"valid_rgi": "RGI60-11.01238", "action": "drop"},
        "sanktanna": {"valid_rgi": "RGI60-11.01367", "action": "drop"},
        "sexrouge": {"valid_rgi": "RGI60-11.02244", "action": "drop"},
        "silvretta": {"valid_rgi": "RGI60-11.00804", "action": "drop"},
        "tsanfleuron": {"valid_rgi": "RGI60-11.02249", "action": "drop"},
        "unteraar": {"action": "remove_glacier"},
    }

    for glacier, details in corrections.items():
        if details["action"] == "drop":
            df.drop(
                df[(df.GLACIER == glacier) & (df.RGIId != details["valid_rgi"])].index,
                inplace=True,
            )
        elif details["action"] == "replace":
            df.loc[df.GLACIER == glacier, "RGIId"] = details["valid_rgi"]
        elif details["action"] == "remove_glacier":
            df.drop(df[df.GLACIER == glacier].index, inplace=True)

    return df


def remove_close_points(df_gl):
    df_gl_cleaned = pd.DataFrame()
    for year in df_gl.YEAR.unique():
        for period in ["annual", "winter"]:
            df_gl_y = df_gl[(df_gl.YEAR == year) & (df_gl.PERIOD == period)]
            if len(df_gl_y) <= 1:
                df_gl_cleaned = pd.concat([df_gl_cleaned, df_gl_y])
                continue

            # Calculate distances to other points
            df_gl_y["x"], df_gl_y["y"] = latlon_to_laea(
                df_gl_y["POINT_LAT"], df_gl_y["POINT_LON"]
            )

            distance = cdist(df_gl_y[["x", "y"]], df_gl_y[["x", "y"]], "euclidean")

            # Merge close points
            merged_indices = set()
            for i in range(len(df_gl_y)):
                if i in merged_indices:
                    continue  # Skip already merged points

                # Find close points (distance < 10m)
                close_indices = np.where(distance[i, :] < 10)[0]
                close_indices = [idx for idx in close_indices if idx != i]

                if close_indices:
                    mean_MB = df_gl_y.iloc[close_indices + [i]].POINT_BALANCE.mean()

                    # Assign mean balance to the first point
                    df_gl_y.loc[df_gl_y.index[i], "POINT_BALANCE"] = mean_MB

                    # Mark other indices for removal
                    merged_indices.update(close_indices)

            # Drop surplus points
            indices_to_drop = list(merged_indices)
            df_gl_y = df_gl_y.drop(df_gl_y.index[indices_to_drop])

            # Append cleaned DataFrame
            df_gl_cleaned = pd.concat([df_gl_cleaned, df_gl_y])

    # Final output
    df_gl_cleaned.reset_index(drop=True, inplace=True)
    points_dropped = len(df_gl) - len(df_gl_cleaned)
    log.info(f"--- Number of points dropped: {points_dropped}")
    return df_gl_cleaned if points_dropped > 0 else df_gl


def latlon_to_laea(lat, lon):
    # Define the transformer: WGS84 to ETRS89 / LAEA Europe
    transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:3035")

    # Perform the transformation
    easting, northing = transformer.transform(lat, lon)
    return easting, northing


def check_point_ids_contain_glacier(dataframe):
    """
    Checks that each row's POINT_ID contains the name of the GLACIER.

    Parameters:
        dataframe (pd.DataFrame): A pandas DataFrame with columns "GLACIER" and "POINT_ID".

    Returns:
        bool: True if all rows satisfy the condition, False otherwise.
        pd.DataFrame: A DataFrame of rows where the condition is not met.
    """
    if "GLACIER" not in dataframe.columns or "POINT_ID" not in dataframe.columns:
        raise ValueError("The dataframe must contain 'GLACIER' and 'POINT_ID' columns.")

    # Check condition
    invalid_rows = dataframe[
        ~dataframe.apply(lambda row: row["GLACIER"] in row["POINT_ID"], axis=1)
    ]

    # Report
    if invalid_rows.empty:
        print("All POINT_IDs correctly contain their respective GLACIER names.")
        return True, None
    else:
        print(
            f"Found {len(invalid_rows)} rows where POINT_ID does not contain the GLACIER name."
        )
        return False, invalid_rows


def process_dat_fileGLWMB(fileName, path_dat, path_csv):
    # crate path_csv if does not exist
    if not os.path.exists(path_csv):
        os.makedirs(path_csv)

    with open(path_dat + fileName + ".dat", "r", encoding="latin-1") as dat_file:
        with open(
            path_csv + fileName + ".csv", "w", newline="", encoding="latin-1"
        ) as csv_file:
            for num_rows, row in enumerate(dat_file):
                if num_rows == 0:
                    row = [value.strip() for value in row.split(";")]
                    num_el_bands = row[4]
                if num_rows == 1:
                    row = [value.strip() for value in row.split(";")]
                    # Add columns for each el band
                    # b_w_eb_i  :  n columns with area-mean winter balance of each elevation band  [mm w.e.]
                    # b_a_eb_i  :  n columns with area-mean annual balance of each elevation band  [mm w.e.]
                    # A_eb_i    :  n columns with area of each elevation band  [km2]
                    row += ["b_w_eb_" + str(i) for i in range(int(num_el_bands))]
                    row += ["b_a_eb" + str(i) for i in range(int(num_el_bands))]
                    row += ["A_eb_" + str(i) for i in range(int(num_el_bands))]
                    csv_file.write(",".join(row[:-1]) + "\n")
                if num_rows > 3:
                    row = [value.strip() for value in row.split(" ")]
                    # replace commas if there are any otherwise will create bug:
                    row = [value.replace(",", "-") for value in row]
                    # remove empty spaces
                    row = [i for i in row if i]
                    csv_file.write(",".join(row) + "\n")


def convert_to_xarray(grid_data, metadata, num_months):
    # Extract metadata values
    ncols = int(metadata["ncols"])
    nrows = int(metadata["nrows"])
    xllcorner = metadata["xllcorner"]
    yllcorner = metadata["yllcorner"]
    cellsize = metadata["cellsize"]

    # Create x and y coordinates based on the metadata
    x_coords = xllcorner + np.arange(ncols) * cellsize
    y_coords = yllcorner + np.arange(nrows) * cellsize

    time_coords = np.arange(num_months)

    if grid_data.shape != (num_months, nrows, ncols):
        raise ValueError(
            f"Expected grid_data shape ({num_months}, {nrows}, {ncols}), got {grid_data.shape}"
        )

    # Create the xarray DataArray
    data_array = xr.DataArray(
        np.flip(grid_data, axis=1),
        # grid_data,
        dims=("time", "y", "x"),
        coords={"time": time_coords, "y": y_coords, "x": x_coords},
        name="grid_data",
    )
    return data_array


def transform_xarray_coords_lv03_to_wgs84_time(data_array):
    # Extract time, y, and x dimensions
    time_dim = data_array.coords["time"]

    # Flatten the DataArray (values) and extract x and y coordinates for each time step
    flattened_values = data_array.values.reshape(
        -1
    )  # Flatten entire 3D array (time, y, x)

    # flattened_values = data_array.values.flatten()
    y_coords, x_coords = np.meshgrid(
        data_array.y.values, data_array.x.values, indexing="ij"
    )

    # Flatten the coordinate arrays
    flattened_x = np.tile(
        x_coords.flatten(), len(time_dim)
    )  # Repeat for each time step
    flattened_y = np.tile(
        y_coords.flatten(), len(time_dim)
    )  # Repeat for each time step

    # Create a DataFrame with columns for x, y, and value
    df = pd.DataFrame(
        {"x_pos": flattened_x, "y_pos": flattened_y, "value": flattened_values}
    )
    df["z_pos"] = 0

    # Convert to lat/lon
    df = LV03toWGS84(df)

    # Transform LV03 to WGS84 (lat, lon)
    lon, lat = df.lon.values, df.lat.values

    # Reshape the flattened WGS84 coordinates back to the original grid shape (time, y, x)
    lon = lon.reshape((len(time_dim), *x_coords.shape))  # Shape: (time, y, x)
    lat = lat.reshape((len(time_dim), *y_coords.shape))  # Shape: (time, y, x)

    # Assign the 1D WGS84 coordinates for swapping
    lon_1d = lon[0, 0, :]  # Use the first time slice, and take x (lon) values
    lat_1d = lat[0, :, 0]  # Use the first time slice, and take y (lat) values

    # Assign the WGS84 coordinates back to the xarray
    data_array = data_array.assign_coords(lon=("x", lon_1d))  # Assign longitudes
    data_array = data_array.assign_coords(lat=("y", lat_1d))  # Assign latitudes

    # First, swap 'x' with 'lon' and 'y' with 'lat', keeping the time dimension intact
    data_array = data_array.swap_dims({"x": "lon", "y": "lat"})

    # Reorder the dimensions to be (time, lon, lat)
    data_array = data_array.transpose("time", "lon", "lat")

    return data_array


def get_geodetic_MB(cfg):
    """
    Reads and processes the geodetic mass balance dataset for Swiss glaciers.
    - Filters out invalid date entries.
    - Ensures Astart matches the year from date_start, and Aend matches the year from date_end.
    - Identifies duplicates based on (Astart, Aend) and keeps only the row where date_end is closest to the end of the month.

    Returns:
        pd.DataFrame: Processed geodetic mass balance data.
    """

    # Load necessary data
    glacier_ids = get_glacier_ids(cfg)
    data_glamos = pd.read_csv(
        cfg.dataPath + path_PMB_GLAMOS_csv + "CH_wgms_dataset_all.csv"
    )

    # Read geodetic MB dataset
    geodetic_mb = pd.read_csv(
        cfg.dataPath + path_geodetic_MB_glamos + "dV_DOI2024_allcomb.csv"
    )
    # geodetic_mb = pd.read_csv(path_geodetic_MB_glamos +
    #                           'volumechange.csv', sep = ',')

    # Get RGI IDs for the glaciers
    rgi_gl = data_glamos.RGIId.unique()
    sgi_gl = [
        glacier_ids[glacier_ids["rgi_id.v6"] == rgi]["sgi-id"].values[0]
        for rgi in rgi_gl
    ]
    geodetic_mb = geodetic_mb[geodetic_mb["SGI-ID"].isin(sgi_gl)]

    # Add glacier_name to geodetic_mb based on SGI-ID
    glacier_names = [
        glacier_ids[glacier_ids["sgi-id"] == sgi_id].index[0]
        for sgi_id in geodetic_mb["SGI-ID"].values
    ]
    geodetic_mb["glacier_name"] = glacier_names

    # Replace 'claridenL' with 'clariden'
    geodetic_mb["glacier_name"] = geodetic_mb["glacier_name"].replace(
        "claridenL", "clariden"
    )

    # Rename A_end to Aend
    geodetic_mb.rename(columns={"A_end": "Aend"}, inplace=True)

    # geodetic_mb.rename(columns={'A_end': 'Aend'}, inplace=True)
    # geodetic_mb.rename(columns={'A_start': 'Astart'}, inplace=True)

    # Function to replace 9999 with September 30
    def fix_invalid_dates(date):
        date_str = str(date)
        if date_str.endswith("9999"):
            return f"{date_str[:4]}0930"  # Replace '9999' with '0930'
        return date_str

    # Apply the function to both columns
    geodetic_mb["date_start"] = geodetic_mb["date_start"].apply(fix_invalid_dates)
    geodetic_mb["date_end"] = geodetic_mb["date_end"].apply(fix_invalid_dates)

    # Convert to datetime format
    geodetic_mb["date_start"] = pd.to_datetime(
        geodetic_mb["date_start"], format="%Y%m%d", errors="coerce"
    )
    geodetic_mb["date_end"] = pd.to_datetime(
        geodetic_mb["date_end"], format="%Y%m%d", errors="coerce"
    )

    # Manually set Astart and Aend based on date_start and date_end
    geodetic_mb["Astart"] = geodetic_mb["date_start"].dt.year
    geodetic_mb["Aend"] = geodetic_mb["date_end"].dt.year

    glDirect = np.sort(
        [
            re.search(r"xr_direct_(.*?)\.zarr", f).group(1)
            for f in os.listdir(cfg.dataPath + path_pcsr + "zarr/")
        ]
    )

    # filter to glaciers with potential clear sky radiation data
    geodetic_mb = geodetic_mb[geodetic_mb.glacier_name.isin(glDirect)]

    return geodetic_mb


def get_glacier_ids(cfg):
    glacier_ids = pd.read_csv(cfg.dataPath + path_glacier_ids, sep=",")
    glacier_ids.rename(columns=lambda x: x.strip(), inplace=True)
    glacier_ids.sort_values(by="short_name", inplace=True)
    glacier_ids.set_index("short_name", inplace=True)

    return glacier_ids


# --- Main processing functions --- #


def process_pmb_dat_files(cfg):
    """
    Processes annual and winter .dat PMB files into CSV format,
    handles Clariden glacier split, and removes combined files.
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


def process_annual_stake_data(path_csv_folder):
    """
    Processes annual glacier stake .csv files from GLAMOS into a cleaned WGMS-format DataFrame.

    Args:
        path_csv_folder (str): Path to folder containing processed annual PMB CSVs.

    Returns:
        pd.DataFrame: Cleaned and formatted annual mass balance data.
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

        df = transformDates(df)
        df = df.drop_duplicates()
        df = LV03toWGS84(df)
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
    Processes winter glacier stake .csv files into cleaned WGMS-format CSV files.

    Args:
        df_annual_raw (pd.DataFrame): Processed annual DataFrame (used to get valid glaciers).
        path_input (str): Path to input winter CSVs.
        path_output (str): Path to save cleaned winter CSVs.
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
        df_winter = transformDates(df_winter).drop_duplicates()
        df_winter = LV03toWGS84(df_winter)

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
    Combines processed annual and winter stake data into a single DataFrame,
    cleans winter date inconsistencies, removes duplicates, and saves to disk.

    Args:
        df_annual_raw (pd.DataFrame): Preprocessed annual stake data.
        path_winter_clean (str): Path to cleaned winter stake CSVs.
        path_output (str): Output path to save the final combined dataset.

    Returns:
        pd.DataFrame: Combined annual and winter dataset.
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


def add_rgi_ids_to_df(df_all_raw, glacier_outline_fname):
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


def initialize_oggm_glacier_directories(
    cfg,
    working_dir=None,
    rgi_region="11",
    rgi_version="62",
    base_url="https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5_w_data/",
    log_level="WARNING",
    task_list=None,
    from_prepro_level=2,
    prepro_border=10,
):
    # Initialize OGGM config
    oggmCfg.initialize(logging_level=log_level)
    oggmCfg.PARAMS["border"] = 10
    oggmCfg.PARAMS["use_multiprocessing"] = True
    oggmCfg.PARAMS["continue_on_error"] = True

    # Module logger
    log = logging.getLogger(".".join(__name__.split(".")[:-1]))
    log.setLevel(log_level)

    # Set working directory
    if working_dir is None:
        working_dir = cfg.dataPath + path_OGGM
        emptyfolder(working_dir)
    # empty the working directory if it exists
    emptyfolder(working_dir)
    oggmCfg.PATHS["working_dir"] = working_dir

    # Get RGI file
    # rgi_dir = utils.get_rgi_dir(version=rgi_version, reset=False)
    path = utils.get_rgi_region_file(
        region=rgi_region, version=rgi_version, reset=False
    )
    rgidf = gpd.read_file(path)

    # Initialize glacier directories from preprocessed data
    print("Collecting from base_url: ", base_url)
    gdirs = workflow.init_glacier_directories(
        rgidf,
        from_prepro_level=from_prepro_level,
        prepro_base_url=base_url,
        prepro_border=prepro_border,
        reset=True,
        force=True,
    )

    # Default task list if none provided
    if task_list is None:
        task_list = [
            tasks.gridded_attributes,
            # tasks.gridded_mb_attributes,
            # get_gridded_features,
        ]

    # Run tasks
    for task in task_list:
        workflow.execute_entity_task(task, gdirs, print_log=False)

    return gdirs, rgidf


def export_oggm_grids(cfg, gdirs, subset_rgis=None, output_path=None):

    # Save OGGM xr for all needed glaciers:
    if output_path is None:
        output_path = cfg.dataPath + path_OGGM_xrgrids
    emptyfolder(output_path)

    records = []

    for gdir in gdirs:
        RGIId = gdir.rgi_id
        # only save a subset if it's not empty
        if subset_rgis is not None:
            # check if the glacier is in the subset
            # if not, skip it
            if RGIId not in subset_rgis:
                continue
        with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
            ds = ds.load()

        vars = ["hugonnet_dhdt", "consensus_ice_thickness", "millan_v"]

        if not all(var in ds for var in vars):
            missing_vars = [var for var in vars if var not in ds]
            records.append(
                {
                    "rgi_id": RGIId,
                    "missing_vars": missing_vars,
                }
            )

        # save ds
        ds.to_zarr(os.path.join(output_path, f"{RGIId}.zarr"))
    df_missing = pd.DataFrame(records)

    return df_missing


def merge_pmb_with_oggm_data(
    df_pmb,
    gdirs,
    rgi_region="11",
    rgi_version="6",
    variables_of_interest=None,
    verbose=True,
):
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
        stake_var_df = stake[variables_of_interest].to_dataframe()

        # Assign to original DataFrame
        for var in variables_of_interest:
            df_pmb.loc[group.index, var] = stake_var_df[var].values

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


def rename_stakes_by_elevation(df_pmb_topo):
    for glacierName in df_pmb_topo.GLACIER.unique():
        gl_data = df_pmb_topo[df_pmb_topo.GLACIER == glacierName]
        stakeIDS = gl_data.groupby("POINT_ID")[
            ["POINT_LAT", "POINT_LON", "POINT_ELEVATION"]
        ].mean()
        stakeIDS.reset_index(inplace=True)
        # Change the ID according to elevation
        new_ids = stakeIDS[["POINT_ID", "POINT_ELEVATION"]].sort_values(
            by="POINT_ELEVATION"
        )
        new_ids["POINT_ID_new"] = [f"{glacierName}_{i}" for i in range(len(new_ids))]
        for i, row in new_ids.iterrows():
            df_pmb_topo.loc[
                (df_pmb_topo.GLACIER == glacierName)
                & (df_pmb_topo.POINT_ID == row.POINT_ID),
                "POINT_ID",
            ] = row.POINT_ID_new
    return df_pmb_topo


def merge_pmb_with_sgi_data(
    df_pmb,  # cleaned PMB DataFrame
    path_masked_grids,  # path to SGI grids
    voi=["masked_aspect", "masked_slope", "masked_elev"],
):

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


def process_SMB_GLAMOS(cfg):
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
        process_dat_fileGLWMB(
            fileName,
            cfg.dataPath + path_SMB_GLAMOS_raw,
            cfg.dataPath + path_SMB_GLAMOS_csv + "obs/",
        )

    # FIX:
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
        process_dat_fileGLWMB(
            fileName,
            cfg.dataPath + path_SMB_GLAMOS_raw,
            cfg.dataPath + path_SMB_GLAMOS_csv + "fix/",
        )


def process_pcsr(cfg):
    glDirect = np.sort(
        os.listdir(cfg.dataPath + path_pcsr + "raw/")
    )  # Glaciers with data
    path_pcsr_save = cfg.dataPath + path_pcsr + "zarr/"

    # check folder exists otherwise create it
    if not os.path.exists(path_pcsr_save):
        os.makedirs(path_pcsr_save)
    # Clean output folder
    emptyfolder(path_pcsr_save)

    for glacierName in tqdm(glDirect, desc="glaciers", position=0):
        grid = os.listdir(cfg.dataPath + path_pcsr + "raw/" + glacierName)
        grid_year = int(re.findall(r"\d+", grid[0])[0])
        daily_grids = os.listdir(
            cfg.dataPath + path_pcsr + "raw/" + glacierName + "/" + grid[0]
        )
        # Sort by day number from 001 to 365
        daily_grids.sort()
        grids = []
        for fileName in daily_grids:
            if "grid" not in fileName:
                continue

            # Load daily grid file
            file_path = (
                cfg.dataPath
                + path_pcsr
                + "raw/"
                + glacierName
                + "/"
                + grid[0]
                + "/"
                + fileName
            )
            metadata, grid_data = load_grid_file(file_path)
            grids.append(grid_data)

        # Take monthly means:
        monthly_grids = []
        for i in range(12):
            num_days_month = monthrange(grid_year, i + 1)[1]
            monthly_grids.append(
                np.mean(
                    np.stack(
                        grids[i * num_days_month : (i + 1) * num_days_month], axis=0
                    ),
                    axis=0,
                )
            )

        monthly_grids = np.array(monthly_grids)
        num_months = monthly_grids.shape[0]

        # Convert to xarray (CH coordinates)
        data_array = convert_to_xarray(monthly_grids, metadata, num_months)

        # Convert to WGS84 (lat/lon) coordinates
        data_array_transf = transform_xarray_coords_lv03_to_wgs84_time(data_array)

        # Save xarray
        if glacierName == "findelen":
            data_array_transf.to_zarr(path_pcsr_save + f"xr_direct_{glacierName}.zarr")
            data_array_transf.to_zarr(path_pcsr_save + f"xr_direct_adler.zarr")
        elif glacierName == "stanna":
            data_array_transf.to_zarr(path_pcsr_save + f"xr_direct_sanktanna.zarr")
        else:
            data_array_transf.to_zarr(path_pcsr_save + f"xr_direct_{glacierName}.zarr")


# --- per-process initializer (caps threads, seeds if you want) ---
def _worker_init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


# --- single task executed in a worker ---
def _process_one_item(item, cfg, type_, path_save, path_SGI_topo):
    """
    Returns tuple: (status, item, msg)
    status in {"ok","skip","err"}
    """
    try:
        # lazily import heavy deps inside worker to reduce parent footprint
        import geopandas as gpd

        # read shapefile once per worker process
        shp_path = os.path.join(
            cfg.dataPath, path_SGI_topo, "inventory_sgi2016_r2020/SGI_2016_glaciers.shp"
        )
        glacier_outline_sgi = gpd.read_file(shp_path)

        # resolve SGI id
        if type_ == "glacier_name":
            sgi_id, rgi_id, rgi_shp = get_rgi_sgi_ids(cfg, item)
            if not sgi_id:
                return ("skip", item, "Missing SGI ID")
        elif type_ == "sgi_id":
            sgi_id = item
        else:
            return ("err", item, f"Unknown type '{type_}'")

        # build dataset
        ds = xr_SGI_masked_topo(glacier_outline_sgi, sgi_id, cfg)
        if ds is None:
            return ("skip", item, "xr_SGI_masked_topo returned None")

        # resample
        ds_resampled = coarsenDS(ds)
        if ds_resampled is None:
            return ("skip", item, "coarsenDS returned None")

        # # load svf file
        # path_xr_svf = os.path.join(cfg.dataPath, "GLAMOS/topo/RGI_v6_11",
        #                    "svf_nc_latlon/")
        # svf_path = os.path.join(path_xr_svf, f"{rgi_id}_svf_latlon.nc")
        # if not os.path.exists(svf_path):
        #     print(f"SVF not found for {rgi_id}: {svf_path}")
        # else:
        #     ds_svf = xr.open_dataset(svf_path)

        # atomic save: write to temp then replace
        final_path = os.path.join(path_save, f"{item}.zarr")
        tmp_dir = tempfile.mkdtemp(prefix=f".tmp_{item}_", dir=path_save)
        try:
            ds_resampled.to_zarr(tmp_dir, mode="w")
            # remove existing if present, then atomic move
            if os.path.exists(final_path):
                shutil.rmtree(final_path)
            os.replace(tmp_dir, final_path)
        except Exception as e:
            # cleanup tmp on failure
            try:
                if os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir)
            except Exception:
                pass
            return ("err", item, f"Save error: {e}")

        return ("ok", item, final_path)

    except Exception as e:
        return ("err", item, str(e))


def create_sgi_topo_masks_parallel(
    cfg, iterator, type="glacier_name", path_save=None, max_workers=None
):
    """
    Parallel version of create_sgi_topo_masks (CPU-only).
    Each item writes <item>.zarr into path_save.
    """
    if path_save is None:
        # assumes 'path_SGI_topo' is defined/importable in this scope
        path_save = os.path.join(cfg.dataPath, path_SGI_topo, "xr_masked_grids/")

    os.makedirs(path_save, exist_ok=True)

    # IMPORTANT: do NOT empty the folder *after* we start; clear it up front if desired:
    # emptyfolder(path_save)  # <- only if you truly want to wipe existing outputs

    iterator = list(iterator)
    n = len(iterator)
    if n == 0:
        print("No items to process.")
        return

    if max_workers is None:
        max_workers = min(max(1, (os.cpu_count() or 2) - 1), 32)

    # Linux: use 'fork' to avoid pickling helpers; great for notebooks too
    ctx = mp.get_context("fork")

    ok = skip = err = 0
    results = {}

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_worker_init,
        mp_context=ctx,
    ) as ex:
        futures = [
            ex.submit(_process_one_item, item, cfg, type, path_save, path_SGI_topo)
            for item in iterator
        ]

        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Processing ({max_workers} workers)",
        ):
            status, item, msg = fut.result()
            results[item] = (status, msg)
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
                # optional: print(f"[SKIP] {item}: {msg}")
            else:
                err += 1
                print(f"[ERR]  {item}: {msg}")

    print(f"Done. ok={ok}  skip={skip}  err={err}  total={n}")
    return results


def getStakesData(cfg):
    data_glamos = pd.read_csv(
        cfg.dataPath + path_PMB_GLAMOS_csv + "CH_wgms_dataset_all.csv"
    )

    # Glaciers with data of potential clear sky radiation
    # Format to same names as stakes:
    glDirect = np.sort(
        [
            re.search(r"xr_direct_(.*?)\.zarr", f).group(1)
            for f in os.listdir(cfg.dataPath + path_pcsr + "zarr/")
        ]
    )

    # Filter out glaciers without data:
    data_glamos = data_glamos[data_glamos.GLACIER.isin(glDirect)]
    return data_glamos
