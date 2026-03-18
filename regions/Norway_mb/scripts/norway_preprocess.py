import pandas as pd
import numpy as np
import pyproj
import xarray as xr
import geopandas as gpd
import math
from itertools import combinations
from geopy.distance import geodesic
from shapely.geometry import Point
from tqdm import tqdm
import logging
from oggm import utils, workflow, tasks
from oggm import cfg as oggmCfg
from scipy.spatial.distance import cdist

from scripts.config_NOR import *
from scripts.helpers import *

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


# --- Preprocess --- #


def fill_missing_dates(df):
    """
    The first measurement date is missing for some glaciers.
    Fill missing prev_yr_min_date values with October 1st assumed hydr. year
    start date. based on curr_yr_min_date.
    """
    # Create a copy to avoid modifying the original dataframe
    df_filled = df.copy()

    # Find rows with missing prev_yr_min_date
    missing_mask = df_filled["prev_yr_min_date"].isna()

    if missing_mask.sum() > 0:
        print(f"Filling {missing_mask.sum()} missing previous year dates...")

        # For each row with missing prev_yr_min_date
        for idx in df_filled[missing_mask].index:
            if pd.notna(df_filled.loc[idx, "curr_yr_max_date"]):
                # Extract date components
                try:
                    date_str = df_filled.loc[idx, "curr_yr_max_date"]
                    date_obj = pd.to_datetime(date_str, format="%d.%m.%Y")

                    # Get previous year
                    prev_year = date_obj.year - 1

                    # Create October 1st date for previous year
                    new_date = f"01.10.{prev_year}"

                    # Fill the value
                    df_filled.loc[idx, "prev_yr_min_date"] = new_date
                    df_filled.loc[idx, "DATA_MODIFICATION"] = (
                        "Filled missing FROM_DATE with October 1st of previous year"
                    )
                except:
                    print(
                        f"Warning: Could not process date '{date_str}' at index {idx}"
                    )
            else:
                print("no curr_yr_max_date entry to extract the year from")

    return df_filled


def split_stake_measurements(df_stakes):
    """
    Split stake measurements into separate annual and winter records.
    Only includes measurements where the mass balance value is not NaN.

    Args:
        df_stakes: DataFrame with combined stake measurement data

    Returns:
        DataFrame with separate rows for annual and winter measurements
    """

    # Create annual measurements dataframe - only where ba is not NaN
    annual_df = df_stakes[df_stakes["ba"].notna()].copy()
    annual_df["FROM_DATE"] = annual_df["prev_yr_min_date"]
    annual_df["TO_DATE"] = annual_df["curr_yr_min_date"]
    annual_df["POINT_BALANCE"] = annual_df["ba"]
    annual_df["PERIOD"] = "annual"
    annual_df["YEAR"] = pd.to_datetime(
        annual_df["curr_yr_max_date"], format="%d.%m.%Y"
    ).dt.year.astype(int)

    # Create winter measurements dataframe - only where bw is not NaN
    winter_df = df_stakes[df_stakes["bw"].notna()].copy()
    winter_df["FROM_DATE"] = winter_df["prev_yr_min_date"]
    winter_df["TO_DATE"] = winter_df["curr_yr_max_date"]
    winter_df["POINT_BALANCE"] = winter_df["bw"]
    winter_df["PERIOD"] = "winter"
    winter_df["YEAR"] = pd.to_datetime(
        winter_df["curr_yr_max_date"], format="%d.%m.%Y"
    ).dt.year.astype(int)
    """
    # Create summer measurements dataframe - only where bs is not NaN
    summer_df = df_stakes[df_stakes['bs'].notna()].copy()
    summer_df['FROM_DATE'] = summer_df['curr_yr_min_date']
    summer_df['TO_DATE'] = summer_df['curr_yr_max_date']
    summer_df['POINT_BALANCE'] = summer_df['bs']
    summer_df['PERIOD'] = 'summer'
    summer_df['YEAR'] = pd.to_datetime(summer_df['curr_yr_max_date'], format='%d.%m.%Y').dt.year.astype(int)
    """
    # Combine both dataframes
    combined_df = pd.concat(
        [annual_df, winter_df], ignore_index=True
    )  # Add, summer_df if needed

    # Select only necessary columns
    columns_to_drop = [
        "bw",
        "bs",
        "ba",
        "prev_yr_min_date",
        "curr_yr_max_date",
        "curr_yr_min_date",
    ]
    result_df = combined_df.drop(columns=columns_to_drop)

    return result_df


def check_period_consistency(df):
    """
    Checks if date ranges make sense for annual and winter periods:
    Returns dataframes with inconsistent periods
    """
    df_check = df.copy()

    # Convert dates to datetime objects
    df_check["FROM_DATE_DT"] = pd.to_datetime(df_check["FROM_DATE"], format="%Y%m%d")
    df_check["TO_DATE_DT"] = pd.to_datetime(df_check["TO_DATE"], format="%Y%m%d")

    df_check["MONTH_DIFF"] = (
        (df_check["TO_DATE_DT"].dt.year - df_check["FROM_DATE_DT"].dt.year) * 12
        + df_check["TO_DATE_DT"].dt.month
        - df_check["FROM_DATE_DT"].dt.month
    )

    # Identify inconsistent periods
    ## 9-15 and 4-9 excludes the normal varying range
    annual_inconsistent = df_check[
        (df_check["PERIOD"] == "annual")
        & ((df_check["MONTH_DIFF"] < 9) | (df_check["MONTH_DIFF"] > 15))
    ]

    winter_inconsistent = df_check[
        (df_check["PERIOD"] == "winter")
        & ((df_check["MONTH_DIFF"] < 4) | (df_check["MONTH_DIFF"] > 9))
    ]

    total_annual = len(df_check[df_check["PERIOD"] == "annual"])
    total_winter = len(df_check[df_check["PERIOD"] == "winter"])

    print(
        f"Annual periods: {len(annual_inconsistent)} out of {total_annual} ({len(annual_inconsistent)/total_annual*100:.1f}%) are inconsistent"
    )
    print(
        f"Winter periods: {len(winter_inconsistent)} out of {total_winter} ({len(winter_inconsistent)/total_winter*100:.1f}%) are inconsistent"
    )

    return annual_inconsistent, winter_inconsistent


def fix_january_to_october_dates(df, annual_inconsistent, winter_inconsistent):
    """
    For any date in the FROM_DATE or TO_DATE columns with month '01',
    change it to '10' for all inconsistent records
    """
    df_fixed = df.copy()

    # Get all indices to check
    all_indices = list(annual_inconsistent.index) + list(winter_inconsistent.index)

    # Count fixes
    fixes_made = 0

    for idx in all_indices:
        # Check and fix FROM_DATE
        from_date = df_fixed.loc[idx, "FROM_DATE"]
        if isinstance(from_date, str) and from_date[4:6] == "01":  # If month is January
            # Replace '01' with '10'
            df_fixed.loc[idx, "FROM_DATE"] = from_date[0:4] + "10" + from_date[6:]
            df_fixed.loc[idx, "DATA_MODIFICATION"] = (
                "Corrected FROM_DATE from January to October"
            )
            fixes_made += 1

        # Check and fix TO_DATE
        to_date = df_fixed.loc[idx, "TO_DATE"]
        if isinstance(to_date, str) and to_date[4:6] == "01":  # If month is January
            # Replace '01' with '10'
            df_fixed.loc[idx, "TO_DATE"] = to_date[0:4] + "10" + to_date[6:]
            df_fixed.loc[idx, "DATA_MODIFICATION"] = (
                "Corrected TO_DATE from January to October"
            )
            fixes_made += 1

    print(
        f"Made {fixes_made} fixes (01 → 10) across {len(all_indices)} inconsistent records"
    )
    return df_fixed


def find_close_stakes(df, distance_threshold=10):
    """
    Find stakes that are within distance_threshold meters of each other.
    Only compares stakes within the same glacier, year, and period.
    """
    close_stakes = []

    # Group by glacier, year, and period
    for (glacier, year, period), group in tqdm(
        df.groupby(["GLACIER", "YEAR", "PERIOD"]),
        desc="Processing glacier-year-periods",
    ):
        # Skip if there's only one stake in this group
        if len(group) <= 1:
            continue

        # Get all unique pairs of stakes in this glacier-year-period
        for idx1, idx2 in combinations(group.index, 2):
            # Get coordinates
            point1 = (group.loc[idx1, "POINT_LAT"], group.loc[idx1, "POINT_LON"])
            point2 = (group.loc[idx2, "POINT_LAT"], group.loc[idx2, "POINT_LON"])

            # Calculate distance in meters
            distance = geodesic(point1, point2).meters

            # Check if within threshold
            if distance <= distance_threshold:
                close_stakes.append(
                    {
                        "GLACIER": glacier,
                        "YEAR": year,
                        "PERIOD": period,
                        "POINT_ID_1": group.loc[idx1, "POINT_ID"],
                        "POINT_ID_2": group.loc[idx2, "POINT_ID"],
                        "LAT_1": point1[0],
                        "LON_1": point1[1],
                        "LAT_2": point2[0],
                        "LON_2": point2[1],
                        "POINT_BALANCE_1": group.loc[idx1, "POINT_BALANCE"],
                        "POINT_BALANCE_2": group.loc[idx2, "POINT_BALANCE"],
                        "DISTANCE_M": distance,
                    }
                )

    # Convert to DataFrame
    if close_stakes:
        result_df = pd.DataFrame(close_stakes)
        print(
            f"Found {len(result_df)} pairs of stakes that are {distance_threshold}m or closer"
        )
        return result_df
    else:
        print(f"No stakes found within {distance_threshold}m of each other")
        return pd.DataFrame(
            columns=[
                "GLACIER",
                "YEAR",
                "PERIOD",
                "POINT_ID_1",
                "POINT_ID_2",
                "LAT_1",
                "LON_1",
                "LAT_2",
                "LON_2",
                "POINT_BALANCE_1",
                "POINT_BALANCE_2",
                "DISTANCE_M",
            ]
        )


def remove_close_points(df_gl, meters=10):
    df_gl_cleaned = pd.DataFrame()
    for year in df_gl.YEAR.unique():
        for period in df_gl.PERIOD.unique():
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
                close_indices = np.where(distance[i, :] < meters)[0]
                close_indices = [idx for idx in close_indices if idx != i]

                if close_indices:
                    mean_MB = df_gl_y.iloc[close_indices + [i]].POINT_BALANCE.mean()

                    # Assign mean balance to the first point
                    df_gl_y.loc[df_gl_y.index[i], "POINT_BALANCE"] = mean_MB

                    # Track which points were merged
                    merged_point_ids = [
                        df_gl_y.iloc[idx]["POINT_ID"] for idx in close_indices
                    ]
                    merged_info = (
                        "Merged with stakes: "
                        + ", ".join(merged_point_ids)
                        + " (taking average MB)"
                    )
                    df_gl_y.loc[df_gl_y.index[i], "DATA_MODIFICATION"] += merged_info

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
    print(f"Number of points dropped: {points_dropped}")
    df_gl_cleaned = df_gl_cleaned.drop(columns=["x", "y"], errors="ignore")
    return df_gl_cleaned if points_dropped > 0 else df_gl


def latlon_to_laea(lat, lon):
    # Define the transformer: WGS84 to ETRS89 / LAEA Europe
    transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:3035")

    # Perform the transformation
    easting, northing = transformer.transform(lat, lon)
    return easting, northing


# --- OGGM --- #


def initialize_oggm_glacier_directories(
    working_dir=None,
    rgi_region="08",
    rgi_version="6",
    base_url="https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5_w_data/",
    log_level="WARNING",
    task_list=None,
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
    oggmCfg.PATHS["working_dir"] = working_dir

    # Get RGI file
    rgi_dir = utils.get_rgi_dir(version=rgi_version)
    path = utils.get_rgi_region_file(region=rgi_region, version=rgi_version)
    rgidf = gpd.read_file(path)

    # Initialize glacier directories from preprocessed data
    gdirs = workflow.init_glacier_directories(
        rgidf,
        from_prepro_level=3,
        prepro_base_url=base_url,
        prepro_border=10,
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


def export_oggm_grids(gdirs, subset_rgis=None, output_path=None):

    # Save OGGM xr for all needed glaciers:
    emptyfolder(output_path)
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
        # save ds
        ds.to_zarr(os.path.join(output_path, f"{RGIId}.zarr"))


def merge_pmb_with_oggm_data(
    df_pmb,
    gdirs,
    rgi_region="08",
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
        # millan_v missing for RGI60-08.01258

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

        # Check which variables are actually available in the dataset, since millan_v is missing in some glaciers
        available_vars = [var for var in variables_of_interest if var in stake]
        if len(available_vars) < len(variables_of_interest):
            missing_vars = set(variables_of_interest) - set(available_vars)
            if verbose:
                log.warning(f"Missing variables for glacier {rgi_id}: {missing_vars}")

        stake_var_df = stake[available_vars].to_dataframe()

        # Assign to original DataFrame
        for var in available_vars:
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
