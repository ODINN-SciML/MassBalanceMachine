import pandas as pd
import numpy as np
import pyproj
import xarray as xr
import geopandas as gpd
import math
import logging
from shapely.geometry import Point
from oggm import utils, workflow, tasks
from oggm import cfg as oggmCfg

from regions.Iceland.scripts.config_ICE import *
from regions.Iceland.scripts.helpers import *

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


# --- Preprocess --- #


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
    annual_df = df_stakes[df_stakes["ba_floating_date"].notna()].copy()
    annual_df["FROM_DATE"] = annual_df["d1"]
    annual_df["TO_DATE"] = annual_df["d3"]
    annual_df["POINT_BALANCE"] = annual_df["ba_floating_date"]
    annual_df["PERIOD"] = "annual"
    annual_df["YEAR"] = annual_df["yr"]

    # Create winter measurements dataframe - only where bw is not NaN
    winter_df = df_stakes[df_stakes["bw_floating_date"].notna()].copy()
    winter_df["FROM_DATE"] = winter_df["d1"]
    winter_df["TO_DATE"] = winter_df["d2"]
    winter_df["POINT_BALANCE"] = winter_df["bw_floating_date"]
    winter_df["PERIOD"] = "winter"
    winter_df["YEAR"] = annual_df["yr"]
    """
    # Create summer measurements dataframe - only where bs is not NaN
    summer_df = df_stakes[df_stakes['bs_floating_date'].notna()].copy()
    summer_df['FROM_DATE'] = summer_df['d2']
    summer_df['TO_DATE'] = summer_df['d3']
    summer_df['POINT_BALANCE'] = summer_df['bs_floating_date']
    summer_df['PERIOD'] = 'summer'
    summer_df['YEAR'] = annual_df['yr']
    """
    # Combine both dataframes
    combined_df = pd.concat(
        [annual_df, winter_df], ignore_index=True
    )  # Add ", summer_df" if needed

    # Select only necessary columns
    columns_to_drop = [
        "ba_floating_date",
        "ba_stratigraphic",
        "bs_floating_date",
        "bs_stratigraphic",
        "bw_floating_date",
        "bw_stratigraphic",
        "d1",
        "d2",
        "d3",
        "ds",
        "dw",
        "fall_elevation",
        "ice_melt_fall",
        "ice_melt_spring",
        "snow_melt_fall",
        "rhos",
        "rhow",
        "yr",
        "nswe_fall",
        "swes",
        "swew",
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

    # Calculate month difference
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


# --- OGGM --- #


def initialize_oggm_glacier_directories(
    working_dir=None,
    rgi_region="06",  # Iceland
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
    rgi_region="06",
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
        # "hugonnet_dhdt" is missing for approx. 10% of the dataset
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


def flag_elevation_mismatch(df, threshold=400):
    """
    Flag rows where POINT_ELEVATION differs from DEM elevation ('topo')
    by more than a given threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'POINT_ELEVATION' and 'topo'.
    threshold : float, optional
        Maximum allowed absolute elevation difference (meters).
        Default is 400 m.

    Returns
    -------
    df_out : pandas.DataFrame
        Copy of input dataframe with:
        - 'elev_diff' : POINT_ELEVATION - topo
        - 'elev_mismatch' : True if abs(diff) > threshold
    mismatches : pandas.DataFrame
        Subset of rows where mismatch is True.
    """
    df_out = df.copy()

    df_out["elev_diff"] = df_out["POINT_ELEVATION"] - df_out["topo"]
    df_out["elev_mismatch"] = df_out["elev_diff"].abs() > threshold

    mismatches = df_out[df_out["elev_mismatch"]]

    print(
        f"{len(mismatches)} out of {len(df_out)} points "
        f"({len(mismatches)/len(df_out)*100:.2f}%) exceed ±{threshold} m elevation difference."
    )

    return df_out, mismatches
