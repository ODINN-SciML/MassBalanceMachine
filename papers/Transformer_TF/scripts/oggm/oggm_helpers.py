import os
import math
import logging
import numpy as np
import pyproj
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import xarray as xr
from oggm import utils, workflow, tasks
from oggm import cfg as oggmCfg

from papers.Transformer_TF.scripts.utils import *
from papers.Transformer_TF.scripts.config_TF_Europe import *

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


# ------------- HELPER FUNCTIONS ----------------- #
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
    rgi_ids_list=None,
):
    """
    Initialize OGGM GlacierDirectories from preprocessed data and run a task list.

    Parameters
    ----------
    cfg : object
        Configuration object with attribute `dataPath`.
    working_dir : str or None, optional
        OGGM working directory. If None, uses `<dataPath>/<path_OGGM>` and empties it.
    rgi_region : str, optional
        RGI region string (default "11").
    rgi_version : str, optional
        RGI version used by OGGM utilities (default "62").
    base_url : str, optional
        URL to preprocessed OGGM directories (L3-L5 files).
    log_level : str, optional
        OGGM logging level.
    task_list : list, optional
        List of OGGM tasks to execute per glacier.
    from_prepro_level : int, optional
        OGGM prepro level to load.
    prepro_border : int, optional
        Border size for preprocessed directories.

    Returns
    -------
    tuple
        (gdirs, rgidf) where:
        - gdirs : list of oggm.GlacierDirectory
        - rgidf : geopandas.GeoDataFrame with RGI outlines/attributes

    Side Effects
    ------------
    Sets OGGM config and empties/creates working directory.
    """
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
        working_dir = os.path.join(cfg.dataPath, "OGGM", f"rgi_region_{rgi_region}")

    # empty the working directory if it exists
    log.info("Emptying working directory: ", working_dir)
    emptyfolder(working_dir)
    oggmCfg.PATHS["working_dir"] = working_dir
    log.info("Collecting from base_url: ", base_url)

    # Get RGI file
    if rgi_ids_list is None:
        rgi_dir = utils.get_rgi_dir(version=rgi_version, reset=False)
        path = utils.get_rgi_region_file(
            region=rgi_region, version=rgi_version, reset=False
        )
        rgidf = gpd.read_file(path)
        # Initialize glacier directories from preprocessed data
        gdirs = workflow.init_glacier_directories(
            rgidf,
            from_prepro_level=from_prepro_level,
            prepro_base_url=base_url,
            prepro_border=prepro_border,
            reset=True,
            force=True,
        )
        rgi_ids_list = rgidf["RGIId"].tolist()
    else:
        # Initialize glacier directories from preprocessed data
        gdirs = workflow.init_glacier_directories(
            rgi_ids_list,
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

    return gdirs, rgi_ids_list


def export_oggm_grids(cfg, gdirs, subset_rgis=None, output_path=None, rgi_region="11"):
    """
    Export OGGM gridded_data datasets to per-glacier Zarr files and report missing variables.

    Parameters
    ----------
    cfg : object
        Configuration object with attribute `dataPath`.
    gdirs : list
        OGGM GlacierDirectory objects.
    subset_rgis : set or list or None, optional
        If provided, only export glaciers whose RGIId is in this subset.
    output_path : str or None, optional
        Output folder for Zarr files. Defaults to `<dataPath>/<path_OGGM_xrgrids>`.

    Returns
    -------
    pandas.DataFrame
        Table listing glaciers with missing expected variables and which variables are missing.

    Side Effects
    ------------
    Empties output folder and writes Zarr datasets.
    """

    # Save OGGM xr for all needed glaciers:
    if output_path is None:
        output_path = os.path.join(
            cfg.dataPath, "OGGM", f"rgi_region_{rgi_region}", "xr_grids"
        )
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
    rgi_version="62",
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
