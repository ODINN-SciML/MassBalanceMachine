import os
import logging
import geopandas as gpd
import pandas as pd
import xarray as xr
from oggm import utils, workflow, tasks
from oggm import cfg as oggmCfg

from regions.TF_Europe.scripts.utils import *
from regions.TF_Europe.scripts.config_TF_Europe import *

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
