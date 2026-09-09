import os
import numpy as np
import xarray as xr
from oggm import workflow, tasks
from oggm import cfg as oggmCfg

import config


def _initialize_oggm_config(custom_working_dir):
    """Initialize OGGM configuration."""
    oggmCfg.initialize(logging_level="WARNING")
    oggmCfg.PARAMS["border"] = 10
    oggmCfg.PARAMS["use_multiprocessing"] = True
    oggmCfg.PARAMS["continue_on_error"] = True
    if len(custom_working_dir) == 0:
        current_path = os.getcwd()
        oggmCfg.PATHS["working_dir"] = os.path.join(current_path, "OGGM")
    else:
        oggmCfg.PATHS["working_dir"] = custom_working_dir


def _initialize_glacier_directories(rgi_ids_list: list, cfg: config.Config) -> list:
    """Initialize glacier directories."""
    base_url = cfg.base_url_w5e5 if cfg.prepro_level >= 3 else cfg.base_url_l2
    glacier_directories = workflow.init_glacier_directories(
        rgi_ids_list,
        reset=False,
        from_prepro_level=cfg.prepro_level,
        prepro_base_url=base_url,
        prepro_border=10,
    )

    workflow.execute_entity_task(
        tasks.gridded_attributes, glacier_directories, print_log=False
    )
    return glacier_directories


def _define_glacier_region_with_dem(gdirs, dem_source, dem_file):
    """Run `tasks.define_glacier_region` on `gdirs` with the requested DEM.

    `dem_source` is the name of a DEM supported by OGGM ('COPDEM30', 'COPDEM90',
    'SRTM', ...). `dem_file` optionally overrides it with a user-supplied raster:
    either a single path applied to every glacier, or a {glacier_id: path} mapping.
    OGGM reads the user raster through the *global* `cfg.PATHS['dem_file']`, so a
    per-glacier mapping has to be applied one directory at a time.
    """
    if dem_file is None:
        workflow.execute_entity_task(
            tasks.define_glacier_region, gdirs, source=dem_source
        )
        return

    prev_dem_file = oggmCfg.PATHS.get("dem_file", "")
    try:
        if isinstance(dem_file, dict):
            missing = [gdir.rgi_id for gdir in gdirs if gdir.rgi_id not in dem_file]
            assert not missing, (
                "A per-glacier DEM mapping was given but it has no entry for "
                f"{missing}. Provide a DEM for every glacier or pass a single path."
            )
            # One glacier at a time: cfg.PATHS['dem_file'] is global, so a parallel
            # run would apply whichever path was set last to all of them.
            for gdir in gdirs:
                oggmCfg.PATHS["dem_file"] = dem_file[gdir.rgi_id]
                tasks.define_glacier_region(gdir, source="USER")
        else:
            oggmCfg.PATHS["dem_file"] = dem_file
            workflow.execute_entity_task(
                tasks.define_glacier_region, gdirs, source="USER"
            )
    finally:
        oggmCfg.PATHS["dem_file"] = prev_dem_file


def _initialize_custom_glacier_directories(
    df,
    splitdf,
    dem_source: str = "COPDEM30",
    dem_file=None,
    reset: bool = True,
    id_column: str = "RGIId",
) -> list:
    """Build OGGM glacier directories from a custom (non-RGI) inventory.

    Args:
        df: RGI-v7-shaped frame of dissolved outlines, one row per glacier.
        splitdf: the *un-dissolved* frame. A glacier of a custom inventory is often
            stored as several polygons (isolated blocks of ice); `df` carries their
            convex hull so that the OGGM grid covers them all, and the true mask is
            rebuilt here from every polygon of `splitdf`.
        dem_source: name of a DEM supported by OGGM ('COPDEM30', 'SRTM', ...).
        dem_file: optional user-supplied raster overriding `dem_source`, either one
            path for every glacier or a {glacier_id: path} mapping.
        reset: wipe and rebuild the directories. When False, glaciers that already
            have a `gridded_data` file are left alone, so a persistent working
            directory does not re-download every DEM on each run.
        id_column: the column of `splitdf` holding the glacier id.
    """
    oggmCfg.PARAMS["use_rgi_area"] = False  # recompute area from geometry
    oggmCfg.PARAMS["use_intersects"] = False
    # oggmCfg.PARAMS['border'] = 10

    gdirs = workflow.init_glacier_directories(
        df,
        reset=reset,
        force=reset,
    )

    todo = gdirs if reset else [g for g in gdirs if not g.has_file("gridded_data")]
    if not todo:
        return gdirs

    # Define the local map projection and get the DEM
    _define_glacier_region_with_dem(todo, dem_source, dem_file)

    # Compute glacier masks, slope, and aspect
    workflow.execute_entity_task(tasks.glacier_masks, todo)

    workflow.execute_entity_task(tasks.gridded_attributes, todo)

    for gdir in todo:
        rgi_id = gdir.rgi_id
        with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
            ds = ds.load()
        mask = np.zeros_like(ds.glacier_mask.values)
        true_geom = splitdf[splitdf[id_column] == rgi_id]
        for i in range(true_geom.shape[0]):
            out = gdir.grid.region_of_interest(geometry=true_geom.iloc[i].geometry)
            mask = mask | out
        ds["glacier_mask"].values = mask
        ds = ds.drop(
            [
                "glacier_ext",
                "glacier_ext_erosion",
                "ice_divides",
                "dis_from_border",
                "topo_valid_mask",
            ]
        )
        # Overwrite netcdf
        save_path = gdir.get_filepath("gridded_data")
        ds.to_netcdf(save_path)

    return gdirs


def _glacier_name(rgi_ids_list: list, cfg: config.Config, custom_working_dir=""):

    # Initialize the OGGM Config
    _initialize_oggm_config(custom_working_dir)
    glacier_directories = _initialize_glacier_directories(rgi_ids_list, cfg)
    return {gdir.rgi_id: gdir.name for gdir in glacier_directories}
