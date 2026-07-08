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


def _initialize_custom_glacier_directories(df, splitdf) -> list:
    oggmCfg.PARAMS["use_rgi_area"] = False  # recompute area from geometry
    oggmCfg.PARAMS["use_intersects"] = False
    # oggmCfg.PARAMS['border'] = 10

    gdirs = workflow.init_glacier_directories(
        df,
        reset=True,
        force=True,
    )

    # Define the local map projection and download the DEM
    # You can pass source='COPDEM90', source='COPDEM30' or any other supported DEM
    workflow.execute_entity_task(tasks.define_glacier_region, gdirs, source="COPDEM30")

    # Compute glacier masks, slope, and aspect
    workflow.execute_entity_task(tasks.glacier_masks, gdirs)

    workflow.execute_entity_task(tasks.gridded_attributes, gdirs)

    for gdir in gdirs:
        rgi_id = gdir.rgi_id
        with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
            ds = ds.load()
        mask = np.zeros_like(ds.glacier_mask.values)
        true_geom = splitdf[splitdf.RGIId == rgi_id]
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
