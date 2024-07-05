"""
This script fetches the topographical features (variables of interest), for each stake measurement available,
via the OGGM library.

For now the best way of running this script is in its own environment, please see the README for instructions.

When running this script please provide the RGI region ID via the terminal. The RGI region IDs can be found here:
https://rgitools.readthedocs.io/en/latest/dems.html#global-data-availability

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 04/06/2024
"""

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
# noinspection PyUnresolvedReferences
from oggm import cfg, utils, workflow, tasks, graphics
from pathlib import Path
import os
import logging


def get_topo_features(df, output_fname, voi, rgi_ids):
    rgi_ids_list = rgi_ids.dropna().unique().tolist()

    # Initialize OGGM configuration
    cfg.initialize(logging_level='WARNING')
    cfg.PARAMS['border'] = 10
    cfg.PARAMS['use_multiprocessing'] = True
    cfg.PARAMS['continue_on_error'] = True

    # Define workspace path to store OGGM data
    current_path = os.getcwd()
    workspace_path = os.path.join(current_path, 'OGGM')

    # Set working directory for OGGM
    cfg.PATHS['working_dir'] = workspace_path

    # Initialize glacier directories
    base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5_w_data/'
    g_dirs = workflow.init_glacier_directories(rgi_ids_list, from_prepro_level=3, prepro_base_url=base_url, prepro_border=10)

    # List of tasks to execute
    task_list = [tasks.gridded_attributes]
    # Execute tasks for each glacier directory
    for task in task_list:
        workflow.execute_entity_task(task, g_dirs, print_log=True)

    # Add columns to dataframe
    df[voi] = np.nan

    # Extract latitude and longitude from dataframe
    lat, lon = df[df[rgi_ids.name].isin(rgi_ids_list)][['POINT_LAT', 'POINT_LON']].values.T

    # Load gridded data for each glacier directory
    g_dirs_dataset = [xr.open_dataset(gdir.get_filepath('gridded_data')).load() for gdir in g_dirs]

    # Convert latitude and longitude to xarray DataArrays
    lon_da = xr.DataArray(lon, dims='points')
    lat_da = xr.DataArray(lat, dims='points')

    # Find nearest stake for each glacier directory
    stakes = [gdir_da.sel(x=lon, y=lat, method='nearest') for gdir_da, lon, lat in zip(g_dirs_dataset, lon_da, lat_da)]

    # Convert stake data to pandas DataFrame
    stakes = [stake[voi].to_pandas() for stake in stakes]

    # Update dataframe with stake data
    for stake_data, rgi_id in zip(stakes, rgi_ids_list):
        rgi_records = df[rgi_ids == rgi_id]
        df.loc[rgi_records.index, voi] = stake_data[voi].values

    # Write updated dataframe to CSV file
    df.to_csv(output_fname, index=False)

    return df

