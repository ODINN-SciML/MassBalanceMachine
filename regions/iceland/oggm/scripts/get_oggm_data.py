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

from argparse import ArgumentParser

# Argument for a single column to be dropped from the dataset
# parser = ArgumentParser()
# parser.add_argument('-RGI', '--RGI', required=True, help='Provide the RGI region ID of the region of interest', type=str)
#
# args = parser.parse_args()
# RGI = args.RGI

RGI = '06'

if not RGI:
    raise ValueError("No RGI ID region provided. Please provide a RGI ID region for the region of interest.")

# Set pandas option to display all columns
pd.set_option('display.max_columns', None)

# Initialize OGGM configuration
cfg.initialize(logging_level='WARNING')
cfg.PARAMS['border'] = 10
cfg.PARAMS['use_multiprocessing'] = True
cfg.PARAMS['continue_on_error'] = True

# Initialize logger
log = logging.getLogger('.'.join(__name__.split('.')[:-1]))

# Define workspace path to store OGGM data
parent_path = '.././'
workspace_path = '/data/oggm'

if not os.path.exists(workspace_path):
    os.makedirs(workspace_path)

# Define path to stake data CSV file
df_path = os.path.join(parent_path, 'mbm/data/files/Iceland_Stake_Data_Reprojected.csv')
df_path_output = os.path.join(parent_path, 'mbm/data/files/Iceland_Stake_Data_T_Attributes.csv')

# Read stake data CSV file
df = pd.read_csv(df_path)

# Get unique RGI IDs
rgi_ids = df['RGIId'].unique().tolist()

# Set working directory for OGGM
cfg.PATHS['working_dir'] = workspace_path

# Define RGI region and version
rgi_region = RGI
rgi_version = '6'
# Get RGI region file
path = utils.get_rgi_region_file(region=rgi_region, version=rgi_version)
rgidf = gpd.read_file(path)

# Initialize glacier directories
base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5_w_data/'
gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=3, prepro_base_url=base_url, prepro_border=10, reset=True, force=True)

# List of tasks to execute
task_list = [tasks.gridded_attributes]
# Execute tasks for each glacier directory
for task in task_list:
    workflow.execute_entity_task(task, gdirs, print_log=False)

# Variables of interest
voi = ['topo', 'aspect', 'slope', 'slope_factor', 'dis_from_border']

# Add columns to dataframe
df[voi] = np.nan

# Extract latitude and longitude from dataframe
lat, lon = df[df['RGIId'].isin(rgi_ids)][['lat', 'lon']].values.T

# Load gridded data for each glacier directory
gdirs_da = [xr.open_dataset(gdir.get_filepath('gridded_data')).load() for gdir in gdirs]

# Convert latitude and longitude to xarray DataArrays
lon_da = xr.DataArray(lon, dims='points')
lat_da = xr.DataArray(lat, dims='points')

# Find nearest stake for each glacier directory
stakes = [gdir_da.sel(x=lon, y=lat, method='nearest') for gdir_da, lon, lat in zip(gdirs_da, lon_da, lat_da)]

# Save all the areas of the associated glacier
areas = [(gdir.rgi_area_km2, gdir.rgi_id) for gdir in gdirs]

# Convert stake data to pandas DataFrame
stakes = [stake[voi].to_pandas() for stake in stakes]

# Update dataframe with stake data
for stake_data, rgi_id in zip(stakes, rgi_ids):
    rgi_records = df[df['RGIId'] == rgi_id]
    df.loc[rgi_records.index, voi] = stake_data[voi].values

# Write updated dataframe to CSV file
df.to_csv(df_path_output, index=False)

# Save the areas for each RGI in a separate CSV file
areas_rgiids = pd.DataFrame(areas, columns=['area', 'rgiid'])
areas_rgiids.to_csv(os.path.join(parent_path, 'mbm/data/files/areas_rgiids.csv'), index=False)
