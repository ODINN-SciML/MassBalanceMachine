import matplotlib.pyplot as plt
import numpy as np
from cartopy import crs as ccrs, feature as cfeature
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import re
import xarray as xr
from matplotlib.colors import to_hex
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import contextily as cx
import geodatasets
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling


def toGeoPandas(ds):
    # Get lat and lon, and variable data
    lat = ds['latitude'].values
    lon = ds['longitude'].values
    variable_data = ds['pred_masked'].values

    # Create meshgrid of coordinates
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Flatten all arrays to match shapes
    lon_flat = lon_grid.flatten()
    lat_flat = lat_grid.flatten()
    variable_data_flat = variable_data.flatten()

    # Verify shapes
    assert len(lon_flat) == len(lat_flat) == len(
        variable_data_flat), "Shapes still don't match!"

    # Create GeoDataFrame
    points = [Point(xy) for xy in zip(lon_flat, lat_flat)]
    gdf = gpd.GeoDataFrame({"data": variable_data_flat},
                           geometry=points,
                           crs="EPSG:4326")
    return gdf, lon, lat


def toRaster(gdf, lon, lat, file_name, source_crs = 'EPSG:4326'):
    # Assuming your GeoDataFrame is named gdf
    # Define the grid dimensions and resolution based on your data
    nrows, ncols = lat.shape[0], lon.shape[
        0]  # Adjust based on your latitude and longitude resolution

    data_array = np.full((nrows, ncols), np.nan)  # Initialize with NaNs

    # Create a raster transformation for the grid (assuming lat/lon range)
    transform = from_origin(min(lon), max(lat), abs(lon[1] - lon[0]),
                            abs(lat[1] - lat[0]))

    # Populate the raster data
    for index, row in gdf.iterrows():
        # Assuming row['geometry'].x and row['geometry'].y give you lon and lat
        lon_idx = np.argmin(np.abs(lon - row.geometry.x))
        lat_idx = np.argmin(np.abs(lat - row.geometry.y))
        data_array[lat_idx, lon_idx] = row['data']

    # Save the raster
    with rasterio.open(
            file_name,
            'w',
            driver='GTiff',
            height=data_array.shape[0],
            width=data_array.shape[1],
            count=1,
            dtype=data_array.dtype,
            crs=source_crs,
            transform=transform,
    ) as dst:
        dst.write(data_array, 1)

    # Read the raster data
    with rasterio.open(file_name) as src:
        raster_data = src.read(1)  # Read the first band
        extent = [
            src.bounds.left, src.bounds.right, src.bounds.bottom,
            src.bounds.top
        ]
    return raster_data, extent
