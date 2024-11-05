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
from rasterio.merge import merge
import glob
from scipy.ndimage import gaussian_filter

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

def reproject_raster_to_lv95(input_raster, output_raster):
    # Define the source and destination CRS
    src_crs = 'EPSG:4326'  # Original CRS (lat/lon)
    dst_crs = 'EPSG:2056'  # Destination CRS (Swiss LV95)

    # Open the source raster
    with rasterio.open(input_raster) as src:
        # Calculate the transform and dimensions for the destination CRS
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        # Set up the destination raster metadata
        dst_meta = src.meta.copy()
        dst_meta.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        # Perform the reprojection
        with rasterio.open(output_raster, 'w', **dst_meta) as dst:
            for i in range(1, src.count + 1):  # Iterate over each band
                # reproject(
                #     source=rasterio.band(src, i),
                #     destination=rasterio.band(dst, i),
                #     src_transform=src.transform,
                #     src_crs=src.crs,
                #     dst_transform=transform,
                #     dst_crs=dst_crs,
                #     resampling=Resampling.nearest  # You can also use other methods, like bilinear
                # )
                # Create an array to hold the reprojected data
                data = np.empty((height, width), dtype=src.meta['dtype'])

                reproject(
                    source=rasterio.band(src, i),
                    destination=data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest  # You can also use other methods, like bilinear
                )

                # Replace any 0 values in `data` with NaN
                data[data == 0] = np.nan

                # Write the modified data to the output raster band
                dst.write(data, i)
    
            
def merge_rasters(raster1_path, raster2_path, output_path):
    # Open the rasters
    raster_files = [raster1_path, raster2_path]
    src_files_to_mosaic = [rasterio.open(fp) for fp in raster_files]
    
    # Merge the rasters
    mosaic, out_transform = merge(src_files_to_mosaic)
    
    # Update the metadata to match the mosaic output
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "crs": src_files_to_mosaic[0].crs
    })
    # replace 0 values with NaN
    mosaic[mosaic == 0] = np.nan
    
    # Write the mosaic raster to disk
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Close all source files
    for src in src_files_to_mosaic:
        src.close()
        
        
def GaussianFilter(ds):
    # Assuming `dataset` is your xarray.Dataset
    sigma = 1  # Define the smoothing level

    # Apply Gaussian filter to each DataArray in the Dataset
    smoothed_dataset = xr.Dataset()

    for var_name, data_array in ds.data_vars.items():
        # Step 1: Create a mask of valid data (non-NaN values)
        mask = ~np.isnan(data_array)

        # Step 2: Replace NaNs with zero (or a suitable neutral value)
        filled_data = data_array.fillna(0)

        # Step 3: Apply Gaussian filter to the filled data
        smoothed_data = gaussian_filter(filled_data, sigma=sigma)

        # Step 4: Restore NaNs to their original locations
        smoothed_data = xr.DataArray(smoothed_data,
                                    dims=data_array.dims,
                                    coords=data_array.coords,
                                    attrs=data_array.attrs).where(
                                        mask)  # Apply the mask to restore NaNs

        # Add the smoothed data to the new Dataset
        smoothed_dataset[var_name] = smoothed_data
    return smoothed_dataset