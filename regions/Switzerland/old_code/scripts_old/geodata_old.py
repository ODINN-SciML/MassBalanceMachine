import os
import re
import glob
from os.path import isfile, join
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rasterio
import pyproj
from scipy.ndimage import gaussian_filter
from dateutil.relativedelta import relativedelta
from tqdm.notebook import tqdm
import warnings
from rasterio.transform import from_bounds

# Project-specific imports
from regions.Switzerland.scripts.config_CH import *
from regions.Switzerland.scripts.wgs84_ch1903 import *


def LV03toLV95(df):
    """Convert Swiss LV03 (EPSG:21781) coordinates to LV95 (EPSG:2056)."""
    transformer = pyproj.Transformer.from_crs("EPSG:21781", "EPSG:2056", always_xy=True)
    x_lv95, y_lv95 = transformer.transform(df.x_pos, df.y_pos)
    df["x_lv95"] = x_lv95
    df["y_lv95"] = y_lv95
    df.drop(["x_pos", "y_pos", "z_pos"], axis=1, inplace=True)
    return df


def transform_xarray_coords_lv03_to_lv95(data_array):
    # Extract and flatten x and y coordinates
    y_coords, x_coords = np.meshgrid(
        data_array.y.values, data_array.x.values, indexing="ij"
    )
    flattened_x = x_coords.flatten()
    flattened_y = y_coords.flatten()
    flattened_values = data_array.values.flatten()

    # Create DataFrame
    df = pd.DataFrame(
        {"x_pos": flattened_x, "y_pos": flattened_y, "value": flattened_values}
    )
    df["z_pos"] = 0  # dummy height

    # Convert from LV03 to LV95
    df = LV03toLV95(df)

    # Reshape coordinates back to 2D
    x_lv95 = df.x_lv95.values.reshape(x_coords.shape)
    y_lv95 = df.y_lv95.values.reshape(y_coords.shape)

    # 1D coordinates to assign back to xarray
    x_lv95_1d = x_lv95[0, :]  # Eastings
    y_lv95_1d = y_lv95[:, 0]  # Northings

    # Assign new LV95 coordinates and swap dims
    data_array = data_array.assign_coords(
        x_lv95=("x", x_lv95_1d), y_lv95=("y", y_lv95_1d)
    )
    data_array = data_array.swap_dims({"x": "x_lv95", "y": "y_lv95"})

    return data_array


def save_xarray_to_grid(data_array, filepath, nodata_value=-9999):
    """
    Save an xarray.DataArray to a .grid (ASCII raster) file.

    Parameters:
    - data_array: xarray.DataArray with 2D shape (y, x)
    - filepath: Path to save the .grid file
    - nodata_value: Value to use for NaNs
    """

    # Ensure it's 2D
    if data_array.ndim != 2:
        raise ValueError("Only 2D DataArrays are supported.")

    # Extract coordinates and data
    values = data_array.values
    values = np.where(np.isnan(values), nodata_value, values)

    nrows, ncols = values.shape
    x = data_array.coords[data_array.dims[1]].values  # x
    y = data_array.coords[data_array.dims[0]].values  # y

    cellsize_x = np.abs(x[1] - x[0])
    cellsize_y = np.abs(y[1] - y[0])

    if not np.allclose(cellsize_x, cellsize_y):
        raise ValueError("Non-square pixels are not supported in .grid format.")

    cellsize = cellsize_x

    xllcorner = x.min() if x[1] > x[0] else x.max() - (ncols - 1) * cellsize
    yllcorner = y.min() if y[1] > y[0] else y.max() - (nrows - 1) * cellsize

    # Write header + data
    with open(filepath, "w") as f:
        f.write(f"ncols         {ncols}\n")
        f.write(f"nrows         {nrows}\n")
        f.write(f"xllcorner     {xllcorner:.6f}\n")
        f.write(f"yllcorner     {yllcorner:.6f}\n")
        f.write(f"cellsize      {cellsize:.6f}\n")
        f.write(f"NODATA_value  {nodata_value}\n")

        for row in values[::-1]:  # Flip vertically
            f.write(" ".join(f"{val:.6f}" for val in row) + "\n")


def organize_rasters_by_hydro_year(path_S2, satellite_years):
    rasters = defaultdict(
        lambda: defaultdict(list)
    )  # Nested dictionary for years and months

    for year in satellite_years:
        folder_path = os.path.join(path_S2, str(year))
        for f in os.listdir(folder_path):
            if f.endswith(".tif"):  # Only process raster files
                # Step 1: Extract the date from the string
                date_str = f.split("_")[3][:8]  # Extract the 8-digit date (YYYYMMDD)
                file_date = datetime.strptime(
                    date_str, "%Y%m%d"
                )  # Convert to datetime object

                closest_month, hydro_year = get_hydro_year_and_month(file_date)
                if hydro_year < 2022:
                    rasters[hydro_year][closest_month].append(f)

    return rasters


def get_hydro_year_and_month(file_date):
    if file_date.day < 15:
        # Move to the first day of the previous month
        file_date -= relativedelta(months=1)  # Move to the previous month
        file_date = file_date.replace(
            day=1
        )  # Set the day to the 1st of the previous month
    else:
        # Move to the first day of the current month
        file_date = file_date.replace(
            day=1
        )  # Set the day to the 1st of the current month

    # Step 2: Determine the closest month
    closest_month = file_date.strftime("%b").lower()  # Get the full name of the month

    # Step 3: Determine the hydrological year
    # Hydrological year runs from September to August
    if file_date.month >= 9:  # September, October, November, December
        hydro_year = file_date.year + 1  # Assign to the next year
    else:  # January to August
        hydro_year = file_date.year  # Assign to the current year

    return closest_month, hydro_year


def IceSnowCover(gdf_class, gdf_class_raster):
    # Exclude pixels with "classes" 5 (cloud) in gdf_class_raster
    valid_classes = gdf_class[gdf_class_raster.classes != 5]

    # Calculate percentage of snow cover (class 1) in valid classes
    snow_cover_glacier = (
        valid_classes.classes[valid_classes.classes == 1].count()
        / valid_classes.classes.count()
    )

    return snow_cover_glacier


def coarsenDS_mercator(ds, target_res_m=50):
    # Get dx, dy directly from coordinates (assumes regular grid)
    x = ds["x"]
    y = ds["y"]
    dx_m = abs(float(x[1] - x[0]))
    dy_m = abs(float(y[1] - y[0]))

    # Compute coarsening factor
    resampling_fac_x = max(1, round(target_res_m / dx_m))
    resampling_fac_y = max(1, round(target_res_m / dy_m))

    if dx_m < target_res_m or dy_m < target_res_m:
        list_vars = [var for var in list(ds.data_vars) if "masked" in var]

        # Coarsen non-binary variables with mean
        ds_non_binary = (
            ds[list_vars]
            .coarsen(x=resampling_fac_x, y=resampling_fac_y, boundary="trim")
            .mean()
        )

        # Coarsen glacier mask with max
        ds_glacier_mask = (
            ds[["glacier_mask"]]
            .coarsen(x=resampling_fac_x, y=resampling_fac_y, boundary="trim")
            .reduce(np.max)
        )

        # Merge and return
        ds_res = xr.merge([ds_non_binary, ds_glacier_mask])
        return ds_res

    return ds


def get_res_from_projected(ds):
    """
    Computes resolution in meters for projected xarray.Dataset.
    Assumes regular grid and 'x' and 'y' coordinate names.
    Returns (dx, dy) in meters.
    """
    x = ds.coords["x"]
    y = ds.coords["y"]

    # Use absolute value to avoid negative spacing (inverted axes)
    dx = abs(float(x[1] - x[0]))
    dy = abs(float(y[1] - y[0]))

    return dx, dy


def create_glacier_grid_RGI(ds: xr.Dataset, years: list, rgi_gl: str):
    mask = ds["glacier_mask"].astype(bool)
    ds_masked = ds.where(mask)

    base_cols = ["masked_aspect", "masked_slope", "masked_elev"]
    opt_cols = [
        v for v in ["masked_hug", "masked_cit", "masked_miv", "svf"] if v in ds_masked
    ]
    # print(f"Available optional columns: {opt_cols}")
    cols = base_cols + opt_cols

    df_grid = (
        ds_masked[cols]
        .to_dataframe()
        .dropna(how="all")  # drop rows where all selected vars are NaN
        .reset_index()
        .rename(
            columns={
                "lat": "POINT_LAT",
                "lon": "POINT_LON",
                "masked_aspect": "aspect",
                "masked_slope": "slope",
                "masked_elev": "topo",
                "masked_hug": "hugonnet_dhdt",
                "masked_cit": "consensus_ice_thickness",
                "masked_miv": "millan_v",
                "svf": "svf",
            }
        )
    )
    df_grid["RGIId"] = rgi_gl

    # Match to WGMS format:
    df_grid["POINT_ID"] = np.arange(1, len(df_grid) + 1)
    df_grid["N_MONTHS"] = 12
    df_grid["POINT_ELEVATION"] = df_grid["topo"]  # no other elevation available
    df_grid["POINT_BALANCE"] = 0  # fake PMB for simplicity (not used)
    num_rows_per_year = len(df_grid)

    # Repeat the DataFrame num_years times
    df_grid = pd.concat([df_grid] * len(years), ignore_index=True)

    # Add the 'year' and date columns to the DataFrame
    df_grid["YEAR"] = np.repeat(
        years, num_rows_per_year
    )  # 'year' column that has len(df_grid) instances of year
    df_grid["FROM_DATE"] = df_grid["YEAR"].apply(lambda x: str(x) + "1001")
    df_grid["TO_DATE"] = df_grid["YEAR"].apply(lambda x: str(x + 1) + "0930")
    df_grid["PERIOD"] = "annual"

    # print(df_grid.columns)

    return df_grid
