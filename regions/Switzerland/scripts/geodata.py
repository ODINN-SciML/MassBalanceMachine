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


def get_glwd_glamos_years(cfg, glacier_name):
    folder = os.path.join(
        cfg.dataPath, path_distributed_MB_glamos, "GLAMOS", glacier_name
    )
    if not os.path.exists(folder):
        print(f"Warning: Folder {folder} does not exist.")
        return []

    # Match: 2005_ann_fix_lv95.grid OR 2005_ann_fix_lv03.grid
    pattern = re.compile(r"^(\d{4})_ann_fix_lv(?:95|03)\.grid$")

    years = []
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            years.append(int(match.group(1)))  # Extract the year as an integer

    years = np.unique(years).tolist()
    years.sort()
    return years


def get_GLAMOS_glwmb(glacier_name, cfg):
    """
    Loads and processes GLAMOS glacier-wide mass balance data.

    Parameters:
    -----------
    glacier_name : str
        The name of the glacier.
    cfg : mbm.Config
        Configuration instance.

    Returns:
    --------
    pd.DataFrame or None
        A DataFrame with columns ['YEAR', 'GLAMOS Balance'] indexed by 'YEAR',
        or None if the file is missing.
    """
    years = get_glwd_glamos_years(cfg, glacier_name)
    if years == []:
        print(f"Warning: No GLAMOS data found for {glacier_name}.")

    def pick_ann_file(cfg, glacier_name, year):
        base = os.path.join(
            cfg.dataPath, path_distributed_MB_glamos, "GLAMOS", glacier_name
        )
        cand_lv95 = os.path.join(base, f"{year}_ann_fix_lv95.grid")
        cand_lv03 = os.path.join(base, f"{year}_ann_fix_lv03.grid")
        if os.path.exists(cand_lv95):
            return cand_lv95, "lv95"
        if os.path.exists(cand_lv03):
            return cand_lv03, "lv03"
        return None, None

    glamos_glwd_mb = []
    for year in years:
        grid_path_ann, proj = pick_ann_file(cfg, glacier_name, year)
        if grid_path_ann is None:
            warnings.warn(
                f"No ann file found for {glacier_name} {year} (lv95/lv03). Skipping."
            )
            continue

        metadata_ann, grid_data_ann = load_grid_file(grid_path_ann)
        ds_glamos_ann = convert_to_xarray_geodata(grid_data_ann, metadata_ann)

        if proj == "lv03":
            ds_glamos_wgs84_ann = transform_xarray_coords_lv03_to_wgs84(ds_glamos_ann)
        elif proj == "lv95":
            ds_glamos_wgs84_ann = transform_xarray_coords_lv95_to_wgs84(ds_glamos_ann)
        else:
            raise RuntimeError(f"Unknown projection for {grid_path_ann}")

        glamos_glwd_mb.append(float(ds_glamos_wgs84_ann.mean().values))

    df = pd.DataFrame({"GLAMOS Balance": glamos_glwd_mb, "YEAR": years})

    # set index years
    df.set_index("YEAR", inplace=True)
    return df


def apply_gaussian_filter(obj, variable_name="pred_masked", sigma: float = 1):
    """
    Apply Gaussian smoothing to either:
      - an xarray.Dataset + variable name
      - an xarray.DataArray directly

    Returns the same type as provided (Dataset or DataArray).
    """

    # --- Case 1: Input is a Dataset ---
    if isinstance(obj, xr.Dataset):

        if variable_name is None:
            raise ValueError("variable_name must be provided when passing a Dataset.")

        data_array = obj[variable_name]
        is_dataset = True

    # --- Case 2: Input is a DataArray ---
    elif isinstance(obj, xr.DataArray):
        data_array = obj
        is_dataset = False

    else:
        raise TypeError("Input must be an xarray.Dataset or xarray.DataArray.")

    # Step 1: mask of valid data
    mask = ~np.isnan(data_array)

    # Step 2: replace NaNs
    filled_data = data_array.fillna(0)

    # Step 3: Gaussian filter
    smoothed_np = gaussian_filter(filled_data.data, sigma=sigma)

    # Step 4: restore coords / dims / attrs + mask NaNs back
    smoothed_da = xr.DataArray(
        smoothed_np,
        dims=data_array.dims,
        coords=data_array.coords,
        attrs=data_array.attrs,
    ).where(mask)

    # --- Return in the same structure as input ---
    if is_dataset:
        out = obj.copy()
        out[variable_name] = smoothed_da
        return out
    else:
        return smoothed_da


def convert_to_xarray_geodata(grid_data, metadata):
    """Converts .grid file data to an xarray DataArray.

    Args:
        grid_data (.grid): grid file of glacier DEM
        metadata (dic): metadata of the grid file

    Returns:
        xr.DataSet: xarray DataArray of the grid data in LV95 coordinates
    """

    # Extract metadata values
    ncols = int(metadata["ncols"])
    nrows = int(metadata["nrows"])
    xllcorner = metadata["xllcorner"]
    yllcorner = metadata["yllcorner"]
    cellsize = metadata["cellsize"]

    # Create x and y coordinates based on the metadata
    x_coords = xllcorner + np.arange(ncols) * cellsize
    y_coords = yllcorner + np.arange(nrows) * cellsize

    # Create the xarray DataArray
    data_array = xr.DataArray(
        np.flip(grid_data, axis=0),
        dims=("y", "x"),
        coords={"y": y_coords, "x": x_coords},
        name="grid_data",
    )
    return data_array


def transform_xarray_coords_lv95_to_wgs84(data_array):
    # Flatten the DataArray (values) and extract x and y coordinates for each time step
    flattened_values = data_array.values.reshape(-1)  # Flatten entire 2D array (y, x)

    # flattened_values = data_array.values.flatten()
    y_coords, x_coords = np.meshgrid(
        data_array.y.values, data_array.x.values, indexing="ij"
    )

    # Flatten the coordinate arrays
    flattened_x = x_coords.flatten()  # Repeat for each time step
    flattened_y = y_coords.flatten()  # Repeat for each time step

    # Create a DataFrame with columns for x, y, and value
    df = pd.DataFrame(
        {"x_pos": flattened_x, "y_pos": flattened_y, "value": flattened_values}
    )
    df["z_pos"] = 0

    # Convert to lat/lon
    # df = LV03toWGS84(df)
    df = LV95toWGS84(df)

    # Transform LV95 to WGS84 (lat, lon)
    lon, lat = df.lon.values, df.lat.values

    # Reshape the flattened WGS84 coordinates back to the original grid shape (time, y, x)
    lon = lon.reshape(x_coords.shape)  # Shape: (y, x)
    lat = lat.reshape(y_coords.shape)  # Shape: (y, x)

    # Assign the 1D WGS84 coordinates for swapping
    lon_1d = lon[0, :]  # take x (lon) values
    lat_1d = lat[:, 0]  # take y (lat) values

    # Assign the WGS84 coordinates back to the xarray
    data_array = data_array.assign_coords(lon=("x", lon_1d))  # Assign longitudes
    data_array = data_array.assign_coords(lat=("y", lat_1d))  # Assign latitudes

    # First, swap 'x' with 'lon' and 'y' with 'lat'
    data_array = data_array.swap_dims({"x": "lon", "y": "lat"})

    return data_array


def transform_xarray_coords_lv03_to_wgs84(data_array):
    # Flatten the DataArray (values) and extract x and y coordinates for each time step
    flattened_values = data_array.values.reshape(-1)  # Flatten entire 2D array (y, x)

    # flattened_values = data_array.values.flatten()
    y_coords, x_coords = np.meshgrid(
        data_array.y.values, data_array.x.values, indexing="ij"
    )

    # Flatten the coordinate arrays
    flattened_x = x_coords.flatten()  # Repeat for each time step
    flattened_y = y_coords.flatten()  # Repeat for each time step

    # Create a DataFrame with columns for x, y, and value
    df = pd.DataFrame(
        {"x_pos": flattened_x, "y_pos": flattened_y, "value": flattened_values}
    )
    df["z_pos"] = 0

    # Convert to lat/lon
    df = LV03toWGS84(df)
    # df = LV95toWGS84(df)

    # Transform LV95 to WGS84 (lat, lon)
    lon, lat = df.lon.values, df.lat.values

    # Reshape the flattened WGS84 coordinates back to the original grid shape (time, y, x)
    lon = lon.reshape(x_coords.shape)  # Shape: (y, x)
    lat = lat.reshape(y_coords.shape)  # Shape: (y, x)

    # Assign the 1D WGS84 coordinates for swapping
    lon_1d = lon[0, :]  # take x (lon) values
    lat_1d = lat[:, 0]  # take y (lat) values

    # Assign the WGS84 coordinates back to the xarray
    data_array = data_array.assign_coords(lon=("x", lon_1d))  # Assign longitudes
    data_array = data_array.assign_coords(lat=("y", lat_1d))  # Assign latitudes

    # First, swap 'x' with 'lon' and 'y' with 'lat'
    data_array = data_array.swap_dims({"x": "lon", "y": "lat"})

    return data_array


def LV03toWGS84(df):
    """Converts from swiss data coordinate system to lat/lon/height
    Args:
        df (pd.DataFrame): data in x/y swiss coordinates
    Returns:
        pd.DataFrame: data in lat/lon/coords
    """
    converter = GPSConverter()
    lat, lon, height = converter.LV03toWGS84(df["x_pos"], df["y_pos"], df["z_pos"])
    df["lat"] = lat
    df["lon"] = lon
    df["height"] = height
    df.drop(["x_pos", "y_pos", "z_pos"], axis=1, inplace=True)
    return df


def LV95toWGS84(df):
    """Converts from swiss data coordinate system to lat/lon/height
    Args:
        df (pd.DataFrame): data in x/y swiss coordinates
    Returns:
        pd.DataFrame: data in lat/lon/coords
    """
    transformer = pyproj.Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)

    # Sample CH1903+ / LV95 coordinates (Easting and Northing)

    # Transform to Latitude and Longitude (WGS84)
    lon, latitude = transformer.transform(df.x_pos, df.y_pos)

    df["lat"] = latitude
    df["lon"] = lon
    df.drop(["x_pos", "y_pos", "z_pos"], axis=1, inplace=True)
    return df


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


def xr_SGI_masked_topo(gdf_shapefiles, sgi_id, cfg):
    path_aspect = os.path.join(cfg.dataPath, path_SGI_topo, "aspect")
    path_slope = os.path.join(cfg.dataPath, path_SGI_topo, "slope")
    path_DEM = os.path.join(cfg.dataPath, path_SGI_topo, "dem_HR")

    # Get SGI topo files
    aspect_gl = [f for f in os.listdir(path_aspect) if sgi_id in f][0]
    slope_gl = [f for f in os.listdir(path_slope) if sgi_id in f][0]
    dem_gl = [f for f in os.listdir(path_DEM) if sgi_id in f][0]

    metadata_aspect, grid_data_aspect = load_grid_file(join(path_aspect, aspect_gl))
    metadata_slope, grid_data_slope = load_grid_file(join(path_slope, slope_gl))
    metadata_dem, grid_data_dem = load_grid_file(join(path_DEM, dem_gl))

    # Convert to xarray
    aspect = convert_to_xarray_geodata(grid_data_aspect, metadata_aspect)
    slope = convert_to_xarray_geodata(grid_data_slope, metadata_slope)
    dem = convert_to_xarray_geodata(grid_data_dem, metadata_dem)

    # Transform to WGS84
    aspect_wgs84 = transform_xarray_coords_lv95_to_wgs84(aspect)
    slope_wgs84 = transform_xarray_coords_lv95_to_wgs84(slope)
    dem_wgs84 = transform_xarray_coords_lv95_to_wgs84(dem)

    # 2016 shapefile of glacier
    gdf_mask_gl = gdf_shapefiles[gdf_shapefiles["sgi-id"] == sgi_id]

    # Mask over glacier outline
    mask, masked_aspect = extract_topo_over_outline(aspect_wgs84, gdf_mask_gl)
    mask, masked_slope = extract_topo_over_outline(slope_wgs84, gdf_mask_gl)
    mask, masked_dem = extract_topo_over_outline(dem_wgs84, gdf_mask_gl)

    # Create new dataset
    ds = xr.Dataset(
        {
            "masked_aspect": masked_aspect,
            "masked_slope": masked_slope,
            "masked_elev": masked_dem,
            "glacier_mask": mask,
        }
    )

    # Mask elevations below 0 (bug values)
    ds["masked_elev"] = ds.masked_elev.where(ds.masked_elev >= 0, np.nan)
    return ds


def extract_topo_over_outline(
    aspect_xarray: xr.DataArray,
    glacier_polygon_gdf,
    *,
    target_crs: int = 4326,
):
    """
    Mask aspect values by a glacier outline.

    Works with rasters in either EPSG:2056 (LV95) or EPSG:4326 (WGS84).
    By default, assumes EPSG:4326 if no CRS information is found.

    Parameters
    ----------
    aspect_xarray : xarray.DataArray
        2D array of aspect values. Coordinates can be (lon, lat) or (x, y).
        If `rioxarray` CRS is available, it will be used.
    glacier_polygon_gdf : geopandas.GeoDataFrame
        Glacier outline(s). Must have a defined CRS.
    target_crs : int, optional
        EPSG code of target coordinate system (default 4326).

    Returns
    -------
    mask_xarray : xarray.DataArray (bool/int)
        1 inside glacier, 0 outside, aligned to `aspect_xarray`.
    masked_aspect : xarray.DataArray
        aspect_xarray masked to the glacier outline (NaN outside).
    """
    # --- Validate glacier CRS ---
    if glacier_polygon_gdf.crs is None:
        raise ValueError("Glacier GeoDataFrame must have a defined CRS.")

    # --- Identify coordinate names ---
    if {"lon", "lat"}.issubset(aspect_xarray.coords):
        x_name, y_name = "lon", "lat"
    elif {"x", "y"}.issubset(aspect_xarray.coords):
        x_name, y_name = "x", "y"
    else:
        raise ValueError(
            "aspect_xarray must have coordinates named (lon, lat) or (x, y)."
        )

    x = aspect_xarray.coords[x_name].values
    y = aspect_xarray.coords[y_name].values
    nx, ny = len(x), len(y)

    # --- Determine the raster CRS ---
    try:
        raster_crs_obj = aspect_xarray.rio.crs  # type: ignore[attr-defined]
        if raster_crs_obj is not None and raster_crs_obj.to_epsg() is not None:
            raster_epsg = int(raster_crs_obj.to_epsg())
        else:
            raster_epsg = target_crs  # default to provided target CRS (usually 4326)
    except Exception:
        raster_epsg = target_crs  # fallback default

    # --- Reproject glacier polygons to raster CRS ---
    if glacier_polygon_gdf.crs.to_epsg() != raster_epsg:
        glacier_polygon_gdf = glacier_polygon_gdf.to_crs(f"EPSG:{raster_epsg}")

    # --- Compute transform ---
    left, right = float(np.min(x)), float(np.max(x))
    bottom, top = float(np.min(y)), float(np.max(y))
    transform = from_bounds(left, bottom, right, top, width=nx, height=ny)

    # --- Rasterize glacier polygon ---
    shapes = [(geom, 1) for geom in glacier_polygon_gdf.geometry if not geom.is_empty]
    mask = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=(ny, nx),
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    # --- Flip if y ascending (south to north) ---
    if y[0] < y[-1]:
        mask = np.flipud(mask)

    # --- Build outputs ---
    mask_xarray = xr.DataArray(
        mask,
        coords={
            y_name: aspect_xarray.coords[y_name],
            x_name: aspect_xarray.coords[x_name],
        },
        dims=(y_name, x_name),
        name="glacier_mask",
        attrs={"crs": f"EPSG:{raster_epsg}"},
    )

    masked_aspect = aspect_xarray.where(mask_xarray == 1)

    return mask_xarray, masked_aspect


def coarsenDS(
    ds, target_res_m=50, vars=["masked_slope", "masked_aspect", "masked_elev"]
):
    dx_m, dy_m = get_res_from_degrees(ds)  # Get resolution in meters

    # Compute resampling factor
    resampling_fac_lon = max(1, round(target_res_m / dx_m))
    resampling_fac_lat = max(1, round(target_res_m / dy_m))

    # print(f"Resampling factor: lon={resampling_fac_lon}, lat={resampling_fac_lat}")

    if dx_m < target_res_m or dy_m < target_res_m:
        # Coarsen non-binary variables with mean
        ds_non_binary = (
            ds[vars]
            .coarsen(lon=resampling_fac_lon, lat=resampling_fac_lat, boundary="trim")
            .mean()
        )

        # Coarsen glacier mask with max
        ds_glacier_mask = (
            ds[["glacier_mask"]]
            .coarsen(lon=resampling_fac_lon, lat=resampling_fac_lat, boundary="trim")
            .reduce(np.max)
        )

        # Merge back into a single dataset
        ds_res = xr.merge([ds_non_binary, ds_glacier_mask])
        return ds_res

    return ds


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


def get_rgi_sgi_ids(cfg, glacier_name):
    rgi_df = pd.read_csv(cfg.dataPath + path_glacier_ids, sep=",")
    rgi_df.rename(columns=lambda x: x.strip(), inplace=True)
    rgi_df.sort_values(by="short_name", inplace=True)
    rgi_df.set_index("short_name", inplace=True)

    # Handle 'clariden' separately due to its unique ID format
    if glacier_name == "clariden":
        sgi_id = (
            rgi_df.at["claridenU", "sgi-id"].strip()
            if "claridenU" in rgi_df.index
            else ""
        )
        rgi_id = (
            rgi_df.at["claridenU", "rgi_id.v6"] if "claridenU" in rgi_df.index else ""
        )
        rgi_shp = (
            rgi_df.at["claridenU", "rgi_id_v6_2016_shp"]
            if "claridenU" in rgi_df.index
            else ""
        )
    else:
        sgi_id = (
            rgi_df.at[glacier_name, "sgi-id"].strip()
            if glacier_name in rgi_df.index
            else ""
        )
        rgi_id = (
            rgi_df.at[glacier_name, "rgi_id.v6"] if glacier_name in rgi_df.index else ""
        )
        rgi_shp = (
            rgi_df.at[glacier_name, "rgi_id_v6_2016_shp"]
            if glacier_name in rgi_df.index
            else ""
        )

    return sgi_id, rgi_id, rgi_shp


def create_glacier_grid_SGI(glacierName, year, rgi_id, ds, start_month="10"):
    glacier_indices = np.where(ds["glacier_mask"].values == 1)

    # Glacier mask as boolean array:
    gl_mask_bool = ds["glacier_mask"].values.astype(bool)

    lon_coords = ds["lon"].values
    lat_coords = ds["lat"].values

    lon = lon_coords[glacier_indices[1]]
    lat = lat_coords[glacier_indices[0]]

    # Create a DataFrame
    data_grid = {
        "RGIId": [rgi_id] * len(ds.masked_elev.values[gl_mask_bool]),
        "POINT_LAT": lat,
        "POINT_LON": lon,
        "aspect": ds.masked_aspect.values[gl_mask_bool],
        "slope": ds.masked_slope.values[gl_mask_bool],
        "topo": ds.masked_elev.values[gl_mask_bool],
        "svf": ds.svf.values[gl_mask_bool],
    }
    df_grid = pd.DataFrame(data_grid)

    # Match to WGMS format:
    df_grid["POINT_ID"] = np.arange(1, len(df_grid) + 1)
    df_grid["N_MONTHS"] = 12
    df_grid["POINT_ELEVATION"] = df_grid["topo"]  # no other elevation available
    df_grid["POINT_BALANCE"] = 0  # fake PMB for simplicity (not used)

    # Add metadata that is not in WGMS dataset
    df_grid["PERIOD"] = "annual"
    df_grid["GLACIER"] = glacierName
    # Add the 'year' and date columns to the DataFrame
    df_grid["YEAR"] = np.tile(year, len(df_grid))
    df_grid["FROM_DATE"] = df_grid["YEAR"].apply(lambda x: str(x) + f"{start_month}01")
    df_grid["TO_DATE"] = df_grid["YEAR"].apply(lambda x: str(x + 1) + "0930")

    return df_grid


def add_OGGM_features(df_y_gl, voi, path_OGGM):
    df_pmb = df_y_gl.copy()

    # Initialize empty columns for the variables
    for var in voi:
        df_pmb[var] = np.nan

    # Path to OGGM datasets
    path_to_data = path_OGGM + "xr_grids/"

    # Group rows by RGIId
    grouped = df_pmb.groupby("RGIId")

    # Process each group
    for rgi_id, group in grouped:
        file_path = f"{path_to_data}{rgi_id}.zarr"

        try:
            # Load the xarray dataset for the current RGIId
            ds_oggm = xr.open_dataset(file_path)
        except FileNotFoundError:
            print(f"File not found for RGIId: {file_path}")
            continue

        # Define the coordinate transformation
        transf = pyproj.Transformer.from_proj(
            pyproj.CRS.from_user_input("EPSG:4326"),  # Input CRS (WGS84)
            pyproj.CRS.from_user_input(ds_oggm.pyproj_srs),  # Output CRS from dataset
            always_xy=True,
        )

        # Transform all coordinates in the group
        lon, lat = group["POINT_LON"].values, group["POINT_LAT"].values
        x_stake, y_stake = transf.transform(lon, lat)
        # Select nearest values for all points
        try:
            stake = ds_oggm.sel(
                x=xr.DataArray(x_stake, dims="points"),
                y=xr.DataArray(y_stake, dims="points"),
                method="nearest",
            )

            # Extract variables of interest
            stake_var = stake[voi]

            # Convert the extracted data to a DataFrame
            stake_var_df = stake_var.to_dataframe()

            # Update the DataFrame with the extracted values
            for var in voi:
                df_pmb.loc[group.index, var] = stake_var_df[var].values
        except KeyError as e:
            print(f"Variable missing in dataset {file_path}: {e}")
            continue
    return df_pmb


def xr_GLAMOS_masked_topo(cfg, sgi_id, ds_gl):
    path_aspect = os.path.join(cfg.dataPath, path_SGI_topo, "aspect")
    path_slope = os.path.join(cfg.dataPath, path_SGI_topo, "slope")

    # Load SGI topo files
    aspect_gl = [f for f in os.listdir(path_aspect) if sgi_id in f][0]
    slope_gl = [f for f in os.listdir(path_slope) if sgi_id in f][0]

    metadata_aspect, grid_data_aspect = load_grid_file(join(path_aspect, aspect_gl))
    metadata_slope, grid_data_slope = load_grid_file(join(path_slope, slope_gl))

    # Convert to xarray
    aspect = convert_to_xarray_geodata(grid_data_aspect, metadata_aspect)
    slope = convert_to_xarray_geodata(grid_data_slope, metadata_slope)

    # Transform to WGS84
    aspect_wgs84 = transform_xarray_coords_lv95_to_wgs84(aspect)
    slope_wgs84 = transform_xarray_coords_lv95_to_wgs84(slope)

    # Compute original resolution (for checking)
    dx_m, dy_m = get_res_from_degrees(aspect_wgs84)
    # print(f"aspect resolution: {dx_m} x {dy_m} meters")

    # Step 1: Downsample aspect & slope to glacier mask resolution
    aspect_resampled = aspect_wgs84.interp_like(ds_gl["glacier_mask"], method="nearest")
    slope_resampled = slope_wgs84.interp_like(ds_gl["glacier_mask"], method="nearest")

    # Compute new resolution (after downsampling)
    dx_m, dy_m = get_res_from_degrees(ds_gl["glacier_mask"])
    # print(f"New resolution (after downsampling): {dx_m} x {dy_m} meters")

    # Step 2: Apply the glacier mask
    masked_aspect = aspect_resampled.where(ds_gl["glacier_mask"] == 1, np.nan)
    masked_slope = slope_resampled.where(ds_gl["glacier_mask"] == 1, np.nan)

    # Resample DEM to the same resolution
    dem_resampled = ds_gl["dem"].interp_like(ds_gl["glacier_mask"], method="nearest")

    # Create a new dataset
    ds = xr.Dataset(
        {
            "masked_aspect": masked_aspect,
            "masked_elev": dem_resampled,
            "masked_slope": masked_slope,
            "glacier_mask": ds_gl["glacier_mask"],
        }
    )

    # Mask elevations below 0 (to remove erroneous values)
    ds["masked_elev"] = ds.masked_elev.where(ds.masked_elev >= 0, np.nan)

    return ds


def get_res_from_degrees(ds):
    # Get central latitude (mean of lat values)
    lat_center = ds.lat.values.mean()

    # Earth's approximate conversion factor (meters per degree)
    meters_per_degree_lat = 111320  # Roughly constant for latitude
    meters_per_degree_lon = 111320 * np.cos(
        np.radians(lat_center)
    )  # Adjust for longitude

    # Compute resolution
    dx_m = np.round(abs(ds.lon[1] - ds.lon[0]).values * meters_per_degree_lon, 3)
    dy_m = np.round(abs(ds.lat[1] - ds.lat[0]).values * meters_per_degree_lat, 3)

    return dx_m, dy_m


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


def get_gl_area(cfg):
    # Load glacier metadata
    rgi_df = pd.read_csv(cfg.dataPath + path_glacier_ids, sep=",")
    rgi_df.rename(columns=lambda x: x.strip(), inplace=True)
    rgi_df.sort_values(by="short_name", inplace=True)
    rgi_df.set_index("short_name", inplace=True)

    # Load the shapefile
    shapefile_path = os.path.join(
        cfg.dataPath,
        path_SGI_topo,
        "inventory_sgi2016_r2020",
        "SGI_2016_glaciers_copy.shp",
    )
    gdf_shapefiles = gpd.read_file(shapefile_path)

    gl_area = {}

    for glacierName in rgi_df.index:
        if glacierName == "clariden":
            rgi_shp = (
                rgi_df.loc["claridenL", "rgi_id_v6_2016_shp"]
                if "claridenL" in rgi_df.index
                else None
            )
        else:
            rgi_shp = rgi_df.loc[glacierName, "rgi_id_v6_2016_shp"]

        # Skip if rgi_shp is not found
        if pd.isna(rgi_shp) or rgi_shp is None:
            continue

        # Ensure matching data types
        rgi_shp = str(rgi_shp)
        gdf_mask_gl = gdf_shapefiles[gdf_shapefiles.RGIId.astype(str) == rgi_shp]

        # If a glacier is found, get its area
        if not gdf_mask_gl.empty:
            gl_area[glacierName] = gdf_mask_gl.Area.iloc[0]  # Use .iloc[0] safely

    return gl_area


def load_grid_file(filepath):
    with open(filepath, "r") as file:
        # Read metadata
        metadata = {}
        for _ in range(6):  # First 6 lines are metadata
            line = file.readline().strip().split()
            metadata[line[0].lower()] = float(line[1])

        # Get ncols from metadata to control the number of columns
        ncols = int(metadata["ncols"])
        nrows = int(metadata["nrows"])

        # Initialize an empty list to store rows of the grid
        data = []

        # Read the grid data line by line
        row_ = []
        for line in file:
            row = line.strip().split()
            if len(row_) < ncols:
                row_ += row
            if len(row_) == ncols:
                data.append(
                    [
                        np.nan if float(x) == metadata["nodata_value"] else float(x)
                        for x in row_
                    ]
                )
                # reset row_
                row_ = []

        # Convert list to numpy array
        grid_data = np.array(data)

        # Check that shape of grid data is correct
        assert grid_data.shape == (nrows, ncols)

    return metadata, grid_data


def datetime_obj(value):
    date = str(value)
    year = date[:4]
    month = date[4:6]
    day = date[6:8]
    return pd.to_datetime(month + "-" + day + "-" + year)


def transformDates(df_or):
    """Some dates are missing in the original GLAMOS data and need to be corrected.

    Args:
        df_or (pd.DataFrame): Raw GLAMOS DataFrame

    Returns:
        pd.DataFrame: DataFrame with corrected dates
    """
    df = df_or.copy()

    # Ensure 'date0' and 'date1' are datetime objects
    df["date0"] = df["date0"].apply(lambda x: datetime_obj(x))
    df["date1"] = df["date1"].apply(lambda x: datetime_obj(x))

    # Initialize new columns with NaT (not np.nan, since we'll use datetime later)
    df["date_fix0"] = pd.NaT
    df["date_fix1"] = pd.NaT

    # Assign fixed dates using .loc to avoid chained assignment warning
    for i in range(len(df)):
        year = df.loc[i, "date0"].year
        df.loc[i, "date_fix0"] = pd.Timestamp(f"{year}-10-01")
        df.loc[i, "date_fix1"] = pd.Timestamp(f"{year + 1}-09-30")

    # Format original dates for WGMS
    df["date0"] = df["date0"].apply(lambda x: x.strftime("%Y%m%d"))
    df["date1"] = df["date1"].apply(lambda x: x.strftime("%Y%m%d"))

    return df


def check_missing_years(folder_path, glacier_name, period):
    start_year, end_year = period
    expected_years = set(range(start_year, end_year + 1))

    # Extract years from filenames
    available_years = set()
    pattern = re.compile(rf"{glacier_name}_(\d{{4}})_annual\.zarr")

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            year = int(match.group(1))
            available_years.add(year)

    missing_years = list(expected_years - available_years)
    missing_years.sort()
    if missing_years:
        return True, missing_years
    else:
        return False, []


def process_geodetic_mass_balance_comparison(
    glacier_list,
    path_SMB_GLAMOS_csv,
    periods_per_glacier,
    geoMB_per_glacier,  # {'aletsch': [(mb, sigma), ...], ...}
    gl_area,
    test_glaciers,
    path_predictions,
    cfg,
):
    """
    For each glacier and period, compute mean MBM and GLAMOS MBs, attach the
    corresponding geodetic mass balance and its uncertainty (sigma), and return
    a DataFrame of results.
    """
    # Storage
    mbm_mb_mean, glamos_mb_mean = [], []
    mbm_mb_var, glamos_mb_var = [], []
    geodetic_mb, geodetic_sigma = [], []
    gl, gl_type, area = [], [], []
    period_len, start_year, end_year = [], [], []

    for glacier_name in tqdm(glacier_list, desc="Processing glaciers"):
        # Load GLAMOS annual balances
        glamos_file = os.path.join(
            path_SMB_GLAMOS_csv, "fix", f"{glacier_name}_fix.csv"
        )
        if os.path.exists(glamos_file):
            GLAMOS_glwmb = get_GLAMOS_glwmb(glacier_name, cfg)
            if GLAMOS_glwmb is None:
                GLAMOS_glwmb = pd.DataFrame()
        else:
            print(f"GLAMOS file not found for {glacier_name}. Using NaNs.")
            GLAMOS_glwmb = pd.DataFrame()

        periods = periods_per_glacier.get(glacier_name, [])
        geo_tuples = geoMB_per_glacier.get(glacier_name, [])

        if not periods or not geo_tuples:
            print(f"Skipping {glacier_name}: No geodetic mass balance data available.")
            continue

        # Path to model predictions
        folder_path = os.path.join(path_predictions, glacier_name)

        for i, period in enumerate(periods):
            # require matching geodetic tuple by index
            if (
                i >= len(geo_tuples)
                or not isinstance(geo_tuples[i], (tuple, list))
                or len(geo_tuples[i]) < 2
            ):
                print(
                    f"Skipping {glacier_name} {period}: missing geodetic (mb, sigma) tuple at index {i}."
                )
                continue

            # Special case skip
            if period[1] == 2021 and glacier_name == "silvretta":
                continue

            # Check input availability (your helper)
            is_missing, years_missing = check_missing_years(
                folder_path, glacier_name, period
            )
            if is_missing:
                print(
                    f"Skipping {glacier_name} {period}: Missing years: {years_missing}"
                )
                continue

            mbm_mb, glamos_mb = [], []
            for year in range(period[0], period[1] + 1):
                zarr_path = os.path.join(
                    folder_path, f"{glacier_name}_{year}_annual.zarr"
                )
                if not os.path.exists(zarr_path):
                    print(f"Warning: Missing MBM file for {glacier_name} ({year}).")
                    mbm_mb.append(np.nan)
                else:
                    # Zarr -> open_zarr (not open_dataset)
                    ds = xr.open_zarr(zarr_path)
                    mbm_mb.append(ds["pred_masked"].mean().values)
                glamos_mb.append(GLAMOS_glwmb["GLAMOS Balance"].get(year, np.nan))

            # Aggregate period stats
            mbm_mb_mean.append(np.nanmean(mbm_mb))
            mbm_mb_var.append(np.nanstd(mbm_mb))
            glamos_mb_mean.append(np.nanmean(glamos_mb))
            glamos_mb_var.append(np.nanstd(glamos_mb))

            # Geodetic (mb, sigma)
            g_mb, g_sig = geo_tuples[i][0], geo_tuples[i][1]
            geodetic_mb.append(g_mb)
            geodetic_sigma.append(g_sig)

            # Meta
            gl.append(glacier_name)
            gl_type.append(glacier_name in test_glaciers)
            period_len.append(period[1] - period[0])  # keep your original convention
            area.append(gl_area.get(glacier_name, np.nan))
            start_year.append(period[0])
            end_year.append(period[1])

    # Assemble DataFrame
    df_all = pd.DataFrame(
        {
            "MBM MB": mbm_mb_mean,
            "GLAMOS MB": glamos_mb_mean,
            "MBM MB std": mbm_mb_var,
            "GLAMOS MB std": glamos_mb_var,
            "Geodetic MB": geodetic_mb,
            "Geodetic MB sigma": geodetic_sigma,
            "GLACIER": gl,
            "Period Length": period_len,
            "Test Glacier": gl_type,
            "Area": area,
            "start_year": start_year,
            "end_year": end_year,
        }
    )

    df_all.sort_values(by="Area", inplace=True, ascending=True)
    return df_all


def prepareGeoTargets(geodetic_mb, periods_per_glacier, glacier_name=None):
    """
    Prepare the vector of geodetic targets for a given glacier by looping over the
    periods defined for this glacier.

    Parameters:
    -----------
    geodetic_mb: pd.Dataframe
        Dataframe that contains the geodetic mass balance.
    periods_per_glacier: Dictionary of list of tuples.
        Each key is the name of a glacier and the list associated to each entry
        contains tuples of integers that define the time window over which data
        are defined.
    glacier_name: str or None
        The name of the glacier to process. If not specified, the geodetic targets
        are generated for all the keys of `periods_per_glacier`.

    Returns:
    --------
        Dictionary of numpy arrays or numpy array depending if `glacier_name` is
        specified or not.
    """
    if glacier_name is not None:
        geodetic_MB_target = []
        Bgeod_key = (
            "Bgeod" if "Bgeod" in geodetic_mb.keys() else "Bgeod_mwe_a"
        )  # Handle the dV_DOI2025_allcomb_prelim.csv file
        for geodetic_period in periods_per_glacier[glacier_name]:
            mask = (
                (geodetic_mb.glacier_name == glacier_name)
                & (geodetic_mb.Astart == geodetic_period[0])
                & (geodetic_mb.Aend == geodetic_period[1])
            )
            geodetic_MB_target.append(geodetic_mb[mask][Bgeod_key].values[0])

        return np.array(geodetic_MB_target)
    else:
        return {
            glacier_name: prepareGeoTargets(
                geodetic_mb, periods_per_glacier, glacier_name=glacier_name
            )
            for glacier_name in periods_per_glacier
        }


def build_periods_per_glacier(geodetic_mb):
    """
    Builds the dictionary that contains the geodetic periods for each glacier.

    Parameters:
    -----------
    geodetic_mb: pd.Dataframe
        Dataframe that contains the geodetic mass balance.

    Returns:
    --------
    periods_per_glacier: Dictionary of list of tuples.
        Each key is the name of a glacier and the list associated to each entry
        contains tuples of integers that define the time window over which data
        are defined.
    geoMB_per_glacier: Dictionary of list of floats.
        Each key is the name of a glacier and the list associated to each entry
        contains the geodetic mass balance.
    """

    periods_per_glacier = defaultdict(list)
    geoMB_per_glacier = defaultdict(list)
    geoMB_sigma_per_glacier = defaultdict(list)

    # Iterate through the DataFrame rows
    for _, row in geodetic_mb.iterrows():
        glacier_name = row["glacier_name"]
        start_year = row["Astart"]
        end_year = row["Aend"]
        geoMB = row["Bgeod_mwe_a"]
        sigma = row["sigma_mwe_a"]

        # Append the (start, end) tuple to the glacier's list
        # Only if period is longer than 5 years
        if end_year - start_year >= 5:
            periods_per_glacier[glacier_name].append((start_year, end_year))
            # append geodetic MB and its uncertainty
            geoMB_per_glacier[glacier_name].append((geoMB, sigma))

    # sort by glacier_list
    periods_per_glacier = dict(sorted(periods_per_glacier.items()))
    geoMB_per_glacier = dict(sorted(geoMB_per_glacier.items()))

    return periods_per_glacier, geoMB_per_glacier


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


# ----------------- LSTM specific helpers -----------------

import os
import re
import glob
import numpy as np
import pandas as pd
import xarray as xr


# ---------- utils ----------
def _period_tags(period: str):
    """Map period -> (lstm_tag, glamos_tag)."""
    p = period.lower()
    if p in ("annual", "ann", "yearly"):
        return "annual", "ann"
    if p in ("winter", "win"):
        return "winter", "win"
    raise ValueError(f"Unknown period: {period}")


# ---------- list years ----------
def list_years_from_lstm(glacier_name, path_pred_lstm, period="annual"):
    lstm_tag, _ = _period_tags(period)
    base_lstm = os.path.join(path_pred_lstm, glacier_name)
    pattern = os.path.join(base_lstm, f"*_{lstm_tag}.zarr")
    years = []
    for f in glob.glob(pattern):
        # e.g. aletsch_2010_annual.zarr OR aletsch_2010_winter.zarr
        m = re.match(r".*[\\/](\D+)?(\d{4})_" + re.escape(lstm_tag) + r"\.zarr$", f)
        if m:
            years.append(int(m.group(2)))
    return sorted(set(years))


# ---------- paths for a given year ----------
def paths_for_year(path_pred_lstm, glacier_name, year, cfg, period="annual"):
    lstm_tag, glamos_tag = _period_tags(period)

    # look for GLAMOS in lv95 then lv03
    base_gl = os.path.join(
        cfg.dataPath, path_distributed_MB_glamos, "GLAMOS", glacier_name
    )
    cand_lv95 = os.path.join(base_gl, f"{year}_{glamos_tag}_fix_lv95.grid")
    cand_lv03 = os.path.join(base_gl, f"{year}_{glamos_tag}_fix_lv03.grid")
    grid_path = (
        cand_lv95
        if os.path.exists(cand_lv95)
        else (cand_lv03 if os.path.exists(cand_lv03) else None)
    )

    # LSTM file
    mbm_file_lstm = os.path.join(
        path_pred_lstm, glacier_name, f"{glacier_name}_{year}_{lstm_tag}.zarr"
    )

    return grid_path, mbm_file_lstm  # grid_path may be None


# ---------- per-year processing ----------
def process_year(glacier_name, path_pred_lstm, year, cfg, period="annual"):
    grid_path, mbm_file_lstm = paths_for_year(
        path_pred_lstm, glacier_name, year, cfg, period=period
    )

    # ---- GLAMOS (load + WGS84) ----
    metadata, grid_data = load_grid_file(grid_path)
    ds_glamos = convert_to_xarray_geodata(grid_data, metadata)

    # decide transform from filename suffix
    if grid_path.endswith("_lv03.grid"):
        ds_glamos_wgs84 = transform_xarray_coords_lv03_to_wgs84(ds_glamos)
    else:
        ds_glamos_wgs84 = transform_xarray_coords_lv95_to_wgs84(ds_glamos)

    # ---- LSTM (load + smooth) ----
    ds_mbm_lstm = apply_gaussian_filter(xr.open_dataset(mbm_file_lstm, engine="zarr"))

    # ---- coord name resolution ----
    lon_lstm = "lon" if "lon" in ds_mbm_lstm.coords else "longitude"
    lat_lstm = "lat" if "lat" in ds_mbm_lstm.coords else "latitude"

    lon_gl = "lon" if "lon" in ds_glamos_wgs84.coords else "longitude"
    lat_gl = "lat" if "lat" in ds_glamos_wgs84.coords else "latitude"

    # ---- LSTM: raster -> dataframe + elevation merge ----
    df_pred_lstm = (
        ds_mbm_lstm["pred_masked"]
        .to_dataframe()
        .reset_index()
        .drop(["x", "y"], axis=1, errors="ignore")
        .merge(
            ds_mbm_lstm["masked_elev"]
            .to_dataframe()
            .reset_index()
            .drop(["x", "y"], axis=1, errors="ignore"),
            on=[lat_lstm, lon_lstm],
            how="left",
        )
        .dropna()
        .rename(
            columns={
                "pred_masked": "pred",
                "masked_elev": "POINT_ELEVATION",
                lat_lstm: "lat",
                lon_lstm: "lon",
            }
        )
    )
    df_pred_lstm["YEAR"] = year
    df_pred_lstm["PERIOD"] = _period_tags(period)[0]  # 'annual' or 'winter'

    # ---- 100 m binning (LSTM) ----
    min_alt = np.floor(df_pred_lstm["POINT_ELEVATION"].min() / 100) * 100
    max_alt = np.ceil(df_pred_lstm["POINT_ELEVATION"].max() / 100) * 100
    bins = np.arange(min_alt, max_alt + 100, 100)
    df_pred_lstm["altitude_interval"] = pd.cut(
        df_pred_lstm["POINT_ELEVATION"], bins=bins, right=False
    )
    centers = {
        iv: round((iv.left + iv.right) / 2)
        for iv in df_pred_lstm["altitude_interval"].cat.categories
    }
    df_pred_lstm["altitude_interval"] = df_pred_lstm["altitude_interval"].map(centers)

    # ---- GLAMOS: sample elevation from LSTM masked_elev (nearest) ----
    elev_da = ds_mbm_lstm["masked_elev"].rename({lat_lstm: "lat", lon_lstm: "lon"})
    glamos_da = ds_glamos_wgs84.rename({lat_gl: "lat", lon_gl: "lon"})

    elev_on_glamos = elev_da.interp(
        lat=glamos_da["lat"], lon=glamos_da["lon"], method="nearest"
    )

    df_pred_glamos = (
        glamos_da.to_dataframe(name="pred")
        .reset_index()
        .drop(["x", "y"], axis=1, errors="ignore")
        .merge(
            elev_on_glamos.to_dataframe(name="POINT_ELEVATION").reset_index(),
            on=["lat", "lon"],
            how="left",
        )
        .dropna(subset=["POINT_ELEVATION"])
    )
    df_pred_glamos["YEAR"] = year
    df_pred_glamos["PERIOD"] = _period_tags(period)[0]  # 'annual' or 'winter'
    df_pred_glamos["SOURCE"] = "GLAMOS"

    # same bins as LSTM for comparability
    df_pred_glamos["altitude_interval"] = pd.cut(
        df_pred_glamos["POINT_ELEVATION"], bins=bins, right=False
    )
    df_pred_glamos = df_pred_glamos.dropna(subset=["altitude_interval"]).copy()
    df_pred_glamos["altitude_interval"] = df_pred_glamos["altitude_interval"].map(
        centers
    )

    return df_pred_lstm, df_pred_glamos


# ---------- driver over all years ----------
def build_all_years_df(glacier_name, path_pred_lstm, cfg, period="annual"):
    """
    Process all available years for a glacier and period ('annual' or 'winter').
    Returns concatenated DataFrames for LSTM, GLAMOS, and both combined.
    """
    years = list_years_from_lstm(glacier_name, path_pred_lstm, period=period)

    dfs_lstm, dfs_glamos = [], []

    def validate_paths(year, grid_path, mbm_file):
        if grid_path is None:
            print(
                f"[skip] {glacier_name} {year}: GLAMOS grid file not found at: {grid_path}"
            )
            return False
        if not os.path.exists(grid_path):
            print(f"[skip] {glacier_name} {year}: GLAMOS grid missing at: {grid_path}")
            return False
        if mbm_file is None:
            print(f"[skip] {glacier_name} {year}: LSTM zarr path not provided.")
            return False
        if not os.path.exists(mbm_file):
            print(f"[skip] {glacier_name} {year}: LSTM zarr missing: {mbm_file}")
            return False
        return True

    for y in years:
        grid_path, mbm_file = paths_for_year(
            path_pred_lstm, glacier_name, y, cfg, period=period
        )
        if not validate_paths(y, grid_path, mbm_file):
            continue

        try:
            df_pred_lstm, df_pred_glamos = process_year(
                glacier_name, path_pred_lstm, y, cfg, period=period
            )
            if df_pred_lstm is not None and len(df_pred_lstm):
                dfs_lstm.append(df_pred_lstm)
            else:
                print(f"[warn] {y}: empty LSTM dataframe.")

            if df_pred_glamos is not None and len(df_pred_glamos):
                dfs_glamos.append(df_pred_glamos)
            else:
                print(f"[warn] {y}: empty GLAMOS dataframe.")
        except Exception as e:
            print(f"[error] {y}: {e}")

    df_all_years_lstm = (
        pd.concat(dfs_lstm, ignore_index=True) if dfs_lstm else pd.DataFrame()
    )
    df_all_years_glamos = (
        pd.concat(dfs_glamos, ignore_index=True) if dfs_glamos else pd.DataFrame()
    )

    if not df_all_years_lstm.empty:
        df_all_years_lstm = df_all_years_lstm.assign(SOURCE="LSTM")
    if not df_all_years_glamos.empty and "SOURCE" not in df_all_years_glamos.columns:
        df_all_years_glamos = df_all_years_glamos.assign(SOURCE="GLAMOS")

    if not df_all_years_lstm.empty or not df_all_years_glamos.empty:
        df_all_years = pd.concat(
            [df_all_years_lstm, df_all_years_glamos], ignore_index=True
        ).drop(columns=["x", "y"], errors="ignore")
    else:
        df_all_years = pd.DataFrame()

    return df_all_years_lstm, df_all_years_glamos, df_all_years


def load_glwd_lstm_predictions(PATH_PREDICTIONS_LSTM, hydro_months):
    """
    Loads LSTM glacier predictions from Zarr files into a single pandas DataFrame.

    Parameters
    ----------
    PATH_PREDICTIONS_LSTM : str
        Path to the directory containing glacier subdirectories.
    hydro_months : list of str
        List of hydrological months (e.g. ['oct', 'nov', 'dec', ...]).

    Returns
    -------
    df_months_LSTM : pandas.DataFrame
        Combined DataFrame containing predictions, elevation, year, and glacier name.
    """
    glaciers = os.listdir(PATH_PREDICTIONS_LSTM)

    # Initialize final storage for all glacier data
    all_glacier_data = []

    # Loop over glaciers
    for glacier_name in tqdm(glaciers):
        glacier_path = os.path.join(PATH_PREDICTIONS_LSTM, glacier_name)
        if not os.path.isdir(glacier_path):
            continue  # skip non-directories

        # Regex pattern adapted for current glacier name
        pattern = re.compile(rf"{glacier_name}_(\d{{4}})_[a-z]{{3}}\.zarr")

        # Extract available years
        years = set()
        for fname in os.listdir(glacier_path):
            match = pattern.match(fname)
            if match:
                years.add(int(match.group(1)))
        years = sorted(years)

        # Collect all year-month data
        all_years_data = []
        for year in years:
            monthly_data = {}
            for month in hydro_months:
                zarr_path = os.path.join(
                    glacier_path, f"{glacier_name}_{year}_{month}.zarr"
                )
                if not os.path.exists(zarr_path):
                    continue

                ds = xr.open_dataset(zarr_path)
                df = (
                    ds.pred_masked.to_dataframe().drop(["x", "y"], axis=1).reset_index()
                )
                df_pred_months = df[df.pred_masked.notna()]

                df_el = (
                    ds.masked_elev.to_dataframe().drop(["x", "y"], axis=1).reset_index()
                )
                df_elv_months = df_el[df.pred_masked.notna()]

                df_pred_months["elevation"] = df_elv_months.masked_elev.values

                monthly_data[month] = df_pred_months.pred_masked.values

            if monthly_data:
                df_months = pd.DataFrame(monthly_data)
                df_months["year"] = year
                df_months["glacier"] = glacier_name  # add glacier name
                df_months["elevation"] = df_pred_months.elevation.values
                all_years_data.append(df_months)

        # Concatenate this glacier's data
        if all_years_data:
            df_glacier = pd.concat(all_years_data, axis=0, ignore_index=True)
            all_glacier_data.append(df_glacier)

    # Final full DataFrame for all glaciers
    df_months_LSTM = pd.concat(all_glacier_data, axis=0, ignore_index=True)
    return df_months_LSTM


def load_glwd_nn_predictions(PATH_PREDICTIONS_NN, hydro_months):
    """
    Loads neural network glacier predictions from Zarr files into a single pandas DataFrame.

    Parameters
    ----------
    PATH_PREDICTIONS_NN : str
        Path to the directory containing glacier subdirectories.
    hydro_months : list of str
        List of hydrological months (e.g. ['oct', 'nov', 'dec', ...]).

    Returns
    -------
    df_months_NN : pandas.DataFrame
        Combined DataFrame containing predictions, elevation, year, and glacier name.
    """
    glaciers = os.listdir(PATH_PREDICTIONS_NN)

    # Initialize final storage for all glacier data
    all_glacier_data = []

    # Loop over glaciers
    for glacier_name in tqdm(glaciers):
        glacier_path = os.path.join(PATH_PREDICTIONS_NN, glacier_name)
        if not os.path.isdir(glacier_path):
            continue  # skip non-directories

        # Regex pattern adapted for current glacier name
        pattern = re.compile(rf"{glacier_name}_(\d{{4}})_[a-z]{{3}}\.zarr")

        # Extract available years
        years = set()
        for fname in os.listdir(glacier_path):
            match = pattern.match(fname)
            if match:
                years.add(int(match.group(1)))
        years = sorted(years)

        # Collect all year-month data
        all_years_data = []
        for year in years:
            monthly_data = {}
            for month in hydro_months:
                zarr_path = os.path.join(
                    glacier_path, f"{glacier_name}_{year}_{month}.zarr"
                )
                if not os.path.exists(zarr_path):
                    continue

                ds = xr.open_dataset(zarr_path)
                df = (
                    ds.pred_masked.to_dataframe().drop(["x", "y"], axis=1).reset_index()
                )
                df_pred_months = df[df.pred_masked.notna()]

                df_el = (
                    ds.masked_elev.to_dataframe().drop(["x", "y"], axis=1).reset_index()
                )
                df_elv_months = df_el[df.pred_masked.notna()]

                df_pred_months["elevation"] = df_elv_months.masked_elev.values

                monthly_data[month] = df_pred_months.pred_masked.values

            if monthly_data:
                df_months = pd.DataFrame(monthly_data)
                df_months["year"] = year
                df_months["glacier"] = glacier_name  # add glacier name
                df_months["elevation"] = df_pred_months.elevation.values
                all_years_data.append(df_months)

        # Concatenate this glacier's data
        if all_years_data:
            df_glacier = pd.concat(all_years_data, axis=0, ignore_index=True)
            all_glacier_data.append(df_glacier)

    # Final full DataFrame for all glaciers
    df_months_NN = pd.concat(all_glacier_data, axis=0, ignore_index=True)
    return df_months_NN


def pick_file_glamos(cfg, glacier, year, period="winter"):
    suffix = "ann" if period == "annual" else "win"
    base = os.path.join(cfg.dataPath, path_distributed_MB_glamos, "GLAMOS", glacier)
    cand_lv95 = os.path.join(base, f"{year}_{suffix}_fix_lv95.grid")
    cand_lv03 = os.path.join(base, f"{year}_{suffix}_fix_lv03.grid")
    if os.path.exists(cand_lv95):
        return cand_lv95, "lv95"
    if os.path.exists(cand_lv03):
        return cand_lv03, "lv03"
    return None, None


def load_glamos_wgs84(cfg, glacier, year, period):
    """Load one GLAMOS .grid file and return it as an xarray in WGS84."""
    path, cs = pick_file_glamos(cfg, glacier, year, period)
    if path is None:
        return None
    meta, arr = load_grid_file(path)
    da = convert_to_xarray_geodata(arr, meta)
    if cs == "lv03":
        return transform_xarray_coords_lv03_to_wgs84(da)
    elif cs == "lv95":
        return transform_xarray_coords_lv95_to_wgs84(da)
    else:
        return None


def load_all_glamos(cfg, glacier_years, path_glamos):
    """
    Loads both annual and winter GLAMOS grids for all glaciers and years,
    interpolates DEM elevation onto MB grids, and returns two DataFrames:
    df_GLAMOS_w, df_GLAMOS_a.
    """
    all_glacier_data_w, all_glacier_data_a = [], []

    for glacier_name in tqdm(glacier_years.keys(), desc="Loading GLAMOS data"):
        glacier_path = os.path.join(path_glamos, glacier_name)
        if not os.path.isdir(glacier_path):
            continue

        years = glacier_years[glacier_name]
        all_years_w, all_years_a = [], []

        for year in years:
            # --- Load DEM ---
            dem_path = (
                cfg.dataPath
                + path_GLAMOS_topo
                + f"xr_masked_grids/{glacier_name}_{year}.zarr"
            )
            if not os.path.exists(dem_path):
                continue
            ds_dem = xr.open_zarr(dem_path)

            # --- Load Winter MB ---
            ds_w = load_glamos_wgs84(cfg, glacier_name, year, period="winter")
            if ds_w is not None:
                masked_elev_interp = ds_dem["masked_elev"].interp_like(
                    ds_w, method="nearest"
                )
                masked_elev_interp = masked_elev_interp.assign_coords(
                    x=ds_w.x, y=ds_w.y
                )
                ds_merged_w = xr.merge(
                    [
                        ds_w.to_dataset(name="mb"),
                        masked_elev_interp.to_dataset(name="masked_elev"),
                    ],
                    compat="override",
                )

                df_w = ds_merged_w.to_dataframe().reset_index()
                df_w = df_w[df_w["mb"].notna() & df_w["masked_elev"].notna()]
                df_w = df_w[["x", "y", "mb", "masked_elev"]]
                df_w["year"] = year
                df_w["glacier"] = glacier_name
                df_w["period"] = "winter"
                all_years_w.append(df_w)

            # --- Load Annual MB ---
            ds_a = load_glamos_wgs84(cfg, glacier_name, year, period="annual")
            if ds_a is not None:
                masked_elev_interp = ds_dem["masked_elev"].interp_like(
                    ds_a, method="nearest"
                )
                masked_elev_interp = masked_elev_interp.assign_coords(
                    x=ds_a.x, y=ds_a.y
                )
                ds_merged_a = xr.merge(
                    [
                        ds_a.to_dataset(name="mb"),
                        masked_elev_interp.to_dataset(name="masked_elev"),
                    ],
                    compat="override",
                )

                df_a = ds_merged_a.to_dataframe().reset_index()
                df_a = df_a[df_a["mb"].notna() & df_a["masked_elev"].notna()]
                df_a = df_a[["x", "y", "mb", "masked_elev"]]
                df_a["year"] = year
                df_a["glacier"] = glacier_name
                df_a["period"] = "annual"
                all_years_a.append(df_a)

        # --- Concatenate per glacier ---
        if all_years_w:
            df_glacier_w = pd.concat(all_years_w, ignore_index=True)
            all_glacier_data_w.append(df_glacier_w)
        if all_years_a:
            df_glacier_a = pd.concat(all_years_a, ignore_index=True)
            all_glacier_data_a.append(df_glacier_a)

    # --- Final combined DataFrames ---
    df_GLAMOS_w = (
        pd.concat(all_glacier_data_w, ignore_index=True)
        if all_glacier_data_w
        else pd.DataFrame()
    )
    df_GLAMOS_a = (
        pd.concat(all_glacier_data_a, ignore_index=True)
        if all_glacier_data_a
        else pd.DataFrame()
    )

    # --- Drop x/y and rename elevation column ---
    if not df_GLAMOS_w.empty:
        df_GLAMOS_w = df_GLAMOS_w.drop(["x", "y", "period"], axis=1).rename(
            columns={"masked_elev": "elevation", "mb": "apr"}
        )
    if not df_GLAMOS_a.empty:
        df_GLAMOS_a = df_GLAMOS_a.drop(["x", "y", "period"], axis=1).rename(
            columns={"masked_elev": "elevation", "mb": "sep"}
        )

    return df_GLAMOS_w, df_GLAMOS_a
