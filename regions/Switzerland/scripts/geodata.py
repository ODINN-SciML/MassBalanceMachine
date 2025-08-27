import os
import pandas as pd
from scipy.ndimage import gaussian_filter
import pyproj
import numpy as np
from collections import defaultdict
from dateutil.relativedelta import relativedelta
import rasterio
from os.path import isfile, join
import geopandas as gpd
import xarray as xr
from datetime import datetime
import re
from tqdm.notebook import tqdm


from regions.Switzerland.scripts.config_CH import *
from regions.Switzerland.scripts.wgs84_ch1903 import *


def get_glwd_glamos_years(cfg, glacier_name):
    folder = os.path.join(cfg.dataPath, path_distributed_MB_glamos, 'GLAMOS', glacier_name)

    if not os.path.exists(folder):
        print(f"Warning: Folder {folder} does not exist.")
        return []
    pattern = re.compile(r'^(\d{4})_ann_fix\.grid$')
    years = []
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            years.append(int(match.group(1)))  # Extract the year as an integer
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

    # Construct file path safely
    file_path = os.path.join(cfg.dataPath, path_SMB_GLAMOS_csv, "fix",
                             f"{glacier_name}_fix.csv")

    # Check if file exists
    if not os.path.exists(file_path):
        print(
            f"Warning: GLAMOS data file not found for {glacier_name}. Skipping..."
        )
        return None
    years = get_glwd_glamos_years(cfg, glacier_name)
    if years == []:
        print(f"Warning: No GLAMOS data found for {glacier_name}.")
    glamos_glwd_mb = []
    for year in years:
        file_ann = f"{year}_ann_fix_lv95.grid"
        grid_path_ann = os.path.join(cfg.dataPath, path_distributed_MB_glamos, 'GLAMOS',
                                    glacier_name, file_ann)

        metadata_ann, grid_data_ann = load_grid_file(grid_path_ann)
        ds_glamos_ann = convert_to_xarray_geodata(grid_data_ann, metadata_ann)
        ds_glamos_wgs84_ann = transform_xarray_coords_lv95_to_wgs84(ds_glamos_ann)
        glamos_glwd_mb.append(float(ds_glamos_wgs84_ann.mean().values))

    df = pd.DataFrame({
        'GLAMOS Balance': glamos_glwd_mb,
        'YEAR': years
    })
    
    # set index years
    df.set_index('YEAR', inplace = True)
    return df
    

def apply_gaussian_filter(ds, variable_name='pred_masked', sigma: float = 1):
    # Get the DataArray for the specified variable
    data_array = ds[variable_name]

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
    ds[variable_name] = smoothed_data
    return ds


def convert_to_xarray_geodata(grid_data, metadata):
    """Converts .grid file data to an xarray DataArray.

    Args:
        grid_data (.grid): grid file of glacier DEM
        metadata (dic): metadata of the grid file

    Returns:
        xr.DataSet: xarray DataArray of the grid data in LV95 coordinates
    """

    # Extract metadata values
    ncols = int(metadata['ncols'])
    nrows = int(metadata['nrows'])
    xllcorner = metadata['xllcorner']
    yllcorner = metadata['yllcorner']
    cellsize = metadata['cellsize']

    # Create x and y coordinates based on the metadata
    x_coords = xllcorner + np.arange(ncols) * cellsize
    y_coords = yllcorner + np.arange(nrows) * cellsize

    # Create the xarray DataArray
    data_array = xr.DataArray(np.flip(grid_data, axis=0),
                              dims=("y", "x"),
                              coords={
                                  "y": y_coords,
                                  "x": x_coords
                              },
                              name="grid_data")
    return data_array


def transform_xarray_coords_lv95_to_wgs84(data_array):
    # Flatten the DataArray (values) and extract x and y coordinates for each time step
    flattened_values = data_array.values.reshape(
        -1)  # Flatten entire 2D array (y, x)

    # flattened_values = data_array.values.flatten()
    y_coords, x_coords = np.meshgrid(data_array.y.values,
                                     data_array.x.values,
                                     indexing='ij')

    # Flatten the coordinate arrays
    flattened_x = x_coords.flatten()  # Repeat for each time step
    flattened_y = y_coords.flatten()  # Repeat for each time step

    # Create a DataFrame with columns for x, y, and value
    df = pd.DataFrame({
        'x_pos': flattened_x,
        'y_pos': flattened_y,
        'value': flattened_values
    })
    df['z_pos'] = 0

    # Convert to lat/lon
    #df = LV03toWGS84(df)
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
    data_array = data_array.assign_coords(lon=("x",
                                               lon_1d))  # Assign longitudes
    data_array = data_array.assign_coords(lat=("y",
                                               lat_1d))  # Assign latitudes

    # First, swap 'x' with 'lon' and 'y' with 'lat'
    data_array = data_array.swap_dims({'x': 'lon', 'y': 'lat'})

    return data_array


def LV03toWGS84(df):
    """Converts from swiss data coordinate system to lat/lon/height
    Args:
        df (pd.DataFrame): data in x/y swiss coordinates
    Returns:
        pd.DataFrame: data in lat/lon/coords
    """
    converter = GPSConverter()
    lat, lon, height = converter.LV03toWGS84(df['x_pos'], df['y_pos'],
                                             df['z_pos'])
    df['lat'] = lat
    df['lon'] = lon
    df['height'] = height
    df.drop(['x_pos', 'y_pos', 'z_pos'], axis=1, inplace=True)
    return df


def LV95toWGS84(df):
    """Converts from swiss data coordinate system to lat/lon/height
    Args:
        df (pd.DataFrame): data in x/y swiss coordinates
    Returns:
        pd.DataFrame: data in lat/lon/coords
    """
    transformer = pyproj.Transformer.from_crs("EPSG:2056",
                                              "EPSG:4326",
                                              always_xy=True)

    # Sample CH1903+ / LV95 coordinates (Easting and Northing)

    # Transform to Latitude and Longitude (WGS84)
    lon, latitude = transformer.transform(df.x_pos, df.y_pos)

    df['lat'] = latitude
    df['lon'] = lon
    df.drop(['x_pos', 'y_pos', 'z_pos'], axis=1, inplace=True)
    return df

def LV03toLV95(df):
    """Convert Swiss LV03 (EPSG:21781) coordinates to LV95 (EPSG:2056)."""
    transformer = pyproj.Transformer.from_crs("EPSG:21781", "EPSG:2056", always_xy=True)
    x_lv95, y_lv95 = transformer.transform(df.x_pos, df.y_pos)
    df['x_lv95'] = x_lv95
    df['y_lv95'] = y_lv95
    df.drop(['x_pos', 'y_pos', 'z_pos'], axis=1, inplace=True)
    return df

def transform_xarray_coords_lv03_to_lv95(data_array):
    # Extract and flatten x and y coordinates
    y_coords, x_coords = np.meshgrid(data_array.y.values, data_array.x.values, indexing='ij')
    flattened_x = x_coords.flatten()
    flattened_y = y_coords.flatten()
    flattened_values = data_array.values.flatten()

    # Create DataFrame
    df = pd.DataFrame({
        'x_pos': flattened_x,
        'y_pos': flattened_y,
        'value': flattened_values
    })
    df['z_pos'] = 0  # dummy height

    # Convert from LV03 to LV95
    df = LV03toLV95(df)

    # Reshape coordinates back to 2D
    x_lv95 = df.x_lv95.values.reshape(x_coords.shape)
    y_lv95 = df.y_lv95.values.reshape(y_coords.shape)

    # 1D coordinates to assign back to xarray
    x_lv95_1d = x_lv95[0, :]  # Eastings
    y_lv95_1d = y_lv95[:, 0]  # Northings

    # Assign new LV95 coordinates and swap dims
    data_array = data_array.assign_coords(x_lv95=("x", x_lv95_1d),
                                          y_lv95=("y", y_lv95_1d))
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
    with open(filepath, 'w') as f:
        f.write(f"ncols         {ncols}\n")
        f.write(f"nrows         {nrows}\n")
        f.write(f"xllcorner     {xllcorner:.6f}\n")
        f.write(f"yllcorner     {yllcorner:.6f}\n")
        f.write(f"cellsize      {cellsize:.6f}\n")
        f.write(f"NODATA_value  {nodata_value}\n")
        
        for row in values[::-1]:  # Flip vertically
            f.write(' '.join(f"{val:.6f}" for val in row) + "\n")

def organize_rasters_by_hydro_year(path_S2, satellite_years):
    rasters = defaultdict(
        lambda: defaultdict(list))  # Nested dictionary for years and months

    for year in satellite_years:
        folder_path = os.path.join(path_S2, str(year))
        for f in os.listdir(folder_path):
            if f.endswith(".tif"):  # Only process raster files
                # Step 1: Extract the date from the string
                date_str = f.split(
                    '_')[3][:8]  # Extract the 8-digit date (YYYYMMDD)
                file_date = datetime.strptime(
                    date_str, "%Y%m%d")  # Convert to datetime object

                closest_month, hydro_year = get_hydro_year_and_month(file_date)
                if hydro_year < 2022:
                    rasters[hydro_year][closest_month].append(f)

    return rasters


def get_hydro_year_and_month(file_date):
    if file_date.day < 15:
        # Move to the first day of the previous month
        file_date -= relativedelta(months=1)  # Move to the previous month
        file_date = file_date.replace(
            day=1)  # Set the day to the 1st of the previous month
    else:
        # Move to the first day of the current month
        file_date = file_date.replace(
            day=1)  # Set the day to the 1st of the current month

    # Step 2: Determine the closest month
    closest_month = file_date.strftime(
        '%b').lower()  # Get the full name of the month

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
    snow_cover_glacier = valid_classes.classes[
        valid_classes.classes == 1].count() / valid_classes.classes.count()

    return snow_cover_glacier


def xr_SGI_masked_topo(gdf_shapefiles, sgi_id, cfg):
    path_aspect = os.path.join(cfg.dataPath, path_SGI_topo, 'aspect')
    path_slope = os.path.join(cfg.dataPath, path_SGI_topo, 'slope')
    path_DEM = os.path.join(cfg.dataPath, path_SGI_topo, 'dem_HR')

    # Get SGI topo files
    aspect_gl = [f for f in os.listdir(path_aspect) if sgi_id in f][0]
    slope_gl = [f for f in os.listdir(path_slope) if sgi_id in f][0]
    dem_gl = [f for f in os.listdir(path_DEM) if sgi_id in f][0]

    metadata_aspect, grid_data_aspect = load_grid_file(
        join(path_aspect, aspect_gl))
    metadata_slope, grid_data_slope = load_grid_file(join(
        path_slope, slope_gl))
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
    gdf_mask_gl = gdf_shapefiles[gdf_shapefiles['sgi-id'] == sgi_id]

    # Mask over glacier outline
    mask, masked_aspect = extract_topo_over_outline(aspect_wgs84, gdf_mask_gl)
    mask, masked_slope = extract_topo_over_outline(slope_wgs84, gdf_mask_gl)
    mask, masked_dem = extract_topo_over_outline(dem_wgs84, gdf_mask_gl)

    # Create new dataset
    ds = xr.Dataset({
        "masked_aspect": masked_aspect,
        "masked_slope": masked_slope,
        "masked_elev": masked_dem,
        "glacier_mask": mask
    })

    # Mask elevations below 0 (bug values)
    ds["masked_elev"] = ds.masked_elev.where(ds.masked_elev >= 0, np.nan)
    return ds


def extract_topo_over_outline(aspect_xarray, glacier_polygon_gdf):
    """
    Extracts aspect values over a glacier outline (polygon) from an xarray in WGS84 coordinates.

    Parameters:
    - aspect_xarray: xarray.DataArray containing the aspect values, with WGS84 coordinates.
    - glacier_polygon_gdf: GeoPandas GeoDataFrame with the glacier outline polygon (WGS84 CRS).

    Returns:
    - A masked xarray.DataArray with aspect values only within the glacier polygon.
    """
    # Ensure the GeoDataFrame is in WGS84 CRS
    if glacier_polygon_gdf.crs is None:
        raise ValueError("Glacier GeoDataFrame must have a defined CRS.")
    if glacier_polygon_gdf.crs.to_epsg() != 4326:
        glacier_polygon_gdf = glacier_polygon_gdf.to_crs("EPSG:4326")

    # Get the x and y coordinates from the xarray
    lon_coords = aspect_xarray.coords['lon'].values
    lat_coords = aspect_xarray.coords['lat'].values

    # Compute the transform using rasterio's from_bounds
    transform = rasterio.transform.from_bounds(lon_coords.min(),
                                               lat_coords.min(),
                                               lon_coords.max(),
                                               lat_coords.max(),
                                               width=len(lon_coords),
                                               height=len(lat_coords))

    # Rasterize the glacier polygon
    shapes = [(geom, 1) for geom in glacier_polygon_gdf.geometry]
    mask = rasterio.features.rasterize(
        shapes,
        out_shape=(len(lat_coords),
                   len(lon_coords)),  # height (rows), width (cols)
        transform=transform,
        fill=0,
        dtype="int32")

    # Apply the mask to the xarray
    masked_aspect = aspect_xarray.where(np.flip(mask, 0) == 1)

    # Convert the mask to an xarray with the same coordinates as aspect_xarray
    mask_xarray = xr.DataArray(
        np.flip(mask, 0),
        coords=[aspect_xarray.coords['lat'], aspect_xarray.coords['lon']],
        dims=['lat', 'lon'])

    return mask_xarray, masked_aspect


def coarsenDS(ds, target_res_m=50):
    dx_m, dy_m = get_res_from_degrees(ds)  # Get resolution in meters

    # Compute resampling factor
    resampling_fac_lon = max(1, round(target_res_m / dx_m))
    resampling_fac_lat = max(1, round(target_res_m / dy_m))

    # print(f"Resampling factor: lon={resampling_fac_lon}, lat={resampling_fac_lat}")

    if dx_m < target_res_m or dy_m < target_res_m:
        # Coarsen non-binary variables with mean
        ds_non_binary = ds[['masked_slope', 'masked_aspect',
                            'masked_elev']].coarsen(lon=resampling_fac_lon,
                                                    lat=resampling_fac_lat,
                                                    boundary="trim").mean()

        # Coarsen glacier mask with max
        ds_glacier_mask = ds[['glacier_mask'
                              ]].coarsen(lon=resampling_fac_lon,
                                         lat=resampling_fac_lat,
                                         boundary="trim").reduce(np.max)

        # Merge back into a single dataset
        ds_res = xr.merge([ds_non_binary, ds_glacier_mask])
        return ds_res

    return ds


def get_rgi_sgi_ids(cfg, glacier_name):
    rgi_df = pd.read_csv(cfg.dataPath+path_glacier_ids, sep=',')
    rgi_df.rename(columns=lambda x: x.strip(), inplace=True)
    rgi_df.sort_values(by='short_name', inplace=True)
    rgi_df.set_index('short_name', inplace=True)

    # Handle 'clariden' separately due to its unique ID format
    if glacier_name == 'clariden':
        sgi_id = rgi_df.at[
            'claridenU',
            'sgi-id'].strip() if 'claridenU' in rgi_df.index else ''
        rgi_id = rgi_df.at['claridenU',
                           'rgi_id.v6'] if 'claridenU' in rgi_df.index else ''
        rgi_shp = rgi_df.at[
            'claridenU',
            'rgi_id_v6_2016_shp'] if 'claridenU' in rgi_df.index else ''
    else:
        sgi_id = rgi_df.at[
            glacier_name,
            'sgi-id'].strip() if glacier_name in rgi_df.index else ''
        rgi_id = rgi_df.at[glacier_name,
                           'rgi_id.v6'] if glacier_name in rgi_df.index else ''
        rgi_shp = rgi_df.at[
            glacier_name,
            'rgi_id_v6_2016_shp'] if glacier_name in rgi_df.index else ''

    return sgi_id, rgi_id, rgi_shp


def create_glacier_grid_SGI(
    glacierName,
    year,
    rgi_id,
    ds,
):
    glacier_indices = np.where(ds['glacier_mask'].values == 1)

    # Glacier mask as boolean array:
    gl_mask_bool = ds['glacier_mask'].values.astype(bool)

    lon_coords = ds['lon'].values
    lat_coords = ds['lat'].values

    lon = lon_coords[glacier_indices[1]]
    lat = lat_coords[glacier_indices[0]]

    # Create a DataFrame
    data_grid = {
        'RGIId': [rgi_id] * len(ds.masked_elev.values[gl_mask_bool]),
        'POINT_LAT': lat,
        'POINT_LON': lon,
        'aspect': ds.masked_aspect.values[gl_mask_bool],
        'slope': ds.masked_slope.values[gl_mask_bool],
        'topo': ds.masked_elev.values[gl_mask_bool],
    }
    df_grid = pd.DataFrame(data_grid)

    # Match to WGMS format:
    df_grid['POINT_ID'] = np.arange(1, len(df_grid) + 1)
    df_grid['N_MONTHS'] = 12
    df_grid['POINT_ELEVATION'] = df_grid[
        'topo']  # no other elevation available
    df_grid['POINT_BALANCE'] = 0  # fake PMB for simplicity (not used)

    # Add metadata that is not in WGMS dataset
    df_grid["PERIOD"] = "annual"
    df_grid['GLACIER'] = glacierName
    # Add the 'year' and date columns to the DataFrame
    df_grid['YEAR'] = np.tile(year, len(df_grid))
    df_grid['FROM_DATE'] = df_grid['YEAR'].apply(lambda x: str(x) + '1001')
    df_grid['TO_DATE'] = df_grid['YEAR'].apply(lambda x: str(x + 1) + '0930')

    return df_grid


def add_OGGM_features(df_y_gl, voi, path_OGGM):
    df_pmb = df_y_gl.copy()

    # Initialize empty columns for the variables
    for var in voi:
        df_pmb[var] = np.nan

    # Path to OGGM datasets
    path_to_data = path_OGGM + 'xr_grids/'

    # Group rows by RGIId
    grouped = df_pmb.groupby("RGIId")

    # Process each group
    for rgi_id, group in grouped:
        file_path = f"{path_to_data}{rgi_id}.zarr"

        try:
            # Load the xarray dataset for the current RGIId
            ds_oggm = xr.open_dataset(file_path)
        except FileNotFoundError:
            print(f"File not found for RGIId: {rgi_id}")
            continue

        # Define the coordinate transformation
        transf = pyproj.Transformer.from_proj(
            pyproj.CRS.from_user_input("EPSG:4326"),  # Input CRS (WGS84)
            pyproj.CRS.from_user_input(
                ds_oggm.pyproj_srs),  # Output CRS from dataset
            always_xy=True)

        # Transform all coordinates in the group
        lon, lat = group["POINT_LON"].values, group["POINT_LAT"].values
        x_stake, y_stake = transf.transform(lon, lat)
        # Select nearest values for all points
        try:
            stake = ds_oggm.sel(x=xr.DataArray(x_stake, dims="points"),
                                y=xr.DataArray(y_stake, dims="points"),
                                method="nearest")

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

    metadata_aspect, grid_data_aspect = load_grid_file(
        join(path_aspect, aspect_gl))
    metadata_slope, grid_data_slope = load_grid_file(join(
        path_slope, slope_gl))

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
    aspect_resampled = aspect_wgs84.interp_like(ds_gl["glacier_mask"],
                                                method="nearest")
    slope_resampled = slope_wgs84.interp_like(ds_gl["glacier_mask"],
                                              method="nearest")

    # Compute new resolution (after downsampling)
    dx_m, dy_m = get_res_from_degrees(ds_gl["glacier_mask"])
    # print(f"New resolution (after downsampling): {dx_m} x {dy_m} meters")

    # Step 2: Apply the glacier mask
    masked_aspect = aspect_resampled.where(ds_gl["glacier_mask"] == 1, np.nan)
    masked_slope = slope_resampled.where(ds_gl["glacier_mask"] == 1, np.nan)

    # Resample DEM to the same resolution
    dem_resampled = ds_gl["dem"].interp_like(ds_gl["glacier_mask"],
                                             method="nearest")

    # Create a new dataset
    ds = xr.Dataset({
        "masked_aspect": masked_aspect,
        "masked_elev": dem_resampled,
        "masked_slope": masked_slope,
        "glacier_mask": ds_gl["glacier_mask"]
    })

    # Mask elevations below 0 (to remove erroneous values)
    ds["masked_elev"] = ds.masked_elev.where(ds.masked_elev >= 0, np.nan)

    return ds


def get_res_from_degrees(ds):
    # Get central latitude (mean of lat values)
    lat_center = ds.lat.values.mean()

    # Earth's approximate conversion factor (meters per degree)
    meters_per_degree_lat = 111320  # Roughly constant for latitude
    meters_per_degree_lon = 111320 * np.cos(
        np.radians(lat_center))  # Adjust for longitude

    # Compute resolution
    dx_m = np.round(
        abs(ds.lon[1] - ds.lon[0]).values * meters_per_degree_lon, 3)
    dy_m = np.round(
        abs(ds.lat[1] - ds.lat[0]).values * meters_per_degree_lat, 3)

    return dx_m, dy_m


def get_gl_area(cfg):
    # Load glacier metadata
    rgi_df = pd.read_csv(cfg.dataPath+path_glacier_ids, sep=',')
    rgi_df.rename(columns=lambda x: x.strip(), inplace=True)
    rgi_df.sort_values(by='short_name', inplace=True)
    rgi_df.set_index('short_name', inplace=True)

    # Load the shapefile
    shapefile_path = os.path.join(cfg.dataPath, path_SGI_topo, 'inventory_sgi2016_r2020',
                                  'SGI_2016_glaciers_copy.shp')
    gdf_shapefiles = gpd.read_file(shapefile_path)

    gl_area = {}

    for glacierName in rgi_df.index:
        if glacierName == 'clariden':
            rgi_shp = rgi_df.loc[
                'claridenL',
                'rgi_id_v6_2016_shp'] if 'claridenL' in rgi_df.index else None
        else:
            rgi_shp = rgi_df.loc[glacierName, 'rgi_id_v6_2016_shp']

        # Skip if rgi_shp is not found
        if pd.isna(rgi_shp) or rgi_shp is None:
            continue

        # Ensure matching data types
        rgi_shp = str(rgi_shp)
        gdf_mask_gl = gdf_shapefiles[gdf_shapefiles.RGIId.astype(str) ==
                                     rgi_shp]

        # If a glacier is found, get its area
        if not gdf_mask_gl.empty:
            gl_area[glacierName] = gdf_mask_gl.Area.iloc[
                0]  # Use .iloc[0] safely

    return gl_area

def load_grid_file(filepath):
    with open(filepath, 'r') as file:
        # Read metadata
        metadata = {}
        for _ in range(6):  # First 6 lines are metadata
            line = file.readline().strip().split()
            metadata[line[0].lower()] = float(line[1])

        # Get ncols from metadata to control the number of columns
        ncols = int(metadata['ncols'])
        nrows = int(metadata['nrows'])

        # Initialize an empty list to store rows of the grid
        data = []

        # Read the grid data line by line
        row_ = []
        for line in file:
            row = line.strip().split()
            if len(row_) < ncols:
                row_ += row
            if len(row_) == ncols:
                data.append([
                    np.nan
                    if float(x) == metadata['nodata_value'] else float(x)
                    for x in row_
                ])
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
    return pd.to_datetime(month + '-' + day + '-' + year)



def transformDates(df_or):
    """Some dates are missing in the original GLAMOS data and need to be corrected.

    Args:
        df_or (pd.DataFrame): Raw GLAMOS DataFrame

    Returns:
        pd.DataFrame: DataFrame with corrected dates
    """
    df = df_or.copy()

    # Ensure 'date0' and 'date1' are datetime objects
    df['date0'] = df['date0'].apply(lambda x: datetime_obj(x))
    df['date1'] = df['date1'].apply(lambda x: datetime_obj(x))

    # Initialize new columns with NaT (not np.nan, since we'll use datetime later)
    df['date_fix0'] = pd.NaT
    df['date_fix1'] = pd.NaT

    # Assign fixed dates using .loc to avoid chained assignment warning
    for i in range(len(df)):
        year = df.loc[i, 'date0'].year
        df.loc[i, 'date_fix0'] = pd.Timestamp(f"{year}-10-01")
        df.loc[i, 'date_fix1'] = pd.Timestamp(f"{year + 1}-09-30")

    # Format original dates for WGMS
    df['date0'] = df['date0'].apply(lambda x: x.strftime('%Y%m%d'))
    df['date1'] = df['date1'].apply(lambda x: x.strftime('%Y%m%d'))

    return df

def check_missing_years(folder_path, glacier_name, period):
    start_year, end_year = period
    expected_years = set(range(start_year, end_year + 1))

    # Extract years from filenames
    available_years = set()
    pattern = re.compile(rf'{glacier_name}_(\d{{4}})_annual\.zarr')

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            year = int(match.group(1))
            available_years.add(year)

    missing_years = expected_years - available_years
    if missing_years:
        return True
    else:
        return False

def process_geodetic_mass_balance_comparison(
    glacier_list,
    path_SMB_GLAMOS_csv,
    periods_per_glacier,
    geoMB_per_glacier,
    gl_area,
    test_glaciers,
    path_predictions,
    cfg
):
    # Storage lists for results
    mbm_mb_mean, glamos_mb_mean, geodetic_mb = [], [], []
    gl, period_len, gl_type, area = [], [], [], []
    start_year, end_year = [], []

    for glacier_name in tqdm(glacier_list, desc="Processing glaciers"):

        # Check if GLAMOS file exists
        glamos_file = os.path.join(path_SMB_GLAMOS_csv, "fix", f"{glacier_name}_fix.csv")
        if not os.path.exists(glamos_file):
            print(f"Skipping {glacier_name}: GLAMOS file not found.")
            continue

        # Load GLAMOS data
        GLAMOS_glwmb = get_GLAMOS_glwmb(glacier_name, cfg)
        if GLAMOS_glwmb is None:
            print(f"Skipping {glacier_name}: Failed to load GLAMOS data.")
            continue

        # Get periods and geodetic MBs
        periods = periods_per_glacier.get(glacier_name, [])
        geoMBs = geoMB_per_glacier.get(glacier_name, [])

        if not periods or not geoMBs:
            print(f"Skipping {glacier_name}: No geodetic mass balance data available.")
            continue

        # Path to model predictions
        folder_path = os.path.join(path_predictions, glacier_name)

        for period in periods:
            mbm_mb, glamos_mb = [], []

            if period[1] == 2021 and glacier_name == 'silvretta':
                continue

            # Skip if required years are missing
            if check_missing_years(folder_path, glacier_name, period):
                print(f"Skipping {glacier_name} {period}: Missing years")
                continue

            for year in range(period[0], period[1] + 1):
                file_path = os.path.join(folder_path, f"{glacier_name}_{year}_annual.zarr")

                if not os.path.exists(file_path):
                    print(f"Warning: Missing MBM file for {glacier_name} ({year}).")
                    mbm_mb.append(np.nan)
                else:
                    ds = xr.open_dataset(file_path)
                    mbm_mb.append(ds["pred_masked"].mean().values)

                glamos_mb.append(GLAMOS_glwmb["GLAMOS Balance"].get(year, np.nan))

            mbm_mb_mean.append(np.nanmean(mbm_mb))
            glamos_mb_mean.append(np.nanmean(glamos_mb))
            geodetic_mb.append(geoMBs[periods.index(period)])
            gl.append(glacier_name)
            gl_type.append(glacier_name in test_glaciers)
            period_len.append(period[1] - period[0])
            area.append(gl_area.get(glacier_name, np.nan))
            start_year.append(period[0])
            end_year.append(period[1])

    # Create and return DataFrame
    df_all = pd.DataFrame({
        "MBM MB": mbm_mb_mean,
        "GLAMOS MB": glamos_mb_mean,
        "Geodetic MB": geodetic_mb,
        "GLACIER": gl,
        "Period Length": period_len,
        "Test Glacier": gl_type,
        "Area": area,
        "start_year": start_year,
        "end_year": end_year
    })

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
        for geodetic_period in periods_per_glacier[glacier_name]:
            mask = ((geodetic_mb.glacier_name == glacier_name) &
                    (geodetic_mb.Astart == geodetic_period[0]) &
                    (geodetic_mb.Aend == geodetic_period[1]))
            geodetic_MB_target.append(geodetic_mb[mask].Bgeod.values[0])

        return np.array(geodetic_MB_target)
    else:
        return {glacier_name: prepareGeoTargets(
            geodetic_mb,
            periods_per_glacier,
            glacier_name=glacier_name
        ) for glacier_name in periods_per_glacier}

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

    # Iterate through the DataFrame rows
    for _, row in geodetic_mb.iterrows():
        glacier_name = row['glacier_name']
        start_year = row['Astart']
        end_year = row['Aend']
        geoMB = row['Bgeod']

        # Append the (start, end) tuple to the glacier's list
        # Only if period is longer than 5 years
        if end_year - start_year >= 5:
            periods_per_glacier[glacier_name].append((start_year, end_year))
            geoMB_per_glacier[glacier_name].append(geoMB)

    # sort by glacier_list
    periods_per_glacier = dict(sorted(periods_per_glacier.items()))
    geoMB_per_glacier = dict(sorted(geoMB_per_glacier.items()))

    return periods_per_glacier, geoMB_per_glacier
