import os
import pandas as pd
from scipy.ndimage import gaussian_filter
import pyproj
import numpy as np
from collections import defaultdict
from dateutil.relativedelta import relativedelta

from scripts.config_CH import *
from scripts.glamos_preprocess import *

def get_GLAMOS_glwmb(glacier_name):
    """
    Loads and processes GLAMOS glacier-wide mass balance data.

    Parameters:
    -----------
    glacier_name : str
        The name of the glacier.

    Returns:
    --------
    pd.DataFrame or None
        A DataFrame with columns ['YEAR', 'GLAMOS Balance'] indexed by 'YEAR',
        or None if the file is missing.
    """

    # Construct file path safely
    file_path = os.path.join(path_SMB_GLAMOS_csv, "fix",
                             f"{glacier_name}_fix.csv")

    # Check if file exists
    if not os.path.exists(file_path):
        print(
            f"Warning: GLAMOS data file not found for {glacier_name}. Skipping..."
        )
        return None

    # Load CSV and transform dates
    df = pd.read_csv(file_path)
    df = transformDates(df)

    # Remove duplicates based on the date column
    df = df.drop_duplicates(subset=["date1"])

    # Ensure required columns exist
    required_columns = {"date1", "Annual Balance"}
    if not required_columns.issubset(df.columns):
        print(
            f"Warning: Missing required columns in {glacier_name} GLAMOS data. Skipping..."
        )
        return None

    # Extract year from date and normalize balance
    df["YEAR"] = pd.to_datetime(df["date1"]).dt.year.astype("int64")
    df["GLAMOS Balance"] = df[
        "Annual Balance"] / 1000  # Convert to meters water equivalent

    # Select relevant columns and set index
    return df[["YEAR", "GLAMOS Balance"]].set_index("YEAR")


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

    # Reorder the dimensions to be (lon, lat)
    # data_array = data_array.transpose("lon", "lat")

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