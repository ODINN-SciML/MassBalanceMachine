import numpy as np
import os
from os import listdir
from os.path import isfile, join
import xarray as xr
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from datetime import datetime
from collections import defaultdict
from dateutil.relativedelta import relativedelta
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import Polygon, LineString, Point
from pyproj import Transformer
import rasterio.features
from rasterio.transform import from_bounds
from tqdm.notebook import tqdm
import pyproj
from pyproj import Transformer
from rasterio import features
from shapely.geometry import shape

def toRaster(gdf, lon, lat, file_name, source_crs='EPSG:4326'):
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
        data_array[lat_idx, lon_idx] = row['pred_masked']

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


def reproject_raster_to_lv95(
        input_raster,
        output_raster,
        dst_crs='EPSG:2056'  # Destination CRS (Swiss LV95) or EPSG:21781
):
    # Define the source and destination CRS
    src_crs = 'EPSG:4326'  # Original CRS (lat/lon)

    # Open the source raster
    with rasterio.open(input_raster) as src:
        # Calculate the transform and dimensions for the destination CRS
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)

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
                    resampling=Resampling.
                    nearest  # You can also use other methods, like bilinear
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


def GaussianFilter(ds, variable_name='pred_masked', sigma=1):
    """
    Apply Gaussian filter only to the specified variable in the xarray.Dataset.
    
    Parameters:
    - ds (xarray.Dataset): Input dataset
    - variable_name (str): The name of the variable to apply the filter to (default 'pred_masked')
    - sigma (float): The standard deviation for the Gaussian filter
    
    Returns:
    - xarray.Dataset: New dataset with smoothed variable
    """
    # Check if the variable exists in the dataset
    if variable_name not in ds:
        raise ValueError(
            f"Variable '{variable_name}' not found in the dataset.")

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

    # Create a new dataset with the smoothed data
    smoothed_dataset = ds.copy()  # Make a copy of the original dataset
    smoothed_dataset[
        variable_name] = smoothed_data  # Replace the original variable with the smoothed one

    return smoothed_dataset


def TransformToRaster(filename_nc,
                      filename_tif,
                      path_nc_wgs84,
                      path_tif_wgs84,
                      path_tif_lv95,
                      dst_crs='EPSG:2056'):
    ds_latlon = xr.open_dataset(path_nc_wgs84 + filename_nc)

    # Smoothing
    ds_latlon_g = GaussianFilter(ds_latlon)

    # Convert to GeoPandas
    gdf, lon, lat = toGeoPandas(ds_latlon_g)

    # Reproject to LV95 (EPSG:2056) swiss coordinates
    # gdf_lv95 = gdf.to_crs("EPSG:2056")

    createPath(path_tif_wgs84)
    createPath(path_tif_lv95)

    # Convert to raster and save
    raster_data, extent = toRaster(gdf,
                                   lon,
                                   lat,
                                   file_name=path_tif_wgs84 + filename_tif)

    # Reproject raster to Swiss coordinates (LV95)
    reproject_raster_to_lv95(path_tif_wgs84 + filename_tif,
                             path_tif_lv95 + filename_tif,
                             dst_crs=dst_crs)

    # Make classes map of snow/ice:
    # Replace values: below 0 with 3, above 0 with 1
    gdf_class = gdf.copy()
    tol = 0
    gdf_class.loc[gdf['pred_masked'] <= 0 + tol, 'pred_masked'] = 3
    gdf_class.loc[gdf['pred_masked'] > 0 + tol, 'pred_masked'] = 1

    path_class_tif_lv95 = path_tif_lv95 + 'classes/'
    path_class_tif_wgs84 = path_tif_wgs84 + 'classes/'

    createPath(path_class_tif_lv95)
    createPath(path_class_tif_wgs84)

    # Convert to raster and save
    raster_data, extent = toRaster(gdf_class,
                                   lon,
                                   lat,
                                   file_name=path_class_tif_wgs84 +
                                   filename_tif)

    # Reproject raster to Swiss coordinates (LV95)
    reproject_raster_to_lv95(path_class_tif_wgs84 + filename_tif,
                             path_class_tif_lv95 + filename_tif,
                             dst_crs=dst_crs)

    return gdf, gdf_class, raster_data, extent


def createPath(path):
    if not os.path.exists(path):
        os.makedirs(path)


# empties a folder
def emptyfolder(path):
    if os.path.exists(path):
        onlyfiles = [f for f in os.listdir(path) if isfile(join(path, f))]
        for f in onlyfiles:
            os.remove(path + f)
    else:
        createPath(path)


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


def IceSnowCover(gdf_class, gdf_class_raster):
    # Exclude pixels with "classes" 5 (cloud) in gdf_class_raster
    valid_classes = gdf_class[gdf_class_raster.classes != 5]

    # Calculate percentage of snow cover (class 1) in valid classes
    snow_cover_glacier = valid_classes.classes[
        valid_classes.classes == 1].count() / valid_classes.classes.count()

    return snow_cover_glacier


def replace_clouds_with_nearest_neighbor(gdf,
                                         class_column='classes',
                                         cloud_class=5):
    """
    Replace cloud pixels in a GeoDataFrame with the most common class among their 
    nearest neighbors, excluding NaN values.

    Parameters:
    - gdf (GeoDataFrame): GeoPandas DataFrame containing pixel data with a geometry column.
    - class_column (str): The column name representing the class of each pixel (integer classes).
    - cloud_class (int): The class to be replaced (e.g., 1 for cloud).
    - n_neighbors (int): The number of nearest neighbors to consider for majority voting.

    Returns:
    - GeoDataFrame: Updated GeoDataFrame with cloud classes replaced.
    """
    # Separate cloud pixels and non-cloud pixels
    cloud_pixels = gdf[gdf[class_column] == cloud_class]
    non_cloud_pixels = gdf[gdf[class_column] != cloud_class]

    # Remove NaN values from non-cloud pixels
    non_cloud_pixels = non_cloud_pixels[non_cloud_pixels[class_column].notna()]

    # If no clouds or no non-NaN non-cloud pixels, return the original GeoDataFrame
    if cloud_pixels.empty or non_cloud_pixels.empty:
        return gdf

    # Extract coordinates for nearest-neighbor search
    cloud_coords = np.array(
        list(cloud_pixels.geometry.apply(lambda geom: (geom.x, geom.y))))
    non_cloud_coords = np.array(
        list(non_cloud_pixels.geometry.apply(lambda geom: (geom.x, geom.y))))

    # Perform nearest-neighbor search
    nbrs = NearestNeighbors(n_neighbors=1,
                            algorithm='auto').fit(non_cloud_coords)
    distances, indices = nbrs.kneighbors(cloud_coords)

    # Map nearest neighbor's class to cloud pixels
    nearest_classes = non_cloud_pixels.iloc[
        indices.flatten()][class_column].values
    gdf.loc[cloud_pixels.index, class_column] = nearest_classes

    return gdf


def snowline(gdf, class_value=1, percentage_threshold=20):
    """
    Find the first elevation band where the percentage of the given class exceeds the specified threshold
    and add a boolean column to gdf indicating the selected band.
    
    Parameters:
    - gdf (GeoDataFrame): Input GeoDataFrame with 'elev_band' and 'classes' columns
    - class_value (int): The class value to check for (default is 1 for snow)
    - percentage_threshold (float): The percentage threshold to exceed (default is 20%)
    
    Returns:
    - gdf (GeoDataFrame): GeoDataFrame with an additional boolean column indicating the selected elevation band
    - first_band (int): The first elevation band that meets the condition
    """
    # Step 1: Group by elevation band and calculate the percentage of 'class_value' in each band
    band_class_counts = gdf.groupby('elev_band')['classes'].value_counts(
        normalize=True)

    # Step 2: Calculate the percentage of the specified class in each band
    class_percentage = band_class_counts.xs(
        class_value, level=1) * 100  # Multiply by 100 to convert to percentage

    # Step 3: Find the first band where the class percentage exceeds the threshold
    first_band = None
    for elev_band, percentage in class_percentage.items():
        if percentage >= percentage_threshold:
            first_band = elev_band
            break

    if first_band is not None:
        # Step 4: Add a new column to the GeoDataFrame to indicate the first elevation band
        gdf['selected_band'] = gdf['elev_band'] == first_band
    else:
        # If no band meets the threshold, the new column will be False for all rows
        gdf['selected_band'] = False

    return gdf, first_band


def classify_elevation_bands(gdf_glacier, band_size=50):
    """
    Classify elevation into bands based on the 'elev_masked' column in the GeoDataFrame.

    Parameters:
        gdf_glacier (GeoDataFrame): A GeoDataFrame containing an 'elev_masked' column.
        band_size (int): The size of each elevation band.

    Returns:
        GeoDataFrame: The input GeoDataFrame with an additional 'elev_band' column.
    """
    # Ensure the 'elev_masked' column exists and contains valid data
    if 'elev_masked' not in gdf_glacier.columns:
        raise ValueError("GeoDataFrame does not contain 'elev_masked' column")

    # Handle NaN values in 'elev_masked' and classify into elevation bands
    gdf_glacier['elev_band'] = (
        gdf_glacier['elev_masked'].fillna(
            -1)  # Replace NaN with a placeholder (e.g., -1 or another value)
        .floordiv(band_size) * band_size  # Calculate the elevation band
    )

    # Optionally set the 'elev_band' of NaN entries back to NaN
    gdf_glacier.loc[gdf_glacier['elev_masked'].isna(), 'elev_band'] = None

    return gdf_glacier


def AddSnowline(gdf_glacier_corr, band_size=100, percentage_threshold=50):
    # Add snowline
    # Remove weird border effect
    #gdf_glacier_corr = gdf_glacier_corr[gdf_glacier_corr.dis_masked > 10]

    gdf_glacier_corr = classify_elevation_bands(gdf_glacier_corr, band_size)

    snowline(gdf_glacier_corr,
             class_value=1,
             percentage_threshold=percentage_threshold)

    return gdf_glacier_corr


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
        #print(f"ncols: {ncols}, nrows: {nrows}")

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
        #print(grid_data.shape)
        assert grid_data.shape == (nrows, ncols)

    return metadata, grid_data


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


def xyzn_to_dataframe(xyzn_filename):
    """
    Reads a .xyzn file and converts it into a pandas DataFrame with columns x_pos, y_pos, and z_pos.

    Parameters:
    - xyzn_filename: Path to the .xyzn file.

    Returns:
    - A pandas DataFrame containing x, y, and z positions.
    """
    # Step 1: Read the .xyzn file and extract X, Y, Z positions
    data = []
    with open(xyzn_filename, 'r') as file:
        for line in file:
            # Split each line by space and extract the first three values (X, Y, Z)
            values = list(map(float, line.split()))
            x, y, z = values[0], values[1], values[2]
            data.append([x, y, z])  # Store X, Y, and Z

    # Step 2: Convert the list to a pandas DataFrame with columns x_pos, y_pos, z_pos
    df = pd.DataFrame(data, columns=['x_pos', 'y_pos', 'z_pos'])

    return df


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
    transformer = Transformer.from_crs("EPSG:2056",
                                       "EPSG:4326",
                                       always_xy=True)

    # Sample CH1903+ / LV95 coordinates (Easting and Northing)

    # Transform to Latitude and Longitude (WGS84)
    lon, latitude = transformer.transform(df.x_pos, df.y_pos)

    df['lat'] = latitude
    df['lon'] = lon
    df.drop(['x_pos', 'y_pos', 'z_pos'], axis=1, inplace=True)
    return df


def draw_glacier_outline(xarray_data, xyzn_filename):
    """
    Add a glacier outline binary mask to an existing xarray. The glacier coordinates 
    might not perfectly align with the xarray grid, but the closest grid points will be used.

    Parameters:
    - xarray_data: The existing xarray with 'x' and 'y' coordinates.
    - xyzn_filename: The .xyzn file containing the glacier outline coordinates.

    Returns:
    - Updated xarray with a new variable `glacier_outline`.
    """
    # Step 1: Read the glacier coordinates from the .xyzn file
    df = xyzn_to_dataframe(
        xyzn_filename
    )  # This function must read the file and return a DataFrame
    if not all(col in df.columns for col in ['x_pos', 'y_pos']):
        raise ValueError("The dataframe must contain 'X' and 'Y' columns.")

    # Step 2: Extract the grid's x and y coordinates from the existing xarray
    x_coords = xarray_data.coords['x'].values
    y_coords = xarray_data.coords['y'].values

    # Step 3: Initialize the glacier outline mask with zeros (matching shape y, x)
    glacier_mask = np.zeros((len(y_coords), len(x_coords)), dtype=int)

    # Step 4: Loop through glacier coordinates and map to closest grid points
    for _, row in df.iterrows():
        gx, gy = row['x_pos'], row['y_pos']

        # Find the closest grid point (using absolute difference)
        closest_x_idx = (np.abs(x_coords - gx)).argmin()
        closest_y_idx = (np.abs(y_coords - gy)).argmin()

        # Update the glacier mask
        glacier_mask[closest_y_idx,
                     closest_x_idx] = 1  # Note: y index first, then x index

    # Step 5: Create an xarray DataArray for the glacier outline mask
    glacier_outline = xr.DataArray(glacier_mask,
                                   coords={
                                       "y": y_coords,
                                       "x": x_coords
                                   },
                                   dims=["y", "x"],
                                   name="glacier_outline")

    return glacier_outline


def xarray_to_geodataframe(xarray_data, var_name, crs=None):
    """
    Converts an xarray.DataArray into a GeoPandas GeoDataFrame with point geometries.

    Parameters:
    - xarray_data: xarray.DataArray or xarray.Dataset
    - var_name: Name of the variable to include in the GeoDataFrame.
    - crs: Coordinate Reference System (e.g., "EPSG:4326") for the GeoDataFrame.

    Returns:
    - GeoPandas GeoDataFrame with x, y coordinates and the variable values.
    """
    # Ensure xarray_data is a DataArray
    if isinstance(xarray_data, xr.Dataset):
        data_array = xarray_data[var_name]
    elif isinstance(xarray_data, xr.DataArray):
        data_array = xarray_data
    else:
        raise ValueError(
            "Input must be an xarray.DataArray or xarray.Dataset.")

    # Flatten the DataArray into a 1D array
    flat_values = data_array.values.flatten()
    lon_coords, lat_coords = data_array.coords[
        'lon'].values, data_array.coords['lat'].values

    # Create a meshgrid of x and y coordinates
    grid_lon, grid_lat = np.meshgrid(lon_coords, lat_coords)

    # Flatten the coordinate grids
    flat_lon = grid_lon.flatten()
    flat_lat = grid_lat.flatten()

    # Create geometries (Point objects) for the GeoDataFrame
    geometries = [Point(lon, lat) for lon, lat in zip(flat_lon, flat_lat)]

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {"value": flat_values},  # Add variable values as a column
        geometry=geometries,  # Add geometries
        crs=crs  # Set CRS if provided
    )

    return gdf

def xarray_to_geopolygon(xarray_data, var_name, crs=None):
    """
    Converts an xarray.DataArray into a GeoPandas GeoDataFrame with polygon geometries.
    
    Parameters:
    - xarray_data: xarray.DataArray or xarray.Dataset
    - var_name: Name of the variable to include in the GeoDataFrame.
    - crs: Coordinate Reference System (e.g., "EPSG:4326") for the GeoDataFrame.
    
    Returns:
    - GeoPandas GeoDataFrame with polygon geometries representing valid (non-NaN) areas.
    """
    # Ensure xarray_data is a DataArray
    if isinstance(xarray_data, xr.Dataset):
        data_array = xarray_data[var_name]
    elif isinstance(xarray_data, xr.DataArray):
        data_array = xarray_data
    else:
        raise ValueError("Input must be an xarray.DataArray or xarray.Dataset.")

    # Create a binary mask (1 where not NaN, 0 where NaN)
    mask = np.where(np.isnan(data_array.values), 0, 1).astype(np.uint8)

    # Define transform assuming regular grid spacing
    lon_coords, lat_coords = data_array.coords['lon'].values, data_array.coords['lat'].values
    transform = [lon_coords[0], lon_coords[1] - lon_coords[0], 0,
                 lat_coords[0], 0, lat_coords[1] - lat_coords[0]]  # Affine-like transform

    # Convert raster to vector polygons
    shapes = features.shapes(mask, transform=transform)

    # Extract polygons where mask is 1
    polygons = [shape(geom) for geom, value in shapes if value == 1]

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)

    return gdf


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
    transform = from_bounds(lon_coords.min(),
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


def xr_SGI_masked_topo(rgi_shp, gdf_shapefiles, path_aspect, path_slope,
                       path_DEM, sgi_id):
    # SGI topo files
    aspect_gl = [f for f in os.listdir(path_aspect) if sgi_id in f][0]
    slope_gl = [f for f in os.listdir(path_slope) if sgi_id in f][0]
    dem_gl = [f for f in os.listdir(path_DEM) if sgi_id in f][0]

    metadata_aspect, grid_data_aspect = load_grid_file(join(path_aspect, aspect_gl))
    metadata_slope, grid_data_slope = load_grid_file(join(path_slope, slope_gl))
    metadata_dem, grid_data_dem = load_grid_file(join(path_DEM, dem_gl))

    # convert to xarray
    aspect = convert_to_xarray_geodata(grid_data_aspect, metadata_aspect)
    slope = convert_to_xarray_geodata(grid_data_slope, metadata_slope)
    dem = convert_to_xarray_geodata(grid_data_dem, metadata_dem)

    # Transform to WGS84
    aspect_wgs84 = transform_xarray_coords_lv95_to_wgs84(aspect)
    slope_wgs84 = transform_xarray_coords_lv95_to_wgs84(slope)
    dem_wgs84 = transform_xarray_coords_lv95_to_wgs84(dem)

    # 2016 shapefile of glacier
    gdf_mask_gl = gdf_shapefiles[gdf_shapefiles.RGIId == rgi_shp]

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

def xr_GLAMOS_masked_topo(path_aspect, path_slope, sgi_id, ds_gl):
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

    # Step 1: Upscale glacier mask to match aspect resolution
    glacier_mask_resampled = ds_gl["glacier_mask"].interp_like(aspect_wgs84, method="nearest")
    # Replace NaN values in glacier mask by 0 
    glacier_mask_resampled = glacier_mask_resampled.fillna(0)
    
    dem_resampled = ds_gl["dem"].interp_like(aspect_wgs84, method="nearest")

    # Step 2: Apply the glacier mask to aspect data (keeping values where glacier_mask == 1)
    masked_aspect = aspect_wgs84.where(glacier_mask_resampled == 1, np.nan)
    masked_slope = slope_wgs84.where(glacier_mask_resampled == 1, np.nan)
    
    # Create new dataset with downsampled variables
    ds = xr.Dataset({
        "masked_aspect": masked_aspect,
        "masked_elev": dem_resampled,
        "masked_slope": masked_slope,
        "glacier_mask": glacier_mask_resampled
    })

    # Mask elevations below 0 (bug values)
    ds["masked_elev"] = ds.masked_elev.where(ds.masked_elev >= 0, np.nan)
    
    return ds


def coarsenDS(ds, resampling_fac = 3):
    # Coarson to 30 m resolution
    # Coarsen non-binary variables with mean
    ds_non_binary = ds[['masked_slope', 'masked_aspect',
                        'masked_elev']].coarsen(lon=resampling_fac, lat=resampling_fac,
                                                boundary="trim").mean()

    # Coarsen glacier mask with max
    ds_glacier_mask = ds[['glacier_mask']].coarsen(lon=resampling_fac, lat=resampling_fac,
                                                boundary="trim").reduce(np.max)

    # Merge back into a single dataset
    ds_res = xr.merge([ds_non_binary, ds_glacier_mask])
    return ds_res


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
        file_path = f"{path_to_data}{rgi_id}.nc"
        
        try:
            # Load the xarray dataset for the current RGIId
            ds_oggm = xr.open_dataset(file_path)
        except FileNotFoundError:
            print(f"File not found for RGIId: {rgi_id}")
            continue

        # Define the coordinate transformation
        transf = pyproj.Transformer.from_proj(
            pyproj.CRS.from_user_input("EPSG:4326"),  # Input CRS (WGS84)
            pyproj.CRS.from_user_input(ds_oggm.pyproj_srs),  # Output CRS from dataset
            always_xy=True
        )

        # Transform all coordinates in the group
        lon, lat = group["POINT_LON"].values, group["POINT_LAT"].values
        x_stake, y_stake = transf.transform(lon, lat)
        # Select nearest values for all points
        try:
            stake = ds_oggm.sel(
                x=xr.DataArray(x_stake, dims="points"),
                y=xr.DataArray(y_stake, dims="points"),
                method="nearest"
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


