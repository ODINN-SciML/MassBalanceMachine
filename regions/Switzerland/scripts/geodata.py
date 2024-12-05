import matplotlib.pyplot as plt
import numpy as np
from cartopy import crs as ccrs, feature as cfeature
import os
from os import listdir
from os.path import isfile, join
import xarray as xr
from matplotlib.colors import to_hex
import geopandas as gpd
from shapely.geometry import Point, box
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
import glob
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from datetime import datetime
from collections import defaultdict
from dateutil.relativedelta import relativedelta
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode


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
        smoothed_data = xr.DataArray(
            smoothed_data,
            dims=data_array.dims,
            coords=data_array.coords,
            attrs=data_array.attrs).where(
                mask)  # Apply the mask to restore NaNs

        # Add the smoothed data to the new Dataset
        smoothed_dataset[var_name] = smoothed_data
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

    # Make classification map of snow/ice:
    # Replace values: below 0 with 3, above 0 with 1
    gdf_class = gdf.copy()
    tol = 0
    gdf_class.loc[gdf['data'] <= 0 + tol, 'data'] = 3
    gdf_class.loc[gdf['data'] > 0 + tol, 'data'] = 1

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


def resampleRaster(gdf_glacier, gdf_raster):
    # Clip raster to glacier extent
    # Step 1: Get the bounding box of the points GeoDataFrame
    bounding_box = gdf_glacier.total_bounds  # [minx, miny, maxx, maxy]
    raster_bounds = gdf_raster.total_bounds  # [minx, miny, maxx, maxy]

    # Problem 1: check if glacier bounds are within raster bounds
    if not (bounding_box[0] >= raster_bounds[0]
            and  # minx of glacier >= minx of raster
            bounding_box[1] >= raster_bounds[1]
            and  # miny of glacier >= miny of raster
            bounding_box[2] <= raster_bounds[2]
            and  # maxx of glacier <= maxx of raster
            bounding_box[3]
            <= raster_bounds[3]  # maxy of glacier <= maxy of raster
            ):
        return 0

    # Step 2: Create a rectangular geometry from the bounding box
    bbox_polygon = box(*bounding_box)

    # Problem 2: Glacier is in regions where raster is NaN
    gdf_clipped = gpd.clip(gdf_raster, bbox_polygon)
    if gdf_clipped.empty:
        return 1

    # Step 3: Clip the raster-based GeoDataFrame to this bounding box
    gdf_clipped = gdf_raster[gdf_raster.intersects(bbox_polygon)]

    # Optionally, further refine the clipping if exact match is needed
    gdf_clipped = gpd.clip(gdf_raster, bbox_polygon)

    # Resample clipped raster to glacier points
    # Extract coordinates and values from gdf_clipped
    clipped_coords = np.array([(geom.x, geom.y)
                               for geom in gdf_clipped.geometry])
    clipped_values = gdf_clipped['data'].values

    # Extract coordinates from gdf_glacier
    points_coords = np.array([(geom.x, geom.y)
                              for geom in gdf_glacier.geometry])

    # Build a KDTree for efficient nearest-neighbor search
    tree = cKDTree(clipped_coords)

    # Query the tree for the nearest neighbor to each point in gdf_glacier
    distances, indices = tree.query(points_coords)

    # Assign the values from the nearest neighbors
    gdf_clipped_res = gdf_glacier.copy()
    gdf_clipped_res['data'] = clipped_values[indices]

    # Assuming 'value' is the column storing the resampled values
    gdf_clipped_res['data'] = np.where(
        gdf_glacier['data'].isna(),  # Check where original values are NaN
        np.nan,  # Assign NaN to those locations
        gdf_clipped_res['data'],  # Keep the resampled values elsewhere
    )
    return gdf_clipped_res



def createRaster(input_raster):
    # Open the raster
    with rasterio.open(input_raster) as src:
        data = src.read(1)  # Read first band
        transform = src.transform
        crs = src.crs

    # Get indices of non-NaN values
    rows, cols = np.where(data != src.nodata)
    values = data[rows, cols]

    # Convert raster cells to points
    points = [
        Point(transform * (col + 0.5, row + 0.5))
        for row, col in zip(rows, cols)
    ]

    # Create GeoDataFrame
    gdf_raster = gpd.GeoDataFrame({"data": values}, geometry=points, crs=crs)
    return gdf_raster


def snowCover(path_nc_wgs84, filename_nc):
    # Open xarray:
    ds_latlon = xr.open_dataset(path_nc_wgs84 + filename_nc)

    # Smoothing
    ds_latlon_g = GaussianFilter(ds_latlon)

    # Convert to GeoPandas
    gdf_glacier, lon, lat = toGeoPandas(ds_latlon_g)

    # Make classification map of snow/ice:
    # Replace values: below 0 with 3, above 0 with 1
    gdf_class = gdf_glacier.copy()
    tol = 0.1
    gdf_class.loc[gdf_glacier['data'] <= 0 + tol, 'data'] = 3
    gdf_class.loc[gdf_glacier['data'] > 0 + tol, 'data'] = 1

    snow_cover_glacier, ice_cover_glacier = IceSnowCover(gdf_class)

    return gdf_glacier, gdf_class, snow_cover_glacier, ice_cover_glacier


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


def IceSnowCover(gdf_class):
    # Calculate percentage of snow cover (class 1)
    snow_cover_glacier = gdf_class.data[gdf_class.data ==
                                        1].count() / gdf_class.data.count()
    ice_cover_glacier = gdf_class.data[gdf_class.data ==
                                       3].count() / gdf_class.data.count()
    return snow_cover_glacier, ice_cover_glacier



def replace_clouds_with_nearest_neighbor(gdf, class_column='data', cloud_class=5):
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
    cloud_coords = np.array(list(cloud_pixels.geometry.apply(lambda geom: (geom.x, geom.y))))
    non_cloud_coords = np.array(list(non_cloud_pixels.geometry.apply(lambda geom: (geom.x, geom.y))))

    # Perform nearest-neighbor search
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(non_cloud_coords)
    distances, indices = nbrs.kneighbors(cloud_coords)

    # Map nearest neighbor's class to cloud pixels
    nearest_classes = non_cloud_pixels.iloc[indices.flatten()][class_column].values
    gdf.loc[cloud_pixels.index, class_column] = nearest_classes

    return gdf




