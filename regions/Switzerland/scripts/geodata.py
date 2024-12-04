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
from shapely.geometry import Point, box
import contextily as cx
import geodatasets
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
import glob
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from datetime import datetime, timedelta


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
    if not (
        bounding_box[0] >= raster_bounds[0] and  # minx of glacier >= minx of raster
        bounding_box[1] >= raster_bounds[1] and  # miny of glacier >= miny of raster
        bounding_box[2] <= raster_bounds[2] and  # maxx of glacier <= maxx of raster
        bounding_box[3] <= raster_bounds[3]     # maxy of glacier <= maxy of raster
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


def plotClasses(gdf_glacier, gdf_class, gdf_class_corr, gdf_raster_res, axs, gl_date, file_date):
    # Define the colors for categories (ensure that your categories match the color list)
    colors_cat = [
        '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
        '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'
    ]

    # Manually map categories to colors (assuming categories 0-5 for example)
    unique_cat = gdf_raster_res.data.unique()
    # remove nan
    unique_cat = unique_cat[~np.isnan(unique_cat)]
    unique_cat.sort()
    map = dict(
        zip(unique_cat,
            colors_cat[:6]))  # Adjust according to the number of categories

    # Set up the basemap provider
    API_KEY = "000378bd-b0f0-46e2-a46d-f2165b0c6c02"
    provider = cx.providers.Stadia.StamenTerrain(api_key=API_KEY)
    provider["url"] = provider["url"] + f"?api_key={API_KEY}"

    # Plot the first figure (Mass balance)
    vmin, vmax = gdf_glacier.data.min(), gdf_glacier.data.max()
    if vmin < 0 and vmax > 0:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    if vmin < 0 and vmax <= 0:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=0.5)
    else:
        norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=vmin, vmax=vmax)
    gdf_clean = gdf_glacier.dropna(subset=["data"])
    gdf_clean.plot(
        column="data",  # Column to visualize
        cmap="RdBu",  # Color map suitable for glacier data
        norm=norm,
        legend=True,  # Display a legend
        ax=axs[0],
        markersize=5,  # Adjust size if points are too small or large
        missing_kwds={"color": "lightgrey"}  # Define color for NaN datas
    )
    #cx.add_basemap(axs[0], crs=gdf_glacier.crs, source=provider)
    axs[0].set_title(f"Mass balance: {gl_date}")

    # Plot the second figure (MBM classes)
    gdf_clean = gdf_class.dropna(subset=["data"])
    gdf_clean['color'] = gdf_clean['data'].map(map)
    # Plot with manually defined colormap
    gdf_clean.plot(
        column="data",  # Column to visualize
        legend=True,  # Display a legend
        markersize=5,  # Adjust size if points are too small or large
        missing_kwds={"color": "lightgrey"},  # Define color for NaN datas
        categorical=True,  # Ensure the plot uses categorical colors
        ax=axs[1],
        color=gdf_clean['color']  # Use the custom colormap
    )
    #cx.add_basemap(axs[1], crs=gdf_glacier.crs, source=provider)
    axs[1].set_title(f"MBM classes: {gl_date}")
    
    # Plot the third figure (MBM classes corrected)
    gdf_clean = gdf_class_corr.dropna(subset=["data"])
    gdf_clean['color'] = gdf_clean['data'].map(map)
    # Plot with manually defined colormap
    gdf_clean.plot(
        column="data",  # Column to visualize
        legend=True,  # Display a legend
        markersize=5,  # Adjust size if points are too small or large
        missing_kwds={"color": "lightgrey"},  # Define color for NaN datas
        categorical=True,  # Ensure the plot uses categorical colors
        ax=axs[2],
        color=gdf_clean['color']  # Use the custom colormap
    )
    #cx.add_basemap(axs[1], crs=gdf_glacier.crs, source=provider)
    axs[2].set_title(f"MBM classes corr.: {gl_date}")

    # Plot the fourth figure (Resampled Sentinel classes)
    gdf_clean = gdf_raster_res.dropna(subset=["data"])
    gdf_clean['color'] = gdf_clean['data'].map(map)
    # Plot with manually defined colormap
    gdf_clean.plot(
        column="data",  # Column to visualize
        legend=True,  # Display a legend
        markersize=5,  # Adjust size if points are too small or large
        missing_kwds={"color": "lightgrey"},  # Define color for NaN datas
        categorical=True,  # Ensure the plot uses categorical colors
        ax=axs[3],
        color=gdf_clean['color']  # Use the custom colormap
    )
    #cx.add_basemap(axs[2], crs=gdf_glacier.crs, source=provider)
    axs[3].set_title(f"Sentinel classes: {file_date.strftime('%Y-%m-%d')}")

    # Manually add custom legend for the third plot
    classes = [
        'snow', 'firn / old snow / bright ice', 'clean ice', 'debris', 'cloud'
    ]
    handles = [
        mpatches.Patch(color=color, label=classes[i])
        for i, color in enumerate(colors_cat[:len(map)])
    ]
    axs[3].legend(handles=handles,
                  title="Classes",
                  bbox_to_anchor=(1.05, 1),
                  loc='upper left')

    # Show the plot with consistent colors
    plt.tight_layout()
    plt.show()


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

    # Calculate percentage of snow cover (class 1)
    snow_cover_glacier = gdf_class.data[gdf_class.data ==
                                        1].count() / gdf_class.data.count()
    ice_cover_glacier = gdf_class.data[gdf_class.data ==
                                       3].count() / gdf_class.data.count()

    return gdf_glacier, gdf_class, snow_cover_glacier, ice_cover_glacier

def get_hydro_year_and_month(file_date):
    if file_date.day >= 15:
        # Move to the next month
        file_date += timedelta(days=(30 - file_date.day + 1))  # Approximate to next month
        
    # Step 2: Determine the closest month
    closest_month = file_date.strftime('%b').lower()  # Get the full name of the month
    
    # Step 3: Determine the hydrological year
    # Hydrological year runs from September to August
    if file_date.month >= 9:  # September, October, November, December
        hydro_year = file_date.year + 1 # Assign to the next year
    else:  # January to August
        hydro_year = file_date.year  # Assign to the current year

    
    return closest_month, hydro_year