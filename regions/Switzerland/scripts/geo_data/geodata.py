import os
from os.path import isfile, join
from collections import defaultdict
from datetime import datetime
import tempfile
import shutil
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rasterio
import pyproj
from scipy.ndimage import gaussian_filter
from dateutil.relativedelta import relativedelta
from tqdm.notebook import tqdm
from rasterio.transform import from_bounds

import massbalancemachine as mbm

from regions.Switzerland.scripts.config_CH import *
from regions.Switzerland.scripts.utils import *
from regions.Switzerland.scripts.geo_data.svf import merge_svf_into_ds


# --------------------------- Pure geospatial/raster utilities --------------------------- #
def LV03_to_WGS84(df):
    """
    Convert Swiss LV03 (CH1903) point coordinates to WGS84 latitude/longitude.

    This function expects LV03 coordinates in columns ``x_pos``, ``y_pos`` and
    an elevation column ``z_pos``. It appends ``lat``, ``lon`` and ``height``
    and drops the original coordinate columns.

    Notes
    -----
    This function **mutates the input DataFrame in-place** (adds/drops columns).

    Parameters
    ----------
    df : pandas.DataFrame
        Input table with columns:
        - ``x_pos`` : float, LV03 easting
        - ``y_pos`` : float, LV03 northing
        - ``z_pos`` : float, elevation (meters)

    Returns
    -------
    pandas.DataFrame
        Same object as input, with columns:
        - ``lat`` : float (degrees, WGS84)
        - ``lon`` : float (degrees, WGS84)
        - ``height`` : float (meters)
        and without ``x_pos``, ``y_pos``, ``z_pos``.

    Raises
    ------
    KeyError
        If required columns are missing.
    """
    converter = GPSConverter()
    lat, lon, height = converter.LV03_to_WGS84(df["x_pos"], df["y_pos"], df["z_pos"])
    df["lat"] = lat
    df["lon"] = lon
    df["height"] = height
    df.drop(["x_pos", "y_pos", "z_pos"], axis=1, inplace=True)
    return df


def LV03_to_WGS84(df):
    """
    Convert coordinates from the Swiss LV03 system (CH1903) to WGS84 latitude/longitude.

    This function transforms planar Swiss grid coordinates (x/y/z in LV03)
    into geographic coordinates in the WGS84 reference system using the
    `GPSConverter` utility.

    The input DataFrame is expected to contain LV03 coordinates in columns
    named `x_pos`, `y_pos`, and `z_pos`. After conversion, these columns
    are replaced by new columns:

        - `lat`  : latitude in degrees (WGS84)
        - `lon`  : longitude in degrees (WGS84)
        - `height` : height in meters (WGS84 ellipsoidal height)

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing Swiss LV03 coordinates with the following required columns:
        - `x_pos` : float, easting in LV03 coordinates
        - `y_pos` : float, northing in LV03 coordinates
        - `z_pos` : float, elevation in meters

    Returns
    -------
    pandas.DataFrame
        The same DataFrame with LV03 coordinate columns removed and replaced by:
        - `lat`
        - `lon`
        - `height`

    Raises
    ------
    KeyError
        If any of the required columns (`x_pos`, `y_pos`, `z_pos`) are missing.

    Notes
    -----
    - The conversion relies on the `GPSConverter` class, which implements
      the official transformation between Swiss CH1903/LV03 and WGS84.
    - The transformation is applied element-wise to all rows of the DataFrame.
    - The input DataFrame is modified in place.
    """
    converter = GPSConverter()
    lat, lon, height = converter.LV03_to_WGS84(df["x_pos"], df["y_pos"], df["z_pos"])
    df["lat"] = lat
    df["lon"] = lon
    df["height"] = height
    df.drop(["x_pos", "y_pos", "z_pos"], axis=1, inplace=True)
    return df


def LV95_to_WGS84(df):
    """
    Convert Swiss LV95 (CH1903+) point coordinates to WGS84 latitude/longitude.

    Uses a pyproj transformation from EPSG:2056 (LV95) to EPSG:4326 (WGS84).
    The function appends ``lat`` and ``lon`` and drops ``x_pos``, ``y_pos``,
    ``z_pos``.

    Notes
    -----
    - This function **mutates the input DataFrame in-place** (adds/drops columns).
    - The input ``z_pos`` is not used in the transformation but is removed for
      convenience to mirror other conversion helpers.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table with columns:
        - ``x_pos`` : float, LV95 easting (EPSG:2056)
        - ``y_pos`` : float, LV95 northing (EPSG:2056)
        - ``z_pos`` : float, elevation (optional; will be dropped)

    Returns
    -------
    pandas.DataFrame
        Same object as input, with columns:
        - ``lat`` : float (degrees, WGS84)
        - ``lon`` : float (degrees, WGS84)
        and without ``x_pos``, ``y_pos``, ``z_pos``.

    Raises
    ------
    KeyError
        If required columns are missing.
    """
    transformer = pyproj.Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)

    # Transform to Latitude and Longitude (WGS84)
    lon, latitude = transformer.transform(df.x_pos, df.y_pos)

    df["lat"] = latitude
    df["lon"] = lon
    df.drop(["x_pos", "y_pos", "z_pos"], axis=1, inplace=True)
    return df


def convert_to_xarray_geodata(grid_data, metadata):
    """
    Convert a loaded .grid raster (values + metadata) into an xarray DataArray in LV95/LV03 coordinates.

    The function builds ``x`` and ``y`` coordinates from the grid metadata and returns a 2D
    ``xarray.DataArray`` named ``"grid_data"`` with dims ``("y", "x")``.
    The input data are flipped vertically (``np.flip(..., axis=0)``) so that the resulting
    ``y`` coordinate increases in the expected direction.

    Parameters
    ----------
    grid_data : numpy.ndarray
        2D array of raster values read from a .grid file (shape nrows x ncols).
    metadata : dict
        Grid metadata with keys:
        - ``ncols`` (int)
        - ``nrows`` (int)
        - ``xllcorner`` (float)
        - ``yllcorner`` (float)
        - ``cellsize`` (float)

    Returns
    -------
    xarray.DataArray
        2D DataArray with dims ``("y", "x")`` and coordinates in the same projected CRS
        as the input grid metadata (commonly LV95 meters).

    Raises
    ------
    KeyError
        If required metadata keys are missing.
    ValueError
        If ``grid_data`` shape is inconsistent with ``nrows``/``ncols``.
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
    """
    Reassign LV95 (EPSG:2056) x/y coordinates of a 2D DataArray to WGS84 lon/lat coordinates.

    The input is assumed to be a 2D grid with dims ``("y", "x")`` in LV95 meters.
    The function computes lon/lat coordinates by transforming every grid point from
    EPSG:2056 to EPSG:4326, assigns 1D ``lon`` and ``lat`` coordinates, and swaps
    dimensions from ``("y", "x")`` to ``("lat", "lon")``.

    Parameters
    ----------
    data_array : xarray.DataArray
        2D raster with dims ``("y", "x")`` and coordinates:
        - ``x`` : LV95 easting (meters)
        - ``y`` : LV95 northing (meters)

    Returns
    -------
    xarray.DataArray
        Same values as input but with WGS84 coordinates and dims swapped to
        ``("lat", "lon")`` (coordinates are 1D and derived from the transformed grid).

    Notes
    -----
    - This implementation assumes the transformed grid is separable into 1D lon and 1D lat
      vectors (it uses the first row/column after transformation). This is a reasonable
      approximation for small domains but is not a fully general curvilinear-grid treatment.
    - The function currently assumes EPSG:2056 (LV95). For LV03, use a dedicated transform.

    Raises
    ------
    AttributeError
        If expected coordinates ``x``/``y`` are missing.
    ValueError
        If the input is not 2D or does not have dims ``("y", "x")``.
    """

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
    df = LV95_to_WGS84(df)

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
    """
    Transform the spatial coordinates of an xarray DataArray from Swiss LV03 (CH1903)
    projected coordinates to WGS84 geographic coordinates (longitude/latitude).

    This function assumes that the input DataArray is a 2-dimensional raster with
    dimensions ("y", "x") representing Swiss grid coordinates in meters (LV03).
    It performs the following steps:

    1. Flattens the 2D grid values and corresponding LV03 x/y coordinates.
    2. Converts all grid coordinates from LV03 to WGS84 using `LV03_to_WGS84`.
    3. Reshapes the transformed longitude and latitude arrays back to the original
       grid shape.
    4. Assigns new 1D longitude and latitude coordinates to the DataArray.
    5. Swaps dimensions so that the resulting DataArray uses ("lat", "lon")
       instead of ("y", "x").

    Parameters
    ----------
    data_array : xarray.DataArray
        Input DataArray with dimensions ("y", "x") and LV03 coordinates stored
        as `data_array.coords["x"]` and `data_array.coords["y"]`.

    Returns
    -------
    xarray.DataArray
        DataArray with spatial dimensions transformed to WGS84 coordinates.
        The returned array has dimensions ("lat", "lon") instead of ("y", "x"),
        and includes new coordinates:
        - `lon` : 1D array of longitudes in degrees (EPSG:4326)
        - `lat` : 1D array of latitudes in degrees (EPSG:4326)

    Raises
    ------
    AttributeError
        If the input DataArray does not contain `x` and `y` coordinates.
    ValueError
        If the coordinate transformation fails or the array shapes are inconsistent.

    Notes
    -----
    - The transformation assumes that the input coordinates are in LV03
      (EPSG:21781). If the data are in LV95 instead, a different converter
      should be used.
    - The conversion is performed by flattening the grid to a table of points,
      transforming them using `LV03_to_WGS84`, and then reconstructing the grid.
    - The function does not modify data values—only the coordinate system.
    """
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
    df = LV03_to_WGS84(df)

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


def coarsen_DS(
    ds, target_res_m=50, vars=["masked_slope", "masked_aspect", "masked_elev"]
):
    """
    Coarsen a glacier-topography dataset to a target spatial resolution.

    The dataset is assumed to be on a regular lat/lon grid (degrees). If the
    current resolution is finer than ``target_res_m`` (in meters), the function
    aggregates:
    - continuous variables in ``vars`` using the mean
    - the binary/indicator ``glacier_mask`` using max (preserving any glacier
      coverage within the coarse cell)

    Coarsening is performed along dimensions ``lon`` and ``lat`` with
    ``boundary="trim"``.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing at least ``glacier_mask`` and the variables listed
        in ``vars``. Must have 1D coordinates ``lon`` and ``lat`` in degrees.
    target_res_m : float, optional
        Target resolution in meters (default 50).
    vars : list of str, optional
        Names of non-binary variables to coarsen with mean.

    Returns
    -------
    xarray.Dataset
        Coarsened dataset if the input resolution is finer than ``target_res_m``;
        otherwise the original dataset is returned unchanged.

    Raises
    ------
    KeyError
        If required variables (``glacier_mask`` or any in ``vars``) are missing.
    """
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


def get_res_from_degrees(ds):
    """
    Estimate dataset grid resolution (meters) from 1D lat/lon coordinates in degrees.

    Uses a spherical approximation:
    - 1 degree latitude ≈ 111,320 m
    - 1 degree longitude ≈ 111,320 * cos(latitude) m

    The resolution is computed from the first two coordinate values in ``ds.lon`` and
    ``ds.lat`` and rounded to 3 decimals.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Object with 1D coordinates ``lat`` and ``lon`` in degrees. Must have at least
        two points along each coordinate.

    Returns
    -------
    tuple[float, float]
        (dx_m, dy_m) estimated grid spacing in meters for longitude and latitude directions.

    Raises
    ------
    AttributeError
        If ``lat``/``lon`` coordinates are missing.
    IndexError
        If ``lat`` or ``lon`` has fewer than 2 elements.
    """

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


def xr_SGI_masked_topo(gdf_shapefiles, sgi_id, cfg):
    """
    Load SGI topography rasters (aspect/slope/DEM), convert to WGS84, and mask by glacier outline.

    This function:
    1) locates SGI .grid files for the given ``sgi_id`` (aspect, slope, dem_HR)
    2) reads them via ``load_grid_file``
    3) converts them to xarray in LV95 coordinates via ``convert_to_xarray_geodata``
    4) transforms coordinates to WGS84 via ``transform_xarray_coords_lv95_to_wgs84``
    5) rasterizes the corresponding glacier outline and masks the topo fields
       using ``extract_topo_over_outline``

    The returned dataset contains masked aspect/slope/elevation and a glacier mask.

    Parameters
    ----------
    gdf_shapefiles : geopandas.GeoDataFrame
        Glacier outlines with a column ``"sgi-id"`` identifying glaciers. CRS must
        be defined (will be reprojected to match raster as needed).
    sgi_id : str
        SGI glacier identifier used to select the topo files and the outline.
    cfg : object
        Configuration object with attribute ``dataPath`` used as base path. The
        function also expects ``path_SGI_topo`` to be available (imported from config).

    Returns
    -------
    xarray.Dataset
        Dataset on a WGS84 lat/lon grid with variables:
        - ``masked_aspect`` : xarray.DataArray
        - ``masked_slope``  : xarray.DataArray
        - ``masked_elev``   : xarray.DataArray (NaN outside glacier; negative values set to NaN)
        - ``glacier_mask``  : xarray.DataArray (0/1)

    Raises
    ------
    IndexError
        If no topo file matching ``sgi_id`` is found in one of the SGI topo folders.
    KeyError
        If ``"sgi-id"`` is missing from ``gdf_shapefiles``.
    ValueError
        If the outline GeoDataFrame has no CRS (raised in ``extract_topo_over_outline``).
    """
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


def get_rgi_sgi_ids(cfg, glacier_name):
    """
    Look up SGI and RGI identifiers (and a 2016 RGI shapefile id) for a glacier short name.

    Reads the glacier-id mapping table from ``cfg.dataPath + path_glacier_ids`` and
    returns the SGI id, the RGI v6 id, and the RGI v6 2016 shapefile identifier.

    Special handling is included for ``glacier_name == "clariden"`` to map to
    the "claridenU" entry.

    Parameters
    ----------
    cfg : object
        Configuration object with attribute ``dataPath``.
    glacier_name : str
        Glacier short name used as index key in the mapping table (after sorting and indexing
        by ``short_name``).

    Returns
    -------
    tuple[str, str, str]
        (sgi_id, rgi_id, rgi_shp). If the glacier is not found, empty strings are returned.

    Notes
    -----
    This function returns empty strings instead of raising if the glacier name is not found.

    Raises
    ------
    FileNotFoundError
        If the mapping CSV cannot be read.
    KeyError
        If expected columns are missing from the CSV (e.g., ``short_name``, ``sgi-id``,
        ``rgi_id.v6``, ``rgi_id_v6_2016_shp``).
    """
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


def load_grid_file(filepath):
    """
    Read an ASCII .grid raster file and return its metadata and data array.

    The function parses the standard ESRI-style grid header (ncols, nrows,
    xllcorner, yllcorner, cellsize, nodata_value) followed by a block of
    numeric values.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the .grid file to be read.

    Returns
    -------
    metadata : dict
        Dictionary containing grid header information with keys:
        - ``ncols`` : int
        - ``nrows`` : int
        - ``xllcorner`` : float
        - ``yllcorner`` : float
        - ``cellsize`` : float
        - ``nodata_value`` : float or int

    grid_data : numpy.ndarray
        2D array of shape (nrows, ncols) containing the raster values.
        Values equal to ``nodata_value`` are preserved as-is.

    Notes
    -----
    - The returned array is oriented exactly as stored in the file.
      Many workflows (e.g., ``convert_to_xarray_geodata``) subsequently
      flip the array vertically so that the y-coordinate increases upward.
    - This function performs no CRS transformation; it only reads raw grid data.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file header is malformed or the number of data values does
        not match the declared grid dimensions.
    """
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


def create_sgi_topo_masks(
    cfg,
    iterator,
    type="glacier_name",
    path_save=None,
    path_xr_svf=None,
    empty_output=True,
):
    """
    Create per-glacier SGI masked topography datasets (optionally merged with SVF) and save as Zarr.

    For each item in `iterator` (either glacier names or SGI ids), this function:
    - resolves SGI id (if needed)
    - loads SGI topo rasters and masks them to the 2016 glacier outline
    - coarsens to a target resolution (via `coarsen_DS`)
    - optionally merges SVF variables from precomputed NetCDFs (`merge_svf_into_ds`)
    - writes each glacier dataset to a Zarr store using an atomic save strategy

    Parameters
    ----------
    cfg : object
        Configuration object with attribute `dataPath`.
    iterator : iterable
        Sequence of glacier identifiers (names or SGI ids).
    type : {"glacier_name", "sgi_id"}, optional
        Interpretation of items in `iterator`.
    path_save : str or None, optional
        Output folder for Zarr datasets.
    path_xr_svf : str or None, optional
        Folder containing `<sgi_id>_svf_latlon.nc` files to merge.
    empty_output : bool, optional
        If True, empties `path_save` before writing.

    Returns
    -------
    None

    Side Effects
    ------------
    Writes Zarr datasets to disk; may delete existing output content if `empty_output=True`.
    """
    if path_save is None:
        path_save = os.path.join(cfg.dataPath, path_SGI_topo, "xr_masked_grids/")
    os.makedirs(path_save, exist_ok=True)
    if empty_output:
        emptyfolder(path_save)  # keep old behavior unless you set empty_output=False

    glacier_outline_sgi = gpd.read_file(
        os.path.join(
            cfg.dataPath, path_SGI_topo, "inventory_sgi2016_r2020/SGI_2016_glaciers.shp"
        )
    )

    for item in tqdm(iterator, desc="Processing glaciers"):
        # resolve SGI id
        if type == "glacier_name":
            sgi_id, rgi_id, rgi_shp = get_rgi_sgi_ids(cfg, item)
            if not sgi_id:
                print(f"Warning: Missing SGI ID or shapefile for {item}. Skipping...")
                continue
        elif type == "sgi_id":
            sgi_id = item
        else:
            print(f"Unknown type '{type}', skipping {item}")
            continue

        # build + coarsen
        try:
            ds = xr_SGI_masked_topo(glacier_outline_sgi, sgi_id, cfg)
            if ds is None:
                print(f"Warning: Failed to load dataset for {item}. Skipping...")
                continue
            ds_resampled = coarsen_DS(ds)
            if ds_resampled is None:
                print(f"Warning: Resampling failed for {item}. Skipping...")
                continue
        except Exception as e:
            print(f"Error preparing dataset for {item}: {e}")
            continue

        # --- NEW: merge SVF (optional) ---
        if path_xr_svf is not None:
            try:
                ds_resampled = merge_svf_into_ds(ds_resampled, sgi_id, path_xr_svf)
            except Exception as e:
                print(f"SVF merge failed for {item} ({sgi_id}): {e}")

        # atomic save
        final_path = os.path.join(path_save, f"{item}.zarr")
        tmp_dir = tempfile.mkdtemp(prefix=f".tmp_{item}_", dir=path_save)
        try:
            ds_resampled.to_zarr(tmp_dir, mode="w")
            if os.path.exists(final_path):
                shutil.rmtree(final_path)
            os.replace(tmp_dir, final_path)
            print(f"Saved {item} to {final_path}")
        except Exception as e:
            try:
                if os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir)
            except Exception:
                pass
            print(f"Error saving dataset for {item}: {e}")


def xr_GLAMOS_masked_topo(cfg, sgi_id, ds_gl):
    """
    Build a glacier-masked topographic dataset (aspect, slope, elevation) for a GLAMOS glacier.

    This function loads SGI topographic raster grids (aspect and slope) corresponding to
    a given SGI glacier identifier, converts them from LV95 projected coordinates to
    WGS84 geographic coordinates, resamples them to the resolution of an existing
    glacier mask dataset, and applies that mask.

    The resulting dataset contains glacier-masked versions of aspect, slope, and
    elevation on a common lat/lon grid.

    Processing steps:
    -----------------
    1. Locate and load SGI aspect and slope `.grid` files for the provided `sgi_id`.
    2. Convert the raw grid data to xarray DataArrays using metadata information.
    3. Transform LV95 (EPSG:2056) coordinates to WGS84 lat/lon.
    4. Resample aspect and slope to match the resolution of `ds_gl["glacier_mask"]`.
    5. Apply the glacier mask to aspect, slope, and DEM values.
    6. Remove negative DEM values (treated as erroneous).
    7. Return a unified masked topography dataset.

    Parameters
    ----------
    cfg : object
        Configuration object containing at least:
        - `dataPath`: base directory for SGI topography data.
        - `path_SGI_topo`: relative path to the SGI topography folder.
    sgi_id : str
        SGI glacier identifier used to select the correct SGI grid files.
    ds_gl : xarray.Dataset
        Existing glacier dataset that must contain:
        - `glacier_mask` : 2D mask array (1 = glacier, 0 = non-glacier)
        - `dem`          : elevation field on the same grid as the mask

    Returns
    -------
    xarray.Dataset
        Dataset on the same grid as `ds_gl` containing:
        - `masked_aspect` : aspect values masked to glacier area
        - `masked_slope`  : slope values masked to glacier area
        - `masked_elev`   : DEM values masked to glacier area
        - `glacier_mask`  : original glacier mask

    Raises
    ------
    FileNotFoundError
        If SGI aspect or slope grid files for the given `sgi_id` cannot be found.
    KeyError
        If required variables are missing from `ds_gl`.
    IndexError
        If multiple or zero SGI files match the given `sgi_id`.

    Notes
    -----
    - Input SGI rasters are assumed to be in LV95 coordinates and are transformed
      to WGS84 using `transform_xarray_coords_lv95_to_wgs84`.
    - Resampling to the glacier mask grid is performed using nearest-neighbor
      interpolation to preserve categorical characteristics.
    - Elevation values below 0 m are set to NaN as a basic quality filter.
    """
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
