# --- Standard library ---
import os
import glob
import shutil
import warnings

# --- Scientific / numeric ---
import numpy as np
import xarray as xr

# --- Geospatial ---
import rasterio
import rioxarray
from rasterio.transform import Affine
import rvt.default
import rvt.vis

from pyproj import CRS

# NOTE: NEEDS TO BE RUN IN SEPARATE RTV ENVIRONMENT WITH RTV INSTALLED


# ----------------------- SVF processing of DEMs -----------------------
# def process_dem_to_netcdf(
#     dem_path: str,
#     out_path: str,
#     svf_n_dir=16,
#     svf_r_max=10,
#     svf_noise=0,
#     asvf_level=1,
#     asvf_dir=315,
# ) -> None:
#     """
#     Compute sky-view factor (SVF), anisotropic SVF (ASVF), and openness from a DEM
#     and write the results to a compressed NetCDF file.

#     The function reads a raster DEM, extracts its spatial metadata using rasterio,
#     computes terrain visibility metrics using the Relief Visualization Toolbox (RVT),
#     and exports the resulting grids as an xarray Dataset in NetCDF format with
#     correct georeferencing.

#     Parameters
#     ----------
#     dem_path : str
#         Path to the input DEM raster file (e.g., GeoTIFF) in a projected metric CRS.
#     out_path : str
#         Path where the output NetCDF file will be written.
#     svf_n_dir : int, optional
#         Number of directions used for SVF computation (default: 16).
#     svf_r_max : int or float, optional
#         Maximum search radius in pixels for SVF computation (default: 10).
#     svf_noise : float, optional
#         Noise parameter for RVT SVF computation (default: 0).
#     asvf_level : int, optional
#         Level parameter for anisotropic SVF computation (default: 1).
#     asvf_dir : int, optional
#         Direction parameter (in degrees) for anisotropic SVF (default: 315).

#     Returns
#     -------
#     None
#         The function writes output directly to disk and returns nothing.

#     Raises
#     ------
#     ValueError
#         If the DEM appears to be in geographic (degree) coordinates instead of a
#         projected metric CRS.

#     Notes
#     -----
#     - The input DEM must be in a metric coordinate reference system (e.g., LV95).
#       SVF calculations in degrees are not meaningful.
#     - Output NetCDF variables include:
#         * svf  – sky-view factor
#         * asvf – anisotropic sky-view factor
#         * opns – positive openness
#     - Compression is applied when possible using h5netcdf or netcdf4 engines.
#     """
#     # --- open with rasterio for authoritative geoinfo ---
#     with rasterio.open(dem_path) as src:
#         transform: Affine = src.transform
#         crs = src.crs
#         width, height = src.width, src.height
#         # pixel sizes (a > 0, e < 0 for north-up rasters)
#         pix_w = float(transform.a)
#         pix_h = abs(float(transform.e))
#         # center-of-pixel coordinates from transform
#         x = transform.c + (np.arange(width) + 0.5) * pix_w
#         y = transform.f + (np.arange(height) + 0.5) * transform.e  # note e is negative

#         # read DEM as array (band 1)
#         dem = src.read(1)
#         no_data = src.nodata

#     if max(pix_w, pix_h) < 0.0001:
#         raise ValueError(
#             "DEM appears to be in degrees; compute SVF on a metric CRS first."
#         )

#     # --- compute SVF / ASVF / OPNS (RVT expects one horizontal resolution) ---
#     out = rvt.vis.sky_view_factor(
#         dem=dem,
#         resolution=pix_w,  # meters
#         compute_svf=True,
#         compute_asvf=True,
#         compute_opns=True,
#         svf_n_dir=svf_n_dir,
#         svf_r_max=svf_r_max,
#         svf_noise=svf_noise,
#         asvf_level=asvf_level,
#         asvf_dir=asvf_dir,
#         no_data=no_data,
#     )

#     svf_arr = out["svf"]
#     asvf_arr = out["asvf"]
#     opns_arr = out["opns"]

#     # --- build Dataset with correct x/y coords ---
#     da_svf = xr.DataArray(
#         svf_arr,
#         dims=("y", "x"),
#         coords={"y": y, "x": x},
#         name="svf",
#         attrs={"description": "Sky-View Factor", "units": "unitless"},
#     )
#     da_asvf = xr.DataArray(
#         asvf_arr,
#         dims=("y", "x"),
#         coords={"y": y, "x": x},
#         name="asvf",
#         attrs={
#             "description": "Anisotropic SVF",
#             "units": "unitless",
#             "asvf_level": asvf_level,
#             "asvf_dir": asvf_dir,
#         },
#     )
#     da_opns = xr.DataArray(
#         opns_arr,
#         dims=("y", "x"),
#         coords={"y": y, "x": x},
#         name="opns",
#         attrs={"description": "Positive Openness", "units": "radians"},
#     )

#     ds_out = xr.Dataset(
#         data_vars={"svf": da_svf, "asvf": da_asvf, "opns": da_opns},
#         coords={"x": x, "y": y},
#         attrs={
#             "source_dem": os.path.basename(dem_path),
#             "no_data": no_data if no_data is not None else np.nan,
#             "crs": str(crs) if crs is not None else "EPSG:2056",
#             "rvt_svf_n_dir": svf_n_dir,
#             "rvt_svf_r_max": svf_r_max,
#             "rvt_svf_noise": svf_noise,
#         },
#     )

#     # compressed NetCDF (no fixed chunksizes to avoid tiny rasters failing)
#     encoding = {
#         v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds_out.data_vars
#     }
#     try:
#         ds_out.to_netcdf(out_path, engine="h5netcdf", encoding=encoding)
#     except Exception:
#         try:
#             ds_out.to_netcdf(out_path, engine="netcdf4", encoding=encoding)
#         except Exception:
#             ds_out.to_netcdf(out_path)  # scipy fallback (no compression)


def process_dem_to_netcdf(
    dem_path: str,
    out_path: str,
    svf_n_dir=16,
    svf_r_max=10,
    svf_noise=0,
    asvf_level=1,
    asvf_dir=315,
) -> None:
    """
    Compute SVF/ASVF/openness from a DEM GeoTIFF and export to NetCDF.

    CRS handling
    ------------
    - The CRS is read from the input GeoTIFF and stored in output metadata.
    - The DEM must be in a projected metric CRS (units in meters). If the DEM CRS
      is geographic (degrees) or missing, the function raises an error.

    Notes
    -----
    - RVT expects a single horizontal resolution; this uses the x pixel size.
      If pixels are not square, you may want to resample beforehand.
    """

    # --- open with rasterio for authoritative geoinfo ---
    with rasterio.open(dem_path) as src:
        transform: Affine = src.transform
        crs = src.crs
        width, height = src.width, src.height

        pix_w = float(transform.a)
        pix_h = abs(float(transform.e))

        # center-of-pixel coords
        x = transform.c + (np.arange(width) + 0.5) * pix_w
        y = (
            transform.f + (np.arange(height) + 0.5) * transform.e
        )  # e is negative for north-up

        dem = src.read(1)
        no_data = src.nodata

    # --- CRS validation (robust) ---
    if crs is None:
        raise ValueError(
            f"DEM has no CRS metadata: {dem_path}. "
            "Please write CRS to the GeoTIFF before computing SVF."
        )

    crs_obj = CRS.from_user_input(crs)

    if crs_obj.is_geographic:
        raise ValueError(
            f"DEM CRS is geographic (degrees): {crs_obj.to_string()}. "
            "SVF must be computed on a projected metric CRS."
        )

    # check units if available (typically 'metre' for projected CRS)
    axis_info = crs_obj.axis_info
    if axis_info and any(
        ("degree" in (ax.unit_name or "").lower()) for ax in axis_info
    ):
        raise ValueError(f"DEM CRS axis units look non-metric: {crs_obj.to_string()}.")

    # --- basic resolution sanity ---
    if pix_w <= 0 or pix_h <= 0:
        raise ValueError(f"Invalid pixel size in DEM: pix_w={pix_w}, pix_h={pix_h}")

    if not np.isclose(pix_w, pix_h, rtol=0.05):
        # not fatal, but worth knowing (RVT uses one resolution)
        print(
            f"[warn] Non-square pixels: pix_w={pix_w:.3f}, pix_h={pix_h:.3f}. Using pix_w for RVT."
        )

    # --- compute SVF / ASVF / OPNS (RVT expects one horizontal resolution) ---
    out = rvt.vis.sky_view_factor(
        dem=dem,
        resolution=pix_w,  # meters
        compute_svf=True,
        compute_asvf=True,
        compute_opns=True,
        svf_n_dir=svf_n_dir,
        svf_r_max=svf_r_max,
        svf_noise=svf_noise,
        asvf_level=asvf_level,
        asvf_dir=asvf_dir,
        no_data=no_data,
    )

    svf_arr = out["svf"]
    asvf_arr = out["asvf"]
    opns_arr = out["opns"]

    # --- build Dataset with correct x/y coords ---
    ds_out = xr.Dataset(
        data_vars={
            "svf": (("y", "x"), svf_arr),
            "asvf": (("y", "x"), asvf_arr),
            "opns": (("y", "x"), opns_arr),
        },
        coords={"x": x, "y": y},
        attrs={
            "source_dem": os.path.basename(dem_path),
            "no_data": no_data if no_data is not None else np.nan,
            "crs": crs_obj.to_string(),
            "crs_wkt": crs_obj.to_wkt(),
            "pixel_width": pix_w,
            "pixel_height": pix_h,
            "rvt_svf_n_dir": svf_n_dir,
            "rvt_svf_r_max": svf_r_max,
            "rvt_svf_noise": svf_noise,
            "asvf_level": asvf_level,
            "asvf_dir": asvf_dir,
        },
    )

    # compressed NetCDF
    encoding = {
        v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds_out.data_vars
    }
    try:
        ds_out.to_netcdf(out_path, engine="h5netcdf", encoding=encoding)
    except Exception:
        try:
            ds_out.to_netcdf(out_path, engine="netcdf4", encoding=encoding)
        except Exception:
            ds_out.to_netcdf(out_path)


def _affine_from_coords(x, y):
    """
    Construct an affine geotransform from 1D coordinate vectors.

    This helper function derives an Affine transform assuming that the provided
    x and y arrays represent center-of-pixel coordinates on a regular grid.
    The resulting transform follows the north-up raster convention.

    Parameters
    ----------
    x : xarray.DataArray
        1D array of x-coordinates (increasing to the right).
    y : xarray.DataArray
        1D array of y-coordinates (typically decreasing downward).

    Returns
    -------
    rasterio.transform.Affine
        Affine transformation mapping pixel indices to spatial coordinates.

    Notes
    -----
    - Assumes uniform spacing in both x and y directions.
    - Designed for grids following standard GIS raster orientation.
    """

    resx = float(np.median(np.diff(x.values)))
    resy = float(np.median(np.diff(y.values)))
    pix_h = -abs(resy)  # north-up convention
    x0 = float(x.values[0]) - resx / 2.0
    y0 = float(y.values[0]) + abs(resy) / 2.0
    return Affine(resx, 0.0, x0, 0.0, pix_h, y0)


# def reproject_file_to_latlon(
#     nc_in: str, nc_out: str, data_vars, src_crs, dst_crs
# ) -> None:
#     """
#     Reproject selected variables from a projected NetCDF grid to latitude/longitude.

#     This function opens a NetCDF file containing gridded data in a projected
#     coordinate system (e.g., LV95), reprojects specified variables to a target CRS
#     (typically WGS84), and writes the result to a new NetCDF file.

#     Reprojection is performed using rioxarray with rasterio under the hood,
#     ensuring correct spatial alignment and coordinate transformation.

#     Parameters
#     ----------
#     nc_in : str
#         Path to the input NetCDF file containing x/y coordinates.
#     nc_out : str
#         Path to the output NetCDF file to be written.
#     data_vars : list of str
#         Names of data variables in the input file that should be reprojected.
#     src_crs : str or rasterio.crs.CRS
#         Source coordinate reference system (e.g., "EPSG:2056").
#     dst_crs : str or rasterio.crs.CRS
#         Target coordinate reference system (e.g., "EPSG:4326").

#     Returns
#     -------
#     None
#         The function writes a new NetCDF file and returns nothing.

#     Raises
#     ------
#     ValueError
#         If the input dataset does not contain x/y coordinates or if none of the
#         requested data variables are present.

#     Notes
#     -----
#     - The function preserves variable attributes and applies float32 encoding.
#     - Output coordinates are renamed to "lon" and "lat".
#     - Compression is applied when supported by the available NetCDF engine.
#     - All variables are aligned to the grid of the first valid variable to ensure
#       consistent spatial resolution.
#     - Designed primarily for DEM-derived products such as SVF, ASVF, and openness.
#     """
#     # Open/close cleanly so we don't leave files locked
#     with xr.open_dataset(nc_in) as ds:
#         # Check coords
#         if ("x" not in ds.coords) or ("y" not in ds.coords):
#             raise ValueError(f"{os.path.basename(nc_in)} is missing x/y coords.")

#         x = ds["x"]
#         y = ds["y"]
#         transform = _affine_from_coords(x, y)

#         vars_present = [v for v in data_vars if v in ds.data_vars]
#         if not vars_present:
#             raise ValueError(
#                 f"No expected data variables {data_vars} in {os.path.basename(nc_in)}"
#             )

#         data_ll = {}
#         first_var = vars_present[0]

#         def mk_rio_da(da_var):
#             da = da_var.astype("float32")
#             da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
#             da = da.rio.write_crs(src_crs, inplace=True)
#             da = da.rio.write_transform(transform, inplace=True)
#             return da

#         # Template grid from first variable
#         template_ll = mk_rio_da(ds[first_var]).rio.reproject(dst_crs)

#         # Reproject all present vars and align to template
#         for v in vars_present:
#             da_src = mk_rio_da(ds[v])
#             da_ll = (
#                 template_ll
#                 if v == first_var
#                 else da_src.rio.reproject_match(template_ll)
#             )
#             da_ll = da_ll.rename({"x": "lon", "y": "lat"})
#             data_ll[v] = da_ll

#         ds_out = xr.Dataset(
#             data_vars=data_ll,
#             coords={"lat": data_ll[first_var]["lat"], "lon": data_ll[first_var]["lon"]},
#             attrs={
#                 "source_file": os.path.basename(nc_in),
#                 "original_crs": src_crs,
#                 "output_crs": dst_crs,
#             },
#         )
#         # carry over useful attrs if present
#         for key in ["rvt_svf_n_dir", "rvt_svf_r_max", "rvt_svf_noise"]:
#             if key in ds.attrs:
#                 ds_out.attrs[key] = ds.attrs[key]

#     # Write compressed NetCDF (prefer h5netcdf/netcdf4; fallback to scipy)
#     encoding = {
#         v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in data_ll.keys()
#     }
#     try:
#         ds_out.to_netcdf(nc_out, engine="h5netcdf", encoding=encoding)
#     except Exception:
#         try:
#             ds_out.to_netcdf(nc_out, engine="netcdf4", encoding=encoding)
#         except Exception:
#             ds_out.to_netcdf(nc_out)  # no compression fallback


def reproject_file_to_latlon(
    nc_in: str,
    nc_out: str,
    data_vars=("svf", "asvf", "opns"),
    dst_crs="EPSG:4326",
) -> None:
    """
    Reproject selected variables from a projected NetCDF grid to lat/lon (WGS84),
    deducing the source CRS from the NetCDF metadata.

    The function expects `x`/`y` coordinates and CRS metadata in the input file,
    ideally written by `process_dem_to_netcdf()`:
      - ds.attrs["crs"] (e.g. "EPSG:32632" or "+proj=utm ...")
      - and/or ds.attrs["crs_wkt"] (preferred when present)

    Parameters
    ----------
    nc_in : str
        Input NetCDF path (projected grid with x/y coords).
    nc_out : str
        Output NetCDF path (lat/lon grid).
    data_vars : iterable of str, optional
        Variables to reproject (only those present are used).
    dst_crs : str or CRS, optional
        Destination CRS (default EPSG:4326).

    Raises
    ------
    ValueError
        If x/y coords missing or none of the requested variables exist.
        If CRS cannot be inferred from the dataset attributes.

    Returns
    -------
    None
    """

    with xr.open_dataset(nc_in) as ds:
        # --- Check coords ---
        if ("x" not in ds.coords) or ("y" not in ds.coords):
            raise ValueError(f"{os.path.basename(nc_in)} is missing x/y coords.")

        x = ds["x"]
        y = ds["y"]
        transform = _affine_from_coords(x, y)

        # --- Determine which variables exist ---
        vars_present = [v for v in data_vars if v in ds.data_vars]
        if not vars_present:
            raise ValueError(
                f"No requested variables {list(data_vars)} found in {os.path.basename(nc_in)}"
            )

        # --- Deduce source CRS ---
        src_crs = None

        # Prefer WKT if available
        if "crs_wkt" in ds.attrs and ds.attrs["crs_wkt"]:
            src_crs = CRS.from_wkt(ds.attrs["crs_wkt"])
        elif "crs" in ds.attrs and ds.attrs["crs"]:
            # could be "EPSG:32632" or proj string
            src_crs = CRS.from_user_input(ds.attrs["crs"])

        if src_crs is None:
            raise ValueError(
                f"Could not infer CRS from {os.path.basename(nc_in)}. "
                "Expected ds.attrs['crs_wkt'] or ds.attrs['crs']."
            )

        # Safety: must be projected (meters)
        if src_crs.is_geographic:
            raise ValueError(
                f"Input CRS is geographic (degrees): {src_crs.to_string()}. "
                "Expected projected CRS in meters."
            )

        # --- Helper to build rio-aware DataArray ---
        def mk_rio_da(da_var):
            da = da_var.astype("float32")
            da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
            da = da.rio.write_crs(src_crs, inplace=False)
            da = da.rio.write_transform(transform, inplace=False)
            return da

        # Template grid from first variable
        first_var = vars_present[0]
        template_ll = mk_rio_da(ds[first_var]).rio.reproject(dst_crs)

        # Reproject all vars and align
        data_ll = {}
        for v in vars_present:
            da_src = mk_rio_da(ds[v])
            da_ll = (
                template_ll
                if v == first_var
                else da_src.rio.reproject_match(template_ll)
            )
            da_ll = da_ll.rename({"x": "lon", "y": "lat"})
            data_ll[v] = da_ll

        ds_out = xr.Dataset(
            data_vars=data_ll,
            coords={
                "lat": data_ll[first_var]["lat"],
                "lon": data_ll[first_var]["lon"],
            },
            attrs={
                "source_file": os.path.basename(nc_in),
                "original_crs": src_crs.to_string(),
                "output_crs": str(dst_crs),
            },
        )

        # carry over useful attrs if present
        for key in [
            "rvt_svf_n_dir",
            "rvt_svf_r_max",
            "rvt_svf_noise",
            "pixel_width",
            "pixel_height",
            "source_dem",
            "no_data",
        ]:
            if key in ds.attrs:
                ds_out.attrs[key] = ds.attrs[key]

    # --- Write compressed NetCDF ---
    encoding = {
        v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in data_ll.keys()
    }
    try:
        ds_out.to_netcdf(nc_out, engine="h5netcdf", encoding=encoding)
    except Exception:
        try:
            ds_out.to_netcdf(nc_out, engine="netcdf4", encoding=encoding)
        except Exception:
            ds_out.to_netcdf(nc_out)
