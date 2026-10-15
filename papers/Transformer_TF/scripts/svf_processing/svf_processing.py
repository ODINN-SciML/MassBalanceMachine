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


def process_dem_to_netcdf(
    dem_path: str,
    # out_path: str,
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

    # # compressed NetCDF
    # encoding = {
    #     v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds_out.data_vars
    # }
    # try:
    #     ds_out.to_netcdf(out_path, engine="h5netcdf", encoding=encoding)
    # except Exception:
    #     try:
    #         ds_out.to_netcdf(out_path, engine="netcdf4", encoding=encoding)
    #     except Exception:
    #         ds_out.to_netcdf(out_path)

    return ds_out


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
#     nc_in: str,
#     nc_out: str,
#     data_vars=("svf", "asvf", "opns"),
#     dst_crs="EPSG:4326",
# ) -> None:
#     """
#     Reproject selected variables from a projected NetCDF grid to lat/lon (WGS84),
#     deducing the source CRS from the NetCDF metadata.

#     The function expects `x`/`y` coordinates and CRS metadata in the input file,
#     ideally written by `process_dem_to_netcdf()`:
#       - ds.attrs["crs"] (e.g. "EPSG:32632" or "+proj=utm ...")
#       - and/or ds.attrs["crs_wkt"] (preferred when present)

#     Parameters
#     ----------
#     nc_in : str
#         Input NetCDF path (projected grid with x/y coords).
#     nc_out : str
#         Output NetCDF path (lat/lon grid).
#     data_vars : iterable of str, optional
#         Variables to reproject (only those present are used).
#     dst_crs : str or CRS, optional
#         Destination CRS (default EPSG:4326).

#     Raises
#     ------
#     ValueError
#         If x/y coords missing or none of the requested variables exist.
#         If CRS cannot be inferred from the dataset attributes.

#     Returns
#     -------
#     None
#     """

#     with xr.open_dataset(nc_in) as ds:
#         # --- Check coords ---
#         if ("x" not in ds.coords) or ("y" not in ds.coords):
#             raise ValueError(f"{os.path.basename(nc_in)} is missing x/y coords.")

#         x = ds["x"]
#         y = ds["y"]
#         transform = _affine_from_coords(x, y)

#         # --- Determine which variables exist ---
#         vars_present = [v for v in data_vars if v in ds.data_vars]
#         if not vars_present:
#             raise ValueError(
#                 f"No requested variables {list(data_vars)} found in {os.path.basename(nc_in)}"
#             )

#         # --- Deduce source CRS ---
#         src_crs = None

#         # Prefer WKT if available
#         if "crs_wkt" in ds.attrs and ds.attrs["crs_wkt"]:
#             src_crs = CRS.from_wkt(ds.attrs["crs_wkt"])
#         elif "crs" in ds.attrs and ds.attrs["crs"]:
#             # could be "EPSG:32632" or proj string
#             src_crs = CRS.from_user_input(ds.attrs["crs"])

#         if src_crs is None:
#             raise ValueError(
#                 f"Could not infer CRS from {os.path.basename(nc_in)}. "
#                 "Expected ds.attrs['crs_wkt'] or ds.attrs['crs']."
#             )

#         # Safety: must be projected (meters)
#         if src_crs.is_geographic:
#             raise ValueError(
#                 f"Input CRS is geographic (degrees): {src_crs.to_string()}. "
#                 "Expected projected CRS in meters."
#             )

#         # --- Helper to build rio-aware DataArray ---
#         def mk_rio_da(da_var):
#             da = da_var.astype("float32")
#             da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
#             da = da.rio.write_crs(src_crs, inplace=False)
#             da = da.rio.write_transform(transform, inplace=False)
#             return da

#         # Template grid from first variable
#         first_var = vars_present[0]
#         template_ll = mk_rio_da(ds[first_var]).rio.reproject(dst_crs)

#         # Reproject all vars and align
#         data_ll = {}
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
#             coords={
#                 "lat": data_ll[first_var]["lat"],
#                 "lon": data_ll[first_var]["lon"],
#             },
#             attrs={
#                 "source_file": os.path.basename(nc_in),
#                 "original_crs": src_crs.to_string(),
#                 "output_crs": str(dst_crs),
#             },
#         )

#         # carry over useful attrs if present
#         for key in [
#             "rvt_svf_n_dir",
#             "rvt_svf_r_max",
#             "rvt_svf_noise",
#             "pixel_width",
#             "pixel_height",
#             "source_dem",
#             "no_data",
#         ]:
#             if key in ds.attrs:
#                 ds_out.attrs[key] = ds.attrs[key]

#     # --- Write compressed NetCDF ---
#     encoding = {
#         v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in data_ll.keys()
#     }
#     try:
#         ds_out.to_netcdf(nc_out, engine="h5netcdf", encoding=encoding)
#     except Exception:
#         try:
#             ds_out.to_netcdf(nc_out, engine="netcdf4", encoding=encoding)
#         except Exception:
#             ds_out.to_netcdf(nc_out)


def reproject_ds_to_latlon(
    ds: xr.Dataset,
    data_vars=("svf", "asvf", "opns"),
    dst_crs="EPSG:4326",
    clip_to_valid: bool = False,
) -> xr.Dataset:
    """
    Reproject selected vars from a projected x/y Dataset to lat/lon (WGS84).
    Works purely in-memory (no need to save/reopen NetCDF).

    Expects:
      - ds.coords["x"], ds.coords["y"]
      - ds.attrs["crs_wkt"] or ds.attrs["crs"]
    """
    if ("x" not in ds.coords) or ("y" not in ds.coords):
        raise ValueError("Dataset is missing x/y coords.")

    # pick vars that exist
    vars_present = [v for v in data_vars if v in ds.data_vars]
    if not vars_present:
        raise ValueError(f"No requested variables found. Requested={list(data_vars)}")

    # infer CRS
    if ds.attrs.get("crs_wkt"):
        src_crs = CRS.from_wkt(ds.attrs["crs_wkt"])
    elif ds.attrs.get("crs"):
        src_crs = CRS.from_user_input(ds.attrs["crs"])
    else:
        raise ValueError(
            "Could not infer CRS from ds.attrs['crs_wkt'] or ds.attrs['crs']."
        )

    if src_crs.is_geographic:
        raise ValueError(f"Input CRS is geographic (degrees): {src_crs.to_string()}")

    # build transform from coords (center-of-pixel)
    transform = _affine_from_coords(ds["x"], ds["y"])

    def mk_rio_da(da: xr.DataArray) -> xr.DataArray:
        da = da.astype("float32")
        da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
        da = da.rio.write_crs(src_crs, inplace=False)
        da = da.rio.write_transform(transform, inplace=False)
        return da

    first = vars_present[0]
    template_ll = mk_rio_da(ds[first]).rio.reproject(dst_crs)
    template_ll = template_ll.rename({"x": "lon", "y": "lat"})

    data_ll = {first: template_ll}

    for v in vars_present[1:]:
        da_ll = mk_rio_da(ds[v]).rio.reproject_match(
            template_ll.rename({"lon": "x", "lat": "y"})
        )
        da_ll = da_ll.rename({"x": "lon", "y": "lat"})
        data_ll[v] = da_ll

    ds_ll = xr.Dataset(
        data_vars=data_ll,
        coords={"lat": data_ll[first]["lat"], "lon": data_ll[first]["lon"]},
        attrs={
            "original_crs": src_crs.to_string(),
            "output_crs": str(dst_crs),
            "source_dem": ds.attrs.get("source_dem", ""),
        },
    )

    # carry over useful attrs
    for key in [
        "rvt_svf_n_dir",
        "rvt_svf_r_max",
        "rvt_svf_noise",
        "asvf_level",
        "asvf_dir",
        "pixel_width",
        "pixel_height",
        "no_data",
    ]:
        if key in ds.attrs:
            ds_ll.attrs[key] = ds.attrs[key]

    if clip_to_valid:
        # optional: clip silly fill edges if any (depends on reprojection/resampling)
        for v in vars_present:
            ds_ll[v] = ds_ll[v].where(np.isfinite(ds_ll[v]))

    return ds_ll
