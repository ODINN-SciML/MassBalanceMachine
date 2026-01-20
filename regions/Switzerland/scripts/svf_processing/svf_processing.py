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

# NOTE: NEEDS TO BE RUN IN SEPARATE RTV ENVIRONMENT WITH RTV INSTALLED


# ----------------------- SVF processing of DEMs -----------------------
def process_dem_to_netcdf(
    dem_path: str,
    out_path: str,
    svf_n_dir=16,
    svf_r_max=10,
    svf_noise=0,
    asvf_level=1,
    asvf_dir=315,
) -> None:
    # --- open with rasterio for authoritative geoinfo ---
    with rasterio.open(dem_path) as src:
        transform: Affine = src.transform
        crs = src.crs
        width, height = src.width, src.height
        # pixel sizes (a > 0, e < 0 for north-up rasters)
        pix_w = float(transform.a)
        pix_h = abs(float(transform.e))
        # center-of-pixel coordinates from transform
        x = transform.c + (np.arange(width) + 0.5) * pix_w
        y = transform.f + (np.arange(height) + 0.5) * transform.e  # note e is negative

        # read DEM as array (band 1)
        dem = src.read(1)
        no_data = src.nodata

    if max(pix_w, pix_h) < 0.0001:
        raise ValueError(
            "DEM appears to be in degrees; compute SVF on a metric CRS first."
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

    # --- build Dataset with correct LV95 x/y coords ---
    da_svf = xr.DataArray(
        svf_arr,
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name="svf",
        attrs={"description": "Sky-View Factor", "units": "unitless"},
    )
    da_asvf = xr.DataArray(
        asvf_arr,
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name="asvf",
        attrs={
            "description": "Anisotropic SVF",
            "units": "unitless",
            "asvf_level": asvf_level,
            "asvf_dir": asvf_dir,
        },
    )
    da_opns = xr.DataArray(
        opns_arr,
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name="opns",
        attrs={"description": "Positive Openness", "units": "radians"},
    )

    ds_out = xr.Dataset(
        data_vars={"svf": da_svf, "asvf": da_asvf, "opns": da_opns},
        coords={"x": x, "y": y},
        attrs={
            "source_dem": os.path.basename(dem_path),
            "no_data": no_data if no_data is not None else np.nan,
            "crs": str(crs) if crs is not None else "EPSG:2056",
            "rvt_svf_n_dir": svf_n_dir,
            "rvt_svf_r_max": svf_r_max,
            "rvt_svf_noise": svf_noise,
        },
    )

    # compressed NetCDF (no fixed chunksizes to avoid tiny rasters failing)
    encoding = {
        v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds_out.data_vars
    }
    try:
        ds_out.to_netcdf(out_path, engine="h5netcdf", encoding=encoding)
    except Exception:
        try:
            ds_out.to_netcdf(out_path, engine="netcdf4", encoding=encoding)
        except Exception:
            ds_out.to_netcdf(out_path)  # scipy fallback (no compression)


def _affine_from_coords(x, y):
    """
    Build an Affine transform from 1D x/y center coordinates.
    Assumes x increases to the right; y typically decreases downward.
    """
    resx = float(np.median(np.diff(x.values)))
    resy = float(np.median(np.diff(y.values)))
    pix_h = -abs(resy)  # north-up convention
    x0 = float(x.values[0]) - resx / 2.0
    y0 = float(y.values[0]) + abs(resy) / 2.0
    return Affine(resx, 0.0, x0, 0.0, pix_h, y0)


def reproject_file_to_latlon(
    nc_in: str, nc_out: str, data_vars, src_crs, dst_crs
) -> None:
    # Open/close cleanly so we don't leave files locked
    with xr.open_dataset(nc_in) as ds:
        # Check coords
        if ("x" not in ds.coords) or ("y" not in ds.coords):
            raise ValueError(f"{os.path.basename(nc_in)} is missing x/y coords.")

        x = ds["x"]
        y = ds["y"]
        transform = _affine_from_coords(x, y)

        vars_present = [v for v in data_vars if v in ds.data_vars]
        if not vars_present:
            raise ValueError(
                f"No expected data variables {data_vars} in {os.path.basename(nc_in)}"
            )

        data_ll = {}
        first_var = vars_present[0]

        def mk_rio_da(da_var):
            da = da_var.astype("float32")
            da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
            da = da.rio.write_crs(src_crs, inplace=True)
            da = da.rio.write_transform(transform, inplace=True)
            return da

        # Template grid from first variable
        template_ll = mk_rio_da(ds[first_var]).rio.reproject(dst_crs)

        # Reproject all present vars and align to template
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
            coords={"lat": data_ll[first_var]["lat"], "lon": data_ll[first_var]["lon"]},
            attrs={
                "source_file": os.path.basename(nc_in),
                "original_crs": src_crs,
                "output_crs": dst_crs,
            },
        )
        # carry over useful attrs if present
        for key in ["rvt_svf_n_dir", "rvt_svf_r_max", "rvt_svf_noise"]:
            if key in ds.attrs:
                ds_out.attrs[key] = ds.attrs[key]

    # Write compressed NetCDF (prefer h5netcdf/netcdf4; fallback to scipy)
    encoding = {
        v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in data_ll.keys()
    }
    try:
        ds_out.to_netcdf(nc_out, engine="h5netcdf", encoding=encoding)
    except Exception:
        try:
            ds_out.to_netcdf(nc_out, engine="netcdf4", encoding=encoding)
        except Exception:
            ds_out.to_netcdf(nc_out)  # no compression fallback
