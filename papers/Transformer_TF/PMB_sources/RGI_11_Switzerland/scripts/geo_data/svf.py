import xarray as xr
import numpy as np
import os


def merge_svf_into_ds(ds_latlon, sgi_id, path_xr_svf):
    """
    Merge SVF-related variables into a glacier lat/lon dataset.

    Loads `<path_xr_svf>/<sgi_id>_svf_latlon.nc` if present, interpolates to the
    grid of `ds_latlon` if needed, and adds masked versions (`masked_*`) when
    `glacier_mask` exists.

    Parameters
    ----------
    ds_latlon : xarray.Dataset
        Dataset on a lat/lon grid (must contain `lat` and `lon` coords).
    sgi_id : str
        SGI glacier identifier used to locate the SVF file.
    path_xr_svf : str
        Folder containing `*_svf_latlon.nc` files.

    Returns
    -------
    xarray.Dataset
        Dataset with SVF variables added if available; otherwise unchanged.
    """
    svf_path = os.path.join(path_xr_svf, f"{sgi_id}_svf_latlon.nc")
    if not os.path.exists(svf_path):
        print(f"SVF not found for {sgi_id}: {svf_path}")
        return ds_latlon

    ds_svf = xr.open_dataset(svf_path)

    # sort for interpolation stability
    if ds_latlon.lon[0] > ds_latlon.lon[-1]:
        ds_latlon = ds_latlon.sortby("lon")
    if ds_latlon.lat[0] > ds_latlon.lat[-1]:
        ds_latlon = ds_latlon.sortby("lat")
    if ds_svf.lon[0] > ds_svf.lon[-1]:
        ds_svf = ds_svf.sortby("lon")
    if ds_svf.lat[0] > ds_svf.lat[-1]:
        ds_svf = ds_svf.sortby("lat")

    svf_vars = [v for v in ["svf", "asvf", "opns"] if v in ds_svf.data_vars]
    if not svf_vars:
        return ds_latlon

    # exact match vs. interpolation
    if np.array_equal(ds_latlon.lon.values, ds_svf.lon.values) and np.array_equal(
        ds_latlon.lat.values, ds_svf.lat.values
    ):
        ds_latlon = xr.merge([ds_latlon, ds_svf[svf_vars]])
    else:
        svf_on_grid = ds_svf[svf_vars].interp(
            lon=ds_latlon.lon, lat=ds_latlon.lat, method="linear"
        )
        for v in svf_vars:
            svf_on_grid[v] = svf_on_grid[v].astype("float32")
        ds_latlon = ds_latlon.assign(**{v: svf_on_grid[v] for v in svf_vars})

    # masked versions (if glacier_mask exists)
    if "glacier_mask" in ds_latlon:
        gmask = xr.where(ds_latlon["glacier_mask"] == 1, 1.0, np.nan)
        for v in svf_vars:
            ds_latlon[f"masked_{v}"] = gmask * ds_latlon[v]
    return ds_latlon
