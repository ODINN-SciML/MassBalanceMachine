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


def open_and_merge_svf(
    ds_latlon: xr.Dataset, glacier_name: str, year: int, path_xr_svf: str
) -> xr.Dataset:
    """
    Load and merge Sky-View-Factor (SVF) data onto a glacier lat/lon dataset.

    This helper function attempts to open a precomputed SVF NetCDF file for a given
    glacier and year, harmonize its coordinates with those of an existing dataset,
    and merge or interpolate the SVF variables onto the target grid.

    Processing steps:
    -----------------
    1. Look for an SVF file named `<glacier_name>_<year>_svf_latlon.nc`.
    2. If the file is missing, return `ds_latlon` unchanged.
    3. Normalize coordinate names (e.g., `longitude` → `lon`, `latitude` → `lat`).
    4. Ensure longitudes are in the same range as the target dataset
       (convert 0–360 to -180–180 if necessary).
    5. Sort both datasets by ascending lon/lat to ensure interpolation stability.
    6. Identify available SVF variables (`svf`, `asvf`, `opns`).
    7. If the SVF grid exactly matches `ds_latlon`, merge directly.
       Otherwise, interpolate SVF fields onto the target grid.
    8. If a `glacier_mask` variable exists, create additional masked SVF fields
       (`masked_svf`, `masked_asvf`, `masked_opns`).

    Parameters
    ----------
    ds_latlon : xarray.Dataset
        Base glacier dataset on a lat/lon grid to which SVF variables will be added.
        Must contain `lon` and `lat` coordinates and optionally `glacier_mask`.
    glacier_name : str
        Name of the glacier used to construct the SVF filename.
    year : int
        Year of the SVF dataset.
    path_xr_svf : str
        Directory containing SVF NetCDF files.

    Returns
    -------
    xarray.Dataset
        Dataset with SVF variables merged in (and optionally masked versions),
        or the original `ds_latlon` if no SVF data are available.

    Notes
    -----
    - Only variables named `svf`, `asvf`, and `opns` are considered.
    - Interpolation is performed using linear interpolation on the lat/lon grid.
    - All merged SVF variables are cast to `float32` for storage efficiency.
    - Missing or malformed SVF files do not raise exceptions; the function
      simply returns the input dataset and prints a warning message.
    """
    svf_path = os.path.join(path_xr_svf, f"{glacier_name}_{year}_svf_latlon.nc")
    if not os.path.exists(svf_path):
        print(f"SVF not found: {svf_path}")
        return ds_latlon

    with xr.open_dataset(svf_path, decode_cf=True) as ds_svf_raw:
        ds_svf = ds_svf_raw

        # normalize coordinate names
        ren = {}
        if "longitude" in ds_svf.coords and "lon" not in ds_svf.coords:
            ren["longitude"] = "lon"
        if "latitude" in ds_svf.coords and "lat" not in ds_svf.coords:
            ren["latitude"] = "lat"
        if ren:
            ds_svf = ds_svf.rename(ren)

        if not ({"lon", "lat"} <= set(ds_svf.coords)):
            print(f"SVF lacks lon/lat: {svf_path}")
            return ds_latlon

        # longitude range normalization (0–360 -> -180–180) if needed
        if float(ds_svf.lon.max()) > 180 and float(ds_latlon.lon.min()) < 0:
            ds_svf = ds_svf.assign_coords(lon=((ds_svf.lon + 180) % 360) - 180)

        # sort ascending for interp stability
        if ds_svf.lon[0] > ds_svf.lon[-1]:
            ds_svf = ds_svf.sortby("lon")
        if ds_svf.lat[0] > ds_svf.lat[-1]:
            ds_svf = ds_svf.sortby("lat")
        if ds_latlon.lon[0] > ds_latlon.lon[-1]:
            ds_latlon = ds_latlon.sortby("lon")
        if ds_latlon.lat[0] > ds_latlon.lat[-1]:
            ds_latlon = ds_latlon.sortby("lat")

        svf_vars = [v for v in ["svf", "asvf", "opns"] if v in ds_svf.data_vars]
        if not svf_vars:
            print(f"No SVF vars in {svf_path}")
            return ds_latlon

        same_lon = np.array_equal(ds_latlon.lon.values, ds_svf.lon.values)
        same_lat = np.array_equal(ds_latlon.lat.values, ds_svf.lat.values)

        if same_lon and same_lat:
            merged = xr.merge([ds_latlon, ds_svf[svf_vars]])
        else:
            svf_on_grid = ds_svf[svf_vars].interp(
                lon=ds_latlon.lon, lat=ds_latlon.lat, method="linear"
            )
            for v in svf_vars:
                svf_on_grid[v] = svf_on_grid[v].astype("float32")
            merged = ds_latlon.assign(**{v: svf_on_grid[v] for v in svf_vars})

        # add masked versions
        if "glacier_mask" in merged:
            gmask = xr.where(merged["glacier_mask"] == 1, 1.0, np.nan).astype("float32")
            for v in svf_vars:
                merged[f"masked_{v}"] = (gmask * merged[v]).astype("float32")
        return merged
