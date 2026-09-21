"""
Interpolate air temperature at a target elevation (given as geopotential
from a fine-resolution DEM, e.g. ERA5-Land) using pressure-level
geopotential-height and temperature fields from a coarser model (e.g.
CMIP6), regridded onto the DEM's grid.

Pipeline
--------
1. Bilinearly regrid ds_geopot_cmip and ds_temp (each pressure level, each
   time step) from the coarse CMIP (lat, lon) grid onto the fine ERA5
   (latitude, longitude) grid. Points outside the CMIP grid's spatial
   coverage come back as NaN and are dropped.
2. Per (latitude, longitude, time) column: invert geopotential-height(plev)
   to find the pressure level p_star matching the DEM elevation (linear
   interpolation, with linear extrapolation beyond the available levels
   using the outermost *valid* segment -- masked/missing levels, e.g.
   below a grid cell's real surface pressure, are dropped per-column
   before sorting/extrapolating).
3. Interpolate temperature(plev) at p_star, the same way.

Inputs
------
ds_geopotential : xr.Dataset
    Fixed-in-time geopotential of the target surface (the output grid),
    e.g. from a DEM. Dims: (longitude, latitude[, time=1]).
    Units: m^2/s^2 (geopotential, not geopotential height).
ds_geopot_cmip : xr.Dataset
    Geopotential HEIGHT at pressure levels, on the coarse model grid.
    Dims: (lat, lon, plev, time). Units: meters.
ds_temp : xr.Dataset
    Air temperature at pressure levels, on the same coarse model grid.
    Dims: (lat, lon, plev, time). Must share the same `plev` coordinate
    as ds_geopot_cmip.

Output
------
temp_at_elevation : xr.DataArray
    Air temperature interpolated to the DEM elevation, on the DEM's
    (latitude, longitude) grid, dims (latitude, longitude, time).
    Points outside the CMIP grid's footprint are NaN. Dask-backed and
    lazy when use_dask=True (call `.compute()`/`.load()` to materialize);
    a plain in-memory numpy-backed array when use_dask=False.
p_star : xr.DataArray
    The interpolated/extrapolated pressure level at each point/time,
    dims (latitude, longitude, time). Useful for sanity-checking how far
    values were extrapolated (see the sanity-check snippet discussed
    earlier in this conversation). Same laziness behavior as above.
"""

import os
import numpy as np
import xarray as xr

from data_processing.climate_projections.climate_data_download import (
    path_climate_data,
    SURFACE_FLUX_VARS,
)
from data_processing.get_climate_data import path_climate_data as path_climate_data_ERA5
from data_processing.get_climate_data import _load_datasets

# ---- 1. Adjust these to your actual variable / coordinate names ----------
GEOPOT_TARGET_VAR = "z"  # variable name in ds_geopotential (m^2/s^2)
GEOPOT_CMIP_VAR = "zg"  # variable name in ds_geopot_cmip (meters)
TEMP_VAR = "ta"  # variable name in ds_temp
PLEV_DIM = "plev"

LAT_DIM_TARGET, LON_DIM_TARGET = (
    "latitude",
    "longitude",
)  # ds_geopotential (output grid)
LAT_DIM_CMIP, LON_DIM_CMIP = "lat", "lon"  # ds_geopot_cmip / ds_temp (input grid)

STANDARD_GRAVITY = 9.80665  # m/s^2, to convert geopotential -> geopotential height

# Chunking defaults for the final vertical interpolation. "points" is the
# stacked (latitude, longitude) dimension -- at ERA5 resolution this can be
# tens of thousands of points, so it needs chunking (along with time) for
# the per-column apply_ufunc step to run in parallel without blowing up
# memory. Tune these to your machine.
DEFAULT_POINTS_CHUNK = 2000
DEFAULT_TIME_CHUNK = 60
# ----------------------------------------------------------------------------


def _linear_interp_extrap(x, xp, fp):
    """Like np.interp(x, xp, fp), but:
    - drops any (xp, fp) pair where either value is NaN before doing
      anything else -- important for CMIP6 pressure-level output, which
      masks levels that fall below a grid cell's actual surface pressure
      (e.g. near-surface levels for higher-elevation columns). Using a
      fixed/pre-sorted order without dropping these first silently lets a
      masked value land at the array boundary and contaminate the
      extrapolation slope with NaN.
    - sorts the remaining valid pairs by xp itself, so `xp`/`fp` need not
      be pre-sorted or share a single fixed order across columns.
    - linearly EXTRAPOLATES beyond both ends using the slope of the
      outermost *valid* segment, instead of clamping to the boundary value.

    Returns NaN if fewer than 2 valid (xp, fp) pairs remain in this column.
    """
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)

    valid = np.isfinite(xp) & np.isfinite(fp)
    if valid.sum() < 2 or not np.isfinite(x):
        return np.nan

    xp_v, fp_v = xp[valid], fp[valid]
    order = np.argsort(xp_v)
    xp_s, fp_s = xp_v[order], fp_v[order]

    y = np.interp(x, xp_s, fp_s)  # clamped baseline; overridden at the tails below

    if x < xp_s[0]:
        slope_low = (fp_s[1] - fp_s[0]) / (xp_s[1] - xp_s[0])
        y = fp_s[0] + slope_low * (x - xp_s[0])
    elif x > xp_s[-1]:
        slope_high = (fp_s[-1] - fp_s[-2]) / (xp_s[-1] - xp_s[-2])
        y = fp_s[-1] + slope_high * (x - xp_s[-1])

    return y


def _regrid_cmip_to_era5_grid(da_cmip, lat_target, lon_target_180):
    """Bilinearly regrid a CMIP field (dims incl. lat, lon, e.g. also plev
    and time) onto the ERA5 target grid (dims latitude, longitude).
    Target points outside the CMIP grid's coverage come back as NaN.
    `lon_target_180` must already be in the same longitude convention as
    the CMIP grid (see the -180/180 conversion in the caller).
    """
    return da_cmip.interp(
        {LAT_DIM_CMIP: lat_target, LON_DIM_CMIP: lon_target_180},
        method="linear",
    )


def interp_temperature_at_elevation(
    ds_geopotential,
    ds_geopot_cmip,
    ds_temp,
    use_dask=True,
    points_chunk=DEFAULT_POINTS_CHUNK,
    time_chunk=DEFAULT_TIME_CHUNK,
):
    """
    use_dask : bool, default True
        If True, chunk the stacked (points, time) arrays and run the
        per-column interpolation lazily/in parallel via dask -- call
        `.compute()`/`.load()` on the results afterward. If False, skip
        chunking entirely and compute eagerly with plain numpy: simpler to
        debug and fine for small grids (e.g. the original coarse-grid
        version of this pipeline), but will load everything into memory
        at once and run single-threaded -- likely too slow/memory-heavy
        at full ERA5 resolution. `points_chunk`/`time_chunk` are ignored
        when use_dask=False.
    """
    # --- Prep the target (ERA5) grid: drop the singleton time dim, convert
    #     geopotential -> geopotential height. This stays at native
    #     resolution; it is NOT regridded.
    target_geopot = ds_geopotential[GEOPOT_TARGET_VAR]
    if "time" in target_geopot.dims:
        target_geopot = target_geopot.squeeze("time", drop=True)
    target_geopot = target_geopot / STANDARD_GRAVITY  # m^2/s^2 -> m

    # ERA5 longitude is typically 0-360 (e.g. wraps 358, 359, 0, 1, ...);
    # convert to -180/180 to match the CMIP grid's convention before using
    # it as the interpolation target.
    lat_target = target_geopot[LAT_DIM_TARGET]
    lon_target_180 = ((target_geopot[LON_DIM_TARGET] + 180) % 360) - 180

    geopot_cmip = ds_geopot_cmip[GEOPOT_CMIP_VAR]
    temp = ds_temp[TEMP_VAR]

    # --- Step 1: regrid the CMIP geopotential-height and temperature
    #     fields (every plev, every time) onto the ERA5 (latitude,
    #     longitude) grid.
    geopot_cmip_era5 = _regrid_cmip_to_era5_grid(
        geopot_cmip, lat_target, lon_target_180
    )
    temp_era5 = _regrid_cmip_to_era5_grid(temp, lat_target, lon_target_180)

    # --- Step 2: drop ERA5 points outside the CMIP grid's spatial coverage.
    #     Such points are NaN across every plev/time after the horizontal
    #     regrid; a point with at least one valid plev/time is considered
    #     "inside" the footprint (missing individual levels are handled
    #     later, per-column, by _linear_interp_extrap).
    footprint_ok = geopot_cmip_era5.notnull().any([PLEV_DIM, "time"])

    target_geopot_stacked = target_geopot.where(footprint_ok).stack(
        points=(LAT_DIM_TARGET, LON_DIM_TARGET)
    )
    target_geopot_valid = target_geopot_stacked.dropna("points")

    geopot_cmip_valid = geopot_cmip_era5.stack(
        points=(LAT_DIM_TARGET, LON_DIM_TARGET)
    ).sel(points=target_geopot_valid["points"])
    temp_valid = temp_era5.stack(points=(LAT_DIM_TARGET, LON_DIM_TARGET)).sel(
        points=target_geopot_valid["points"]
    )

    # --- Step 3: chunk for parallel, memory-friendly processing (or load
    #     eagerly into plain numpy if dask is disabled). At ERA5 resolution
    #     this is ~tens of thousands of points x ~1000 time steps, so both
    #     dims get chunked when using dask (plev stays whole -- it's a core
    #     dim for the per-column interpolation and must not be split).
    if use_dask:
        target_geopot_valid = target_geopot_valid.chunk({"points": points_chunk})
        geopot_cmip_valid = geopot_cmip_valid.chunk(
            {"points": points_chunk, "time": time_chunk}
        )
        temp_valid = temp_valid.chunk({"points": points_chunk, "time": time_chunk})
    else:
        target_geopot_valid = target_geopot_valid.load()
        geopot_cmip_valid = geopot_cmip_valid.load()
        temp_valid = temp_valid.load()

    dask_mode = "parallelized" if use_dask else "forbidden"

    plev_vals = geopot_cmip[PLEV_DIM].values

    # --- Step 4: invert geopotential_height(plev) -> plev at the target
    #     elevation, independently for every (points, time) column.
    def _plev_at_target(target, geopot_profile):
        return _linear_interp_extrap(target, geopot_profile, plev_vals)

    p_star = xr.apply_ufunc(
        _plev_at_target,
        target_geopot_valid,
        geopot_cmip_valid,
        input_core_dims=[[], [PLEV_DIM]],
        output_core_dims=[[]],
        vectorize=True,
        dask=dask_mode,
        output_dtypes=[float],
    )

    # --- Step 5: interpolate temperature(plev) at p_star, again per column.
    def _temp_at_plev(p_target, temp_profile):
        return _linear_interp_extrap(p_target, plev_vals, temp_profile)

    temp_at_elevation = xr.apply_ufunc(
        _temp_at_plev,
        p_star,
        temp_valid,
        input_core_dims=[[], [PLEV_DIM]],
        output_core_dims=[[]],
        vectorize=True,
        dask=dask_mode,
        output_dtypes=[float],
    )

    # --- Restore the rectangular (latitude, longitude) grid; dropped
    #     points come back as NaN automatically.
    temp_at_elevation = temp_at_elevation.unstack("points")
    p_star = p_star.unstack("points")

    temp_at_elevation.name = "t2m"
    temp_at_elevation.attrs.update(
        long_name="Air temperature interpolated to DEM elevation",
        units=temp.attrs.get("units", ""),
    )
    return temp_at_elevation, p_star


def bias_cor_suffix(bias_correction_period):
    """Folder name identifying a bias correction period, None meaning no correction."""
    if bias_correction_period is None:
        return "no_bias_cor"
    return f"bias_cor_{bias_correction_period[0]}_{bias_correction_period[1]}"


def path_gridded_temp(region, ssp, gcm, bias_cor_suffix):
    """Return path of gridded temperature for a given region (string or integer)."""
    if not isinstance(region, str):
        region = f"{region:02d}"
    return f".data/CMIP6/gridded_temp/{bias_cor_suffix}/{region}/{ssp}/{gcm}.nc"


def regridding(region, ssp, gcm):
    gridded_temp_file = path_gridded_temp(region, ssp, gcm, "no_bias_cor")
    if not os.path.isfile(gridded_temp_file):
        os.makedirs(os.path.dirname(gridded_temp_file), exist_ok=True)

        # CMIP data
        ds_temp = xr.open_dataset(
            path_climate_data(region, ssp, gcm, "air_temperature") + "/data.nc"
        )
        ds_geopot_cmip = xr.open_dataset(
            path_climate_data(region, ssp, gcm, "geopotential_height") + "/data.nc"
        )

        # ERA5 geopotential
        local_path = path_climate_data_ERA5(region)
        climate_data = local_path + "era5_monthly_averaged_data.nc"
        geopotential_data = local_path + "era5_geopotential_pressure.nc"
        ds_geopotential_era5 = _load_datasets(climate_data, geopotential_data, False)[1]

        # Project CMIP temperature onto the ERA5 grid
        temp_at_elevation, p_star = interp_temperature_at_elevation(
            ds_geopotential_era5, ds_geopot_cmip, ds_temp, use_dask=False
        )

        plev_min, plev_max = float(ds_geopot_cmip["plev"].min()), float(
            ds_geopot_cmip["plev"].max()
        )
        p_min, p_max = float(p_star.min()), float(p_star.max())

        n_clamped_low = (p_star <= plev_min).sum().item()
        n_clamped_high = (p_star >= plev_max).sum().item()
        n_total = p_star.notnull().sum().item()

        print(f"plev range available: [{plev_min}, {plev_max}]")
        print(f"p_star range obtained: [{p_min}, {p_max}]")
        print(
            f"points clamped at low end (plev <= {plev_min}): {n_clamped_low} / {n_total}"
        )
        print(
            f"points clamped at high end (plev >= {plev_max}): {n_clamped_high} / {n_total}"
        )

        RHO_SURFACE = 1.225  # kg/m^3, standard near-surface air density

        # how far beyond the model's lowest level (100000 Pa) each point was extrapolated
        p_excess = (p_star - plev_max).clip(
            min=0
        )  # Pa, zero where no extrapolation was needed
        z_offset = p_excess / (
            RHO_SURFACE * STANDARD_GRAVITY
        )  # rough elevation offset, meters

        # Check that the extrapolation distance outside of pressure levels is not too large
        thresh = 1000
        assert int((z_offset > thresh).sum()) == 0

        temp_at_elevation.to_netcdf(gridded_temp_file)

    else:
        temp_at_elevation = xr.open_dataset(gridded_temp_file)

    # The grid keeps the longitudes of the ERA5 geopotential, which wrap from 360 to
    # 0 (358, ..., 359.9, 0, ..., 21 for the Alps). The ERA5 climate data it is
    # compared to runs monotonically in -180..180.
    return temp_at_elevation.assign_coords(
        {LON_DIM_TARGET: ((temp_at_elevation[LON_DIM_TARGET] + 180) % 360) - 180}
    ).sortby(LON_DIM_TARGET)


def combine_historical_and_projection(
    gcm_hist: xr.DataArray,
    gcm_proj: xr.DataArray,
) -> xr.DataArray:
    """
    Concatenate a CMIP6 historical run and its corresponding SSP
    projection run into a single continuous time series.

    Handles the two common gotchas: overlapping time steps at the
    historical/SSP boundary, and non-monotonic or duplicate timestamps.
    """
    # --- check grids match ----------------------------------------------
    if not (
        gcm_hist.sizes.get("latitude") == gcm_proj.sizes.get("latitude")
        and gcm_hist.sizes.get("longitude") == gcm_proj.sizes.get("longitude")
    ):
        raise ValueError("gcm_hist and gcm_proj are not on the same grid.")

    # --- check calendars match --------------------------------------
    cal_hist = (
        gcm_hist["time"].dt.calendar
        if hasattr(gcm_hist["time"].dt, "calendar")
        else None
    )
    cal_proj = (
        gcm_proj["time"].dt.calendar
        if hasattr(gcm_proj["time"].dt, "calendar")
        else None
    )
    if cal_hist is not None and cal_proj is not None and cal_hist != cal_proj:
        raise ValueError(
            f"Calendar mismatch: historical uses '{cal_hist}', "
            f"projection uses '{cal_proj}'. Convert one before concatenating."
        )

    # --- concatenate and drop any overlap/duplicates ---------------------
    gcm_full = xr.concat([gcm_hist, gcm_proj], dim="time")
    gcm_full = gcm_full.sortby("time")

    _, unique_idx = np.unique(gcm_full["time"].values, return_index=True)
    n_dupes = gcm_full.sizes["time"] - len(unique_idx)
    if n_dupes > 0:
        print(f"Dropped {n_dupes} duplicate timestep(s) at the hist/SSP boundary.")
        gcm_full = gcm_full.isel(time=sorted(unique_idx))

    return gcm_full


def _bbox_slice(coord: xr.DataArray, lo: float, hi: float) -> slice:
    """Build a .sel-compatible slice matching this coordinate's own
    monotonic direction (ascending or descending)."""
    ascending = bool(coord[0] < coord[-1])
    return slice(lo, hi) if ascending else slice(hi, lo)


def _crop_to_common_grid(
    obs: xr.DataArray,
    gcm: xr.DataArray,
    tolerance: float = 1e-3,
) -> tuple[xr.DataArray, xr.DataArray]:
    obs_lat, gcm_lat = obs["latitude"], gcm["latitude"]
    obs_lon, gcm_lon = obs["longitude"], gcm["longitude"]

    lat_min = max(float(obs_lat.min()), float(gcm_lat.min()))
    lat_max = min(float(obs_lat.max()), float(gcm_lat.max()))
    lon_min = max(float(obs_lon.min()), float(gcm_lon.min()))
    lon_max = min(float(obs_lon.max()), float(gcm_lon.max()))

    if lat_min > lat_max or lon_min > lon_max:
        raise ValueError("No overlapping bounding box between obs and gcm.")

    # slice direction determined SEPARATELY for each array
    obs_cropped = obs.sel(
        latitude=_bbox_slice(obs_lat, lat_min, lat_max),
        longitude=_bbox_slice(obs_lon, lon_min, lon_max),
    )
    gcm_cropped = gcm.sel(
        latitude=_bbox_slice(gcm_lat, lat_min, lat_max),
        longitude=_bbox_slice(gcm_lon, lon_min, lon_max),
    )

    if obs_cropped.sizes["latitude"] == 0 or obs_cropped.sizes["longitude"] == 0:
        raise ValueError(
            "Bounding-box crop of obs is empty -- check obs coordinate order/values."
        )
    if gcm_cropped.sizes["latitude"] == 0 or gcm_cropped.sizes["longitude"] == 0:
        raise ValueError(
            "Bounding-box crop of gcm is empty -- check gcm coordinate order/values."
        )

    gcm_aligned = gcm_cropped.reindex(
        latitude=obs_cropped["latitude"],
        longitude=obs_cropped["longitude"],
        method="nearest",
        tolerance=tolerance,
    )

    gcm_first = gcm_aligned.isel(time=0)
    if isinstance(gcm_first, xr.Dataset):
        gcm_first = gcm_first.to_array().isnull().any("variable")
    else:
        gcm_first = gcm_first.isnull()
    n_nan = int(gcm_first.sum())
    if n_nan > 0:
        raise ValueError(
            f"{n_nan} grid point(s) in gcm did not match any obs point "
            f"within tolerance={tolerance}. Grids may be offset by more "
            f"than expected -- inspect coordinates before proceeding."
        )

    print(
        f"Cropped to common grid: {obs_cropped.sizes['latitude']} x "
        f"{obs_cropped.sizes['longitude']} points."
    )

    return obs_cropped, gcm_aligned


def _select_years(ds, y0, y1, name):
    """Select the years y0 to y1 (inclusive) of `ds`, raising if `ds` does not
    fully cover them."""
    years = ds["time"].dt.year
    y_min, y_max = int(years.min()), int(years.max())
    if y_min > y0 or y_max < y1:
        raise ValueError(
            f"'{name}' does not fully cover the reference period "
            f"{y0}-{y1} (available: {y_min}-{y_max}). If this is a "
            f"future/SSP-only GCM array, concatenate the matching "
            f"historical run first."
        )
    return ds.sel(time=(years >= y0) & (years <= y1))


def bias_correct_temperature(
    obs: xr.DataArray,
    gcm: xr.DataArray,
    bias_correction_period: tuple[int, int],
) -> xr.DataArray:
    """
    Bias-correct a GCM temperature series against an observational
    dataset, using the additive anomaly (delta) method applied per
    grid cell and per calendar month. Automatically crops both inputs
    to their common (latitude, longitude) footprint first.

    hat_T(lat, lon, t) = obs_clim(lat, lon, month(t))
                        + [ gcm(lat, lon, t) - gcm_clim(lat, lon, month(t)) ]

    where obs_clim / gcm_clim are monthly climatologies computed once,
    over `bias_correction_period`, and applied identically to every
    year of `gcm` (historical or future).

    Parameters
    ----------
    obs : xr.DataArray
        Observational data (e.g. ERA5 interpolated to DEM elevation),
        on the same (latitude, longitude) grid as `gcm`. Must cover
        the years in `bias_correction_period`.
    gcm : xr.DataArray
        GCM data (e.g. CMIP6 historical+ssp, interpolated to DEM
        elevation), on the same grid as `obs`. Must ALSO cover the
        years in `bias_correction_period` -- if you only have the
        SSP-era slice (e.g. 2015-2099), concatenate the matching
        historical run onto it before calling this function.
    bias_correction_period : tuple[int, int]
        (start_year, end_year), inclusive.

    Returns
    -------
    xr.DataArray
        `gcm`, bias-corrected, on its original (full) time axis.
    """
    y0, y1 = bias_correction_period

    # --- crop to common grid ---------------------------------------------
    obs, gcm = _crop_to_common_grid(obs, gcm)

    # --- sanity checks --------------------------------------------------
    if not (
        np.allclose(obs["latitude"].values, gcm["latitude"].values)
        and np.allclose(obs["longitude"].values, gcm["longitude"].values)
    ):
        raise ValueError(
            "obs and gcm must share the exact same (latitude, longitude) "
            "grid -- regrid one onto the other before bias-correcting."
        )

    # --- monthly climatologies over the reference period ----------------
    obs_ref = _select_years(obs, y0, y1, "obs")
    gcm_ref = _select_years(gcm, y0, y1, "gcm")

    obs_clim = obs_ref.groupby("time.month").mean("time")  # (month, lat, lon)
    gcm_clim = gcm_ref.groupby("time.month").mean("time")  # (month, lat, lon)

    # --- apply the fixed, per-month correction to the FULL gcm series ---
    gcm_anomaly = gcm.groupby("time.month") - gcm_clim
    gcm_corrected = gcm_anomaly.groupby("time.month") + obs_clim

    gcm_corrected = gcm_corrected.drop_vars("month", errors="ignore")
    # gcm_corrected.name = gcm.name
    gcm_corrected.attrs = dict(gcm.attrs)
    gcm_corrected.attrs["bias_correction"] = (
        f"additive anomaly method, reference period {y0}-{y1}"
    )
    return gcm_corrected


def bias_corrected(region, ssp, gcm, bias_correction_period=None):
    assert (
        bias_correction_period is None or len(bias_correction_period) == 2
    ), f"When provided bias_correction_period should have two elements but the provided value is {bias_correction_period}."

    if bias_correction_period is None:
        return regridding(region, ssp, gcm)
    else:
        if ssp != "historical":
            ds_temp = regridding(region, ssp, gcm)
            ds_temp_hist = regridding(region, "historical", gcm)
            gcm_full = combine_historical_and_projection(ds_temp_hist, ds_temp)
            del ds_temp, ds_temp_hist
        else:
            gcm_full = regridding(region, ssp, gcm)
        local_path = path_climate_data_ERA5(region)
        climate_data = local_path + "era5_monthly_averaged_data.nc"
        geopotential_data = local_path + "era5_geopotential_pressure.nc"
        ds_climate_era5 = _load_datasets(climate_data, geopotential_data, False)[0]
        t2m_corrected = bias_correct_temperature(
            obs=ds_climate_era5,
            gcm=gcm_full,
            bias_correction_period=bias_correction_period,
        )
        return t2m_corrected


# ---------------------------------------------------------------------------
# Surface fluxes and precipitation
# ---------------------------------------------------------------------------

SURFACE_VARS = ["slhf", "sshf", "ssrd", "str", "tp"]
# Precipitation and shortwave radiation are bias corrected with a scaling. The
# turbulent fluxes change sign within the year, which makes the scaling factor
# explode or flip the sign of the series, and a scaling of the net longwave
# radiation amplifies its summer values far beyond the ERA5 range, so these get an
# additive correction.
SCALING_BIAS_COR_VARS = ["ssrd", "tp"]
ADDITIVE_BIAS_COR_VARS = ["slhf", "sshf", "str"]

SECONDS_PER_DAY = 86400.0
WATER_DENSITY = 1000.0  # kg/m^3


def _load_era5(region):
    """Return the ERA5 monthly averaged climate dataset of a region."""
    local_path = path_climate_data_ERA5(region)
    climate_data = local_path + "era5_monthly_averaged_data.nc"
    geopotential_data = local_path + "era5_geopotential_pressure.nc"
    return _load_datasets(climate_data, geopotential_data, False)[0]


def load_cmip_surface_variables(region, ssp, gcm):
    """
    Load the CMIP6 surface fluxes and precipitation on the native GCM grid,
    renamed and converted to the ERA5 conventions:

    - names: slhf, sshf, ssrd, str, tp
    - sign: ECMWF convention, i.e. positive downwards (CMIP6 turbulent
      fluxes hfls/hfss are positive upwards, hence the sign flip)
    - units: ERA5 monthly averaged data stores daily accumulations, i.e.
      J m-2 (per day) for the fluxes and m (per day) for precipitation.
      CMIP6 provides rates in W m-2 and kg m-2 s-1.
    """
    files = [
        path_climate_data(region, ssp, gcm, var) + "data.nc"
        for var in SURFACE_FLUX_VARS
    ]
    ds = xr.merge(
        [
            xr.open_dataset(f).drop_vars(
                ["lat_bnds", "lon_bnds", "time_bnds"], errors="ignore"
            )
            for f in files
        ]
    )

    ds_out = xr.Dataset(
        {
            "tp": ds["pr"] * SECONDS_PER_DAY / WATER_DENSITY,
            "ssrd": ds["rsds"] * SECONDS_PER_DAY,
            "str": (ds["rlds"] - ds["rlus"]) * SECONDS_PER_DAY,
            "slhf": -ds["hfls"] * SECONDS_PER_DAY,
            "sshf": -ds["hfss"] * SECONDS_PER_DAY,
        }
    )
    for v in ["slhf", "sshf", "ssrd", "str"]:
        ds_out[v].attrs["units"] = "J m**-2"
    ds_out["tp"].attrs["units"] = "m"
    return ds_out.load()


def regridding_surface_variables(region, ssp, gcm, ds_era5=None):
    """Nearest-neighbour regridding of the CMIP6 surface fluxes and
    precipitation onto the ERA5 (latitude, longitude) grid."""
    ds_cmip = load_cmip_surface_variables(region, ssp, gcm)
    if ds_era5 is None:
        ds_era5 = _load_era5(region)

    lat_target = ds_era5[LAT_DIM_TARGET]
    lon_target_180 = ((ds_era5[LON_DIM_TARGET] + 180) % 360) - 180

    ds_regridded = ds_cmip.interp(
        {LAT_DIM_CMIP: lat_target, LON_DIM_CMIP: lon_target_180},
        method="nearest",
    ).drop_vars([LAT_DIM_CMIP, LON_DIM_CMIP], errors="ignore")

    # ERA5 points outside the range of the CMIP cell centres come back as NaN:
    # drop them, as is done for the temperature
    return ds_regridded.dropna(LAT_DIM_TARGET, how="all").dropna(
        LON_DIM_TARGET, how="all"
    )


def bias_correct_scaling(
    obs: xr.Dataset,
    gcm: xr.Dataset,
    bias_correction_period: tuple[int, int],
) -> xr.Dataset:
    """
    Bias-correct GCM variables against an observational dataset using the
    multiplicative scaling method, applied per grid cell and per calendar
    month. Both inputs are first cropped to their common footprint.

    hat_X(lat, lon, t) = obs_clim(lat, lon, month(t))
                         * gcm(lat, lon, t) / gcm_clim(lat, lon, month(t))

    where obs_clim / gcm_clim are monthly climatologies computed over
    `bias_correction_period` (inclusive years). The corrected series takes
    the units of `obs`.

    The method assumes that the monthly climatologies of obs and gcm have
    the same sign and are away from zero, which is why it is not used for
    the turbulent fluxes (slhf, sshf). A warning is printed for every
    variable where this does not hold.
    """
    y0, y1 = bias_correction_period

    obs, gcm = _crop_to_common_grid(obs[list(gcm.data_vars)], gcm)

    obs_clim = _select_years(obs, y0, y1, "obs").groupby("time.month").mean("time")
    gcm_clim = _select_years(gcm, y0, y1, "gcm").groupby("time.month").mean("time")

    factor = obs_clim / gcm_clim
    for v in factor.data_vars:
        # obs is NaN over the sea for ERA5-Land, these cells are not an issue
        has_obs = obs_clim[v].notnull()
        n_bad = int((has_obs & (~np.isfinite(factor[v]) | (factor[v] <= 0))).sum())
        if n_bad > 0:
            print(
                f"Warning: scaling factor of '{v}' is non-finite or non-positive "
                f"for {n_bad} / {int(has_obs.sum())} (month, latitude, longitude) "
                f"cells (obs and gcm climatologies of opposite sign or zero)."
            )

    gcm_corrected = (gcm.groupby("time.month") * factor).drop_vars(
        "month", errors="ignore"
    )
    for v in gcm_corrected.data_vars:
        gcm_corrected[v].attrs = dict(obs[v].attrs)
        gcm_corrected[v].attrs[
            "bias_correction"
        ] = f"multiplicative scaling method, reference period {y0}-{y1}"
    return gcm_corrected


def bias_corrected_surface_variables(region, ssp, gcm, bias_correction_period=None):
    """Surface fluxes and precipitation (slhf, sshf, ssrd, str, tp) of a
    CMIP6 run on the ERA5 grid, bias corrected when `bias_correction_period`
    is provided: scaling for SCALING_BIAS_COR_VARS and additive for
    ADDITIVE_BIAS_COR_VARS."""
    assert (
        bias_correction_period is None or len(bias_correction_period) == 2
    ), f"When provided bias_correction_period should have two elements but the provided value is {bias_correction_period}."

    ds_era5 = _load_era5(region)
    if bias_correction_period is None:
        return regridding_surface_variables(region, ssp, gcm, ds_era5)

    if ssp != "historical":
        gcm_full = combine_historical_and_projection(
            regridding_surface_variables(region, "historical", gcm, ds_era5),
            regridding_surface_variables(region, ssp, gcm, ds_era5),
        )
    else:
        gcm_full = regridding_surface_variables(region, ssp, gcm, ds_era5)

    scaled = bias_correct_scaling(
        obs=ds_era5,
        gcm=gcm_full[SCALING_BIAS_COR_VARS],
        bias_correction_period=bias_correction_period,
    )
    shifted = bias_correct_temperature(
        obs=ds_era5[ADDITIVE_BIAS_COR_VARS],
        gcm=gcm_full[ADDITIVE_BIAS_COR_VARS],
        bias_correction_period=bias_correction_period,
    )
    return xr.merge([scaled, shifted], join="inner", combine_attrs="drop")


def bias_corrected_climate(region, ssp, gcm, bias_correction_period=None):
    """All the CMIP6 climate features needed by the model (slhf, sshf, ssrd,
    str, t2m, tp) on the ERA5 grid, bias corrected over
    `bias_correction_period` when provided (additive for t2m, slhf, sshf and
    str, scaling for ssrd and tp)."""
    t2m = bias_corrected(region, ssp, gcm, bias_correction_period)
    if isinstance(t2m, xr.DataArray):
        t2m = t2m.to_dataset(name="t2m")
    surface = bias_corrected_surface_variables(region, ssp, gcm, bias_correction_period)
    return xr.merge([t2m[["t2m"]], surface], join="inner")
