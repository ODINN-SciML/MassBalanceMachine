"""
This code is taken, and refactored, and inspired from the work performed by: Kamilla Hauknes Sjursen

This method fetches the meteorological features (variables of interest), for each stake measurement available,
via the provided NETCDF-3 files fetched from the ERA5Land Reanalysis (monthly averaged) database.

Depending on the amount of variables, and the temporal scale, downloads of the climate data can take up hours.
Climate data can either be downloaded manually via the link provided in the notebook.

Date Created: 21/07/2024
"""

import os
from calendar import month_abbr
from functools import lru_cache
from typing import Optional
import xarray as xr
import numpy as np
import pandas as pd

import config
from data_processing.climate_data_download import path_climate_data
from data_processing.utils.hydro_year import months_hydro_year, _rebuild_month_index


def get_climate_features_(
    df: pd.DataFrame,
    output_fname: str,
    climate_data: str,
    geopotential_data: str,
    change_units: bool,
    months_tail_pad,  # before 'oct'
    months_head_pad,  # after 'sep'
    vois_climate: list = None,
    vois_other: list = None,
    monthly: bool = False,
) -> pd.DataFrame:
    """
    Takes as input ERA5-Land monthly averaged climate data (pre-downloaded), and matches this with the locations
    of the stake measurements.

    Args:
        df (pd.DataFrame): DataFrame containing stake measurement locations and years.
        output_fname (str): Path to the output CSV file. If set to `None`, it is not saved.
        climate_data (str): Path to the ERA5-Land climate data file.
        geopotential_data (str): Path to the geopotential data file.
        change_units (bool): If True, change temperature to Celsius and precipitation to m.w.e.
        vois_climate (list): List of climate variables of interest to smooth out their climate artificats .
        vois_other (list): List of other variables of interest to smooth out their climate artificats (typically ALTITUDE_CLIMATE).

    Returns:
        pd.DataFrame: The updated DataFrame with climate and altitude features.
    """

    # Check if the input files exist.
    if not os.path.exists(climate_data):
        raise FileNotFoundError(f"Climate data file {climate_data} does not exist.")
    if not os.path.exists(geopotential_data):
        raise FileNotFoundError(
            f"Geopotential data file {geopotential_data} does not exist."
        )

    # Load the two climate datasets. Shared between calls, hence read-only here.
    ds_climate, ds_geopotential = _load_datasets(
        climate_data, geopotential_data, change_units
    )

    # Get latitudes and longitudes from the climate dataset.
    lat, lon = ds_climate.latitude, ds_climate.longitude

    # Convert the longitudes
    ds_180 = _adjust_longitude(ds_geopotential)

    # Crop the geopotential height to the region of interest
    ds_geopotential_cropped = _crop_geopotential(ds_180, lat, lon)

    # Remove duplicates
    ds_geopotential_cropped = ds_geopotential_cropped.drop_duplicates(dim="latitude")
    ds_geopotential_cropped = ds_geopotential_cropped.drop_duplicates(dim="longitude")

    # Calculate the geopotential height in meters
    ds_geopotential_metric = _calculate_geopotential_height(ds_geopotential_cropped)

    if monthly:
        return _get_monthly_climate_features(
            df,
            ds_climate,
            ds_geopotential_metric,
            vois_climate=vois_climate,
            vois_other=vois_other,
            output_fname=output_fname,
        )

    # Create a date range for one hydrological year
    df = _add_date_range(df, months_tail_pad, months_head_pad)

    # Get the climate data for the latitudes and longitudes and date ranges as
    # specified
    df["months_in_range"] = df["range_date"].apply(
        lambda rng: [d.strftime("%b").lower() for d in rng] if rng is not None else []
    )

    climate_df = _process_climate_data(ds_climate, df, months_tail_pad, months_head_pad)

    # Get the geopotential height for the latitudes and longitudes as specified,
    # for the locations of the stake measurements.
    altitude_df = _process_altitude_data(ds_geopotential_metric, df)

    # Combine the climate data with the altitude climate data
    df = _combine_dataframes(df, climate_df, altitude_df)

    # Compute the sum of the fluxes per month from the average fluxes per day
    # Cf https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5+data+documentation#ERA5:datadocumentation-Meanrates/fluxesandaccumulations
    fluxes_cols = ["tp", "slhf", "str", "sshf", "ssrd"]
    df_cols = df.columns.values
    month_to_id = {(month_abbr[i].lower()): i for i in range(1, 13)}
    new_cols = {}
    for col_df in df_cols:
        m = [col_df.startswith(c) for c in fluxes_cols]
        if any(m):
            assert np.sum(m) == 1
            flux_col = np.array(fluxes_cols)[np.array(m)][0]
            suffix = col_df.replace(flux_col + "_", "")
            id_month = str(month_to_id[suffix.replace("_", "")])
            # Incorrect because we retrieve the year of the hydrological year and not the true year associated to the measurement
            days_in_month = pd.to_datetime(
                df.YEAR.astype(str) + "-" + id_month + "-01"
            ).dt.days_in_month
            sum_col = flux_col + "_sum_" + suffix
            new_cols[sum_col] = df[col_df].values * days_in_month
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Remove climate artifacts
    df = smooth_era5land_by_mode(df, vois_climate, vois_other)

    # Add a new feature to the dataframe that is the height difference between the elevation
    # of the stake and the recorded height of the climate.
    df = _calculate_elevation_difference(df)

    if output_fname is not None:
        df.to_csv(output_fname, index=False)

    return df


def _get_monthly_climate_features(
    df,
    ds_climate,
    ds_geopotential,
    vois_climate,
    vois_other,
    output_fname,
):
    """Append ERA5 features selected at each row's monthly ``Date``."""
    if "Date" not in df:
        raise ValueError("Monthly climate extraction requires a Date column.")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["range_date"] = df["Date"].map(lambda date: pd.DatetimeIndex([date]))
    climate_df = _process_climate_data(
        ds_climate,
        df,
        months_tail_pad=[],
        months_head_pad=[],
        monthly=True,
        vois_climate=vois_climate,
    )

    # Compute the sum of the fluxes per month from the average fluxes per day
    # Cf https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5+data+documentation#ERA5:datadocumentation-Meanrates/fluxesandaccumulations
    fluxes_cols = ["tp", "slhf", "str", "sshf", "ssrd"]
    days_in_month = df["Date"].dt.days_in_month.to_numpy()
    for variable in fluxes_cols:
        if variable in climate_df and f"{variable}_sum" in (vois_climate or []):
            climate_df[f"{variable}_sum"] = climate_df[variable] * days_in_month

    altitude_df = _process_altitude_data(ds_geopotential, df)
    result = _combine_dataframes(df, climate_df, altitude_df)
    result = _calculate_elevation_difference(result)
    # Note: smooth_era5land_by_mode is NOT applied here. It removes ERA5-Land
    # grid-cell artifacts by collapsing a column to its single most frequent
    # value across all rows, which is appropriate when rows are multiple
    # spatial points sharing one hydrological year. Here rows are consecutive
    # months at a single, fixed station, so that would wipe out the real
    # month-to-month climate signal instead of removing noise.
    if output_fname is not None:
        result.to_csv(output_fname, index=False)
    return result


def get_first_last_month(df):
    df["FROM_DATE"] = pd.to_datetime(df["FROM_DATE"].astype(str), format="%Y%m%d")
    df["TO_DATE"] = pd.to_datetime(df["TO_DATE"].astype(str), format="%Y%m%d")

    # Extract first and last months as numbers (1–12)
    df["FIRST_MONTH_NUM"] = df[["FROM_DATE"]].min(axis=1).dt.month
    df["LAST_MONTH_NUM"] = df[["TO_DATE"]].max(axis=1).dt.month

    # Compute min of all FIRST and max of all LAST
    global_first = df["FIRST_MONTH_NUM"].min()
    global_last = df["LAST_MONTH_NUM"].max()

    # Convert back to abbreviations if needed
    month_abbr = {
        i: pd.to_datetime(str(i), format="%m").strftime("%b").lower()
        for i in range(1, 13)
    }
    global_first_abbr = month_abbr[global_first]
    global_last_abbr = month_abbr[global_last]
    return global_first_abbr, global_last_abbr


def retrieve_clear_sky_rad(df, path_to_file):
    """Takes as input monthly clear sky potential radiation data (pre-processed), and matches this with the locations
    of the stake measurements for a specific glacier (each glacier has another pot. rad file).

    Args:
        df (pd.DataFrame): DataFrame containing stake measurement locations and years for a glacier.
        path_to_file (str): path to radiation data for a specific glacier

    Returns:
        df (pd.DataFrame): same dataframe as input but with 12 additional columns corresponding to the monthly
        potential radiation values.
    """
    # load pot dataset:
    radiation_xr = xr.open_dataset(path_to_file)

    # Create DataArrays for latitude and longitude
    lat_da = xr.DataArray(df["POINT_LAT"].values, dims="points")
    lon_da = xr.DataArray(df["POINT_LON"].values, dims="points")

    # Find closest radiation points (12 months)
    xr_data_points = radiation_xr.sel(
        lat=lat_da,
        lon=lon_da,
        method="nearest",
    )
    climate_df = (
        xr_data_points.to_dataframe()
        .drop(columns=["lat", "lon", "x", "y"])
        .reset_index()
    )
    climate_df = climate_df.drop(columns=["points"])

    # The normal way does not work if there is only one point per glacier
    if len(xr_data_points.points) == 1:
        # Initialize reshaped_ as a list
        reshaped_ = []

        for month in range(0, 12):
            month_ = climate_df[climate_df.time == month].drop(columns=["time"])

            values = month_.values
            # values.shape is (n_points, n_vars) or (1, n_vars) if one point

            reshaped_.append(values)

        # Now stack along axis=0 to get shape (12, n_points) if multiple points,
        # or (12, 1) if only one point, and then transpose to get (n_points, 12)
        reshaped_array = np.vstack(reshaped_).T

        result_df = pd.DataFrame(
            reshaped_array, columns=[f"Month_{i+1}" for i in range(12)]
        )
    elif len(xr_data_points.points) > 1:
        reshaped_ = []
        for month in range(0, 12):
            month_ = climate_df[climate_df.time == month].drop(columns=["time"])
            reshaped_.append(month_.values.squeeze())

        result_df = pd.DataFrame(
            np.array(reshaped_).transpose(), columns=[f"Month_{i+1}" for i in range(12)]
        )
    else:
        raise ValueError("No points found in the radiation dataset.")

    # Set the new column names for the dataframe (normal year not hydrological)
    climate_var = "pcsr"
    months_names = [f"_{month.lower()}" for month in month_abbr[1:]]
    result_df.columns = [f"{climate_var}{month_name}" for month_name in months_names]
    # Concatenate to glacier data
    result_df = result_df.reset_index(drop=True)
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, result_df], axis=1)
    return df


def smooth_era5land_by_mode(df, vois_climate=None, vois_other=None):
    """For big glaciers covered by more than one ERA5-Land grid cell, the
    climate data is the one with the most data points. This function smooths the
    climate data by taking the mode of the data for each grid cell.

    Args:
        vois_climate (str): A string containing the climate variables of interest
        df (pd.DataFrame): DataFrame containing the stakes data.
    """
    if vois_climate is not None:
        # Filter out the climate variable columns based on vois_climate
        climate_cols = [
            col
            for col in df.columns
            if any(col.startswith(vo + "_") for vo in vois_climate)
        ]

        # Replace each climate column with its mode value
        for col in climate_cols:
            mode_val = df[col].mode().iloc[0]  # Most frequent value
            df[col] = mode_val
    if vois_other is not None:
        # also smooth the other variables
        for col in vois_other:
            mode_val = df[col].mode().iloc[0]
            df[col] = mode_val

    return df


def _file_stamp(path: str) -> tuple:
    """Cheap identity of a file, so that replacing it invalidates the cache below."""
    stat = os.stat(path)
    return (stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=2)
def _load_datasets_cached(
    climate_data: str, geopotential_data: str, change_units: bool, stamps: tuple
) -> tuple:
    """Load the ERA5 files and apply the preparation that depends only on them.

    Kept behind a cache because the whole regional climate file is read into
    memory - close to a gigabyte for an Alpine region - and the gridded feature
    generation asks for it once per glacier *and per year*. Reloading it every time
    dominated the cost of a task and made memory use sawtooth, since a worker
    allocated a fresh copy before the previous one was collected.

    `stamps` is not used: it is part of the cache key so that a climate file
    replaced on disk is reloaded instead of served from a stale entry.
    """
    del stamps  # only a cache key
    with (
        xr.open_dataset(climate_data) as dataset_climate,
        xr.open_dataset(geopotential_data) as dataset_geopotential,
    ):
        ds_climate = dataset_climate.load()
        ds_geopotential = dataset_geopotential.load()

    # Makes things easier down the line
    # Change temperature to Celsius and precipitation to m.w.e
    if change_units:
        # Not an in-place assignment: the datasets are shared between calls.
        ds_climate = ds_climate.assign(t2m=ds_climate["t2m"] - 273.15)

    if "expver" in ds_climate.dims:
        # Reduce expver dimension
        ds_climate = ds_climate.reduce(np.nansum, "expver")

    return ds_climate, ds_geopotential


def _load_datasets(
    climate_data: str, geopotential_data: str, change_units: bool = False
) -> tuple:
    """Load climate and geopotential datasets.

    The returned datasets are shared with every other caller asking for the same
    files, so they must be treated as read-only: derive new datasets with
    `assign`, `sel` or `reduce` rather than assigning into them.
    """
    stamps = (_file_stamp(climate_data), _file_stamp(geopotential_data))
    return _load_datasets_cached(climate_data, geopotential_data, change_units, stamps)


def climate_file_paths(region_id) -> tuple:
    """The ERA5 climate and geopotential files of a region."""
    local_path = path_climate_data(region_id)
    return (
        local_path + "era5_monthly_averaged_data.nc",
        local_path + "era5_geopotential_pressure.nc",
    )


def climate_memory_footprint(region_id) -> Optional[int]:
    """Bytes the ERA5 data of a region occupies once loaded, or None if it is not
    downloaded yet.

    Read from the file headers, without loading anything: `nbytes` of a lazily
    opened dataset is computed from the variable shapes and dtypes. This is what a
    worker generating gridded features holds for as long as it lives, and therefore
    the term that decides how many of them fit in memory.
    """
    climate_data, geopotential_data = climate_file_paths(region_id)
    if not (os.path.isfile(climate_data) and os.path.isfile(geopotential_data)):
        return None
    with (
        xr.open_dataset(climate_data) as ds_climate,
        xr.open_dataset(geopotential_data) as ds_geopotential,
    ):
        return ds_climate.nbytes + ds_geopotential.nbytes


def warm_climate_cache(region_id, change_units: bool = True) -> None:
    """Load the climate data of a region into the cache of the current process.

    Called before a pool of workers is forked: the workers inherit the loaded
    arrays instead of each reading the files themselves.
    """
    climate_data, geopotential_data = climate_file_paths(region_id)
    if os.path.isfile(climate_data) and os.path.isfile(geopotential_data):
        _load_datasets(climate_data, geopotential_data, change_units)


def _calculate_geopotential_height(ds_geopotential: xr.Dataset) -> xr.Dataset:
    """Calculate geopotential height in meters."""
    r_earth = 6367.47 * 10e3  # [m] (Grib1 radius)
    g = 9.80665  # [m/s^2]
    return ds_geopotential.assign(
        altitude_climate=lambda ds_geo: r_earth
        * (ds_geo.z / g)
        / (r_earth - (ds_geo.z / g))
    )


def _adjust_longitude(ds: xr.Dataset) -> xr.Dataset:
    """Adjust longitude coordinates to range from -180 to 180."""
    return ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby(
        "longitude"
    )


def _crop_geopotential(
    ds: xr.Dataset, lat: xr.DataArray, lon: xr.DataArray
) -> xr.Dataset:
    """Crop geometric height to grid of climate data."""
    return ds.sel(longitude=lon, latitude=lat, method="nearest")


def _generate_climate_variable_names(
    ds_climate: xr.Dataset,
    months_tail_pad,
    months_head_pad,
) -> list:
    """Generate list of climate variable names for one hydrological year."""
    climate_variables = list(ds_climate.keys())
    months_names = [f"_{month}" for month in months_hydro_year]

    # extend months on both sides for longer periods:
    months_names = (
        [f"_{month.lower()}" for month in months_tail_pad]
        + months_names
        + [f"_{month.lower()}" for month in months_head_pad]
    )
    return [
        f"{climate_var}{month_name}"
        for climate_var in climate_variables
        for month_name in months_names
    ]


def _create_date_range(
    year: int,
    start_month: str,
    end_month: str,
) -> pd.DatetimeIndex:
    """Create a date range for a given hydrological year based on start_month and end_month."""
    if pd.isna(year):
        return None
    year = int(year)
    # Period from 2000-10-01 to 2001-09-30 corresponds to the 2001 hydrological year

    # start month is always in the PREVIOUS year
    start = f"{year - 1}-{start_month}-01"
    # end month can be either the CURRENT year, or the NEXT one depending on the head padding
    if int(end_month) <= 6:
        # If end month is before June this is likely that the measurement associated with hydrological year Y spans after (Y-1)-12-31
        end = f"{year + 1}-{end_month}-01"
    else:
        end = f"{year}-{end_month}-01"

    return pd.date_range(start=start, end=end, freq="MS")


def _create_month_range(months_tail_pad, months_head_pad):
    """Create a month range based on months_tail_pad and months_head_pad."""

    # mapping 'jan' -> '01', ..., 'dec' -> '12'
    abbr_to_num = {month_abbr[i].lower(): f"{i:02d}" for i in range(1, 13)}

    def token_to_num(token: str) -> str:
        clean = token.strip("_")  # remove leading/trailing underscores
        if clean in abbr_to_num:
            return abbr_to_num[clean]
        raise ValueError(f"Unknown month token: {token}")

    month_list, _ = _rebuild_month_index(months_head_pad, months_tail_pad)
    start_token, end_token = month_list[0], month_list[-1]

    start_month = token_to_num(start_token)
    end_month = token_to_num(end_token)

    return start_month, end_month


def _add_date_range(df: pd.DataFrame, months_tail_pad, months_head_pad) -> pd.DataFrame:
    start_month, end_month = _create_month_range(months_tail_pad, months_head_pad)

    return df.assign(
        range_date=df["YEAR"].map(
            lambda y: _create_date_range(y, start_month, end_month)
        )
    )


def _process_climate_data(
    ds_climate: xr.Dataset,
    df: pd.DataFrame,
    months_tail_pad,
    months_head_pad,
    monthly: bool = False,
    vois_climate: list = None,
) -> pd.DataFrame:
    """Process climate data for all points and times.

    Raises
    ------
    ValueError
        If any POINT_LAT / POINT_LON fall outside the spatial bounds of ds_climate.
    """

    # --- Bounds check (spatial) ---
    if "latitude" not in ds_climate.coords or "longitude" not in ds_climate.coords:
        raise ValueError("ds_climate must have 'latitude' and 'longitude' coordinates.")

    # Check that the time window of all the entries is included in the range of the climate data
    start_climate = ds_climate.time.min().values
    end_climate = ds_climate.time.max().values
    min_start_df = pd.to_datetime(df["Date"] if monthly else df.FROM_DATE).min()
    max_end_df = pd.to_datetime(
        df["Date"] if monthly else df.TO_DATE
    ).max() + pd.tseries.offsets.MonthBegin(0)
    assert (
        min_start_df >= start_climate
    ), f"The measurement periods start outside of the climate time range. Climate data start on {start_climate} but measurements start up to {min_start_df}."
    assert (
        max_end_df <= end_climate
    ), f"The measurement periods end outside of the climate time range. Climate data end on {end_climate} but measurements end up to {max_end_df}."

    lats = ds_climate["latitude"].values
    lons = ds_climate["longitude"].values

    lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
    lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))

    pts_lat = df["POINT_LAT"].to_numpy(dtype=float)
    pts_lon = df["POINT_LON"].to_numpy(dtype=float)

    # Handle 0..360 longitudes (common in ERA5)
    ds_uses_0360 = lon_max > 180.0
    if ds_uses_0360:
        pts_lon_chk = np.mod(pts_lon, 360.0)
    else:
        pts_lon_chk = pts_lon

    # Check bounds
    bad_lat = (pts_lat < lat_min) | (pts_lat > lat_max)
    bad_lon = (pts_lon_chk < lon_min) | (pts_lon_chk > lon_max)
    bad = bad_lat | bad_lon

    if bad.any():
        bad_rows = df.loc[bad, ["POINT_ID", "GLACIER", "POINT_LAT", "POINT_LON"]].copy()
        # add also checked lon (after 0..360 conversion) for debugging
        bad_rows["POINT_LON_CHECKED"] = pts_lon_chk[bad]
        msg = (
            "Some points fall outside the spatial bounds of ds_climate.\n"
            f"ds_climate latitude bounds: [{lat_min}, {lat_max}]\n"
            f"ds_climate longitude bounds: [{lon_min}, {lon_max}] "
            f"({'0..360' if ds_uses_0360 else '-180..180'})\n"
            f"Number of out-of-bounds points: {int(bad.sum())}\n"
            "First few offending rows:\n"
            f"{bad_rows.head(10).to_string(index=False)}"
        )
        raise ValueError(msg)

    # --- Create DataArrays for selection ---
    lat_da = xr.DataArray(pts_lat, dims="points")
    lon_da = xr.DataArray(pts_lon_chk if ds_uses_0360 else pts_lon, dims="points")

    if monthly:
        climate_variables = [
            variable
            for variable in (vois_climate or [])
            if not variable.endswith("_sum") and variable in ds_climate
        ]
        missing_variables = [
            variable
            for variable in (vois_climate or [])
            if not variable.endswith("_sum") and variable not in ds_climate
        ]
        if missing_variables:
            raise ValueError(
                f"Climate variables not found in dataset: {missing_variables}"
            )
        ds_climate = ds_climate[climate_variables]
        time_da = xr.DataArray(df["Date"].to_numpy(), dims="points")
    else:
        date_array = np.array([r.values for r in df["range_date"].values])
        time_da = xr.DataArray(date_array, dims=["points", "time"])

    climate_data_points = ds_climate.sel(
        latitude=lat_da,
        longitude=lon_da,
        time=time_da,
        method="nearest",
    )

    # Handle new netcdf format where number and expver are coordinates
    dropColumns = ["latitude", "longitude"]
    if "number" in climate_data_points.coords:
        dropColumns.append("number")
    if "expver" in climate_data_points.coords:
        dropColumns.append("expver")

    climate_df = (
        climate_data_points.to_dataframe().drop(columns=dropColumns).reset_index()
    )

    climate_df = climate_df.drop(
        columns=[column for column in ["points", "time"] if column in climate_df]
    )

    if monthly:
        return climate_df

    num_rows, num_cols = climate_df.shape
    N_MONTHS = date_array.shape[1]

    reshaped_array = climate_df.to_numpy().reshape(-1, N_MONTHS, num_cols)
    result_array = reshaped_array.transpose(0, 2, 1).reshape(-1, N_MONTHS * num_cols)

    result_df = pd.DataFrame(result_array)
    result_df.columns = _generate_climate_variable_names(
        ds_climate, months_tail_pad, months_head_pad
    )
    return result_df


def _process_altitude_data(
    ds_geopotential: xr.Dataset, df: pd.DataFrame
) -> pd.DataFrame:
    """Process altitude data for all points."""

    # 1. Create DataArrays for latitude and longitude
    lat_da = xr.DataArray(df["POINT_LAT"].values, dims="points")
    lon_da = xr.DataArray(df["POINT_LON"].values, dims="points")

    if "valid_time" in ds_geopotential.dims:
        # Handle new netcdf format
        ds_renamed = ds_geopotential.rename({"valid_time": "time"})
    else:
        ds_renamed = ds_geopotential

    altitude_data_points = ds_renamed.sel(
        latitude=lat_da,
        longitude=lon_da,
        method="nearest",
    )

    return altitude_data_points.to_dataframe()


def _combine_dataframes(
    df: pd.DataFrame, climate_df: pd.DataFrame, altitude_df: pd.DataFrame
) -> pd.DataFrame:
    """Combine DataFrames and add altitude data."""
    df = df.drop(columns=["range_date"], errors="ignore").reset_index(drop=True)
    climate_df = climate_df.reset_index(drop=True)
    altitude_df = altitude_df.drop(columns=["latitude", "longitude", "z"]).reset_index(
        drop=True
    )

    df = pd.concat([df, climate_df, altitude_df], axis=1)
    # df["ALTITUDE_CLIMATE"] = altitude_df.altitude_climate.values
    # rename column
    df.rename(columns={"altitude_climate": "ALTITUDE_CLIMATE"}, inplace=True)
    df.dropna(subset=["ALTITUDE_CLIMATE"], inplace=True)

    return df


def _calculate_elevation_difference(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the difference between geopotential height and stake measurement elevation."""
    return df.assign(
        ELEVATION_DIFFERENCE=df["POINT_ELEVATION"] - df["ALTITUDE_CLIMATE"]
    )
