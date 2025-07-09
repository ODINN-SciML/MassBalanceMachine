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
import xarray as xr
import numpy as np
import pandas as pd


def get_climate_features(
    df: pd.DataFrame,
    output_fname: str,
    climate_data: str,
    geopotential_data: str,
    change_units: bool,
    vois_climate: list = None,
    vois_other: list = None,
) -> pd.DataFrame:
    """
    Takes as input ERA5-Land monthly averaged climate data (pre-downloaded), and matches this with the locations
    of the stake measurements.

    Args:
        df (pd.DataFrame): DataFrame containing stake measurement locations and years.
        output_fname (str): Path to the output CSV file.
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
        raise FileNotFoundError(
            f"Climate data file {climate_data} does not exist.")
    if not os.path.exists(geopotential_data):
        raise FileNotFoundError(
            f"Geopotential data file {geopotential_data} does not exist.")

    # Load the two climate datasets
    ds_climate, ds_geopotential = _load_datasets(climate_data,
                                                 geopotential_data)

    # Makes things easier down the line
    # Change temperature to Celsius and precipitation to m.w.e
    if change_units:
        ds_climate["t2m"] = ds_climate["t2m"] - 273.15

    # Get latitudes and longitudes from the climate dataset.
    lat, lon = ds_climate.latitude, ds_climate.longitude

    # Convert the longitudes
    ds_180 = _adjust_longitude(ds_geopotential)

    # Crop the geopotential height to the region of interest
    ds_geopotential_cropped = _crop_geopotential(ds_180, lat, lon)

    # Remove duplicates
    ds_geopotential_cropped = ds_geopotential_cropped.drop_duplicates(
        dim="latitude")
    ds_geopotential_cropped = ds_geopotential_cropped.drop_duplicates(
        dim="longitude")

    # Calculate the geopotential height in meters
    ds_geopotential_metric = _calculate_geopotential_height(
        ds_geopotential_cropped)

    if 'expver' in ds_climate.dims:
        # Reduce expver dimension
        ds_climate = ds_climate.reduce(np.nansum, "expver")

    # Create a date range for one hydrological year
    df = _add_date_range(df)

    # Get the climate data for the latitudes and longitudes and date ranges as
    # specified
    climate_df = _process_climate_data(ds_climate, df)

    # Get the geopotential height for the latitudes and longitudes as specified,
    # for the locations of the stake measurements.
    altitude_df = _process_altitude_data(ds_geopotential_metric, df)

    # Combine the climate data with the altitude climate data
    df = _combine_dataframes(df, climate_df, altitude_df)

    # Remove climate artifacts
    df = smooth_era5land_by_mode(df, vois_climate, vois_other)

    # Add a new feature to the dataframe that is the height difference between the elevation
    # of the stake and the recorded height of the climate.
    df = _calculate_elevation_difference(df)

    df.to_csv(output_fname, index=False)

    return df


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
    climate_df = (xr_data_points.to_dataframe().drop(
        columns=["lat", "lon", "x", "y"]).reset_index())
    climate_df = climate_df.drop(columns=["points"])

    # The normal way does not work if there is only one point per glacier
    if len(xr_data_points.points) == 1:
        # Initialize reshaped_ as a list
        reshaped_ = []

        for month in range(0, 12):
            month_ = climate_df[climate_df.time == month].drop(
                columns=['time'])

            values = month_.values
            # values.shape is (n_points, n_vars) or (1, n_vars) if one point

            reshaped_.append(values)

        # Now stack along axis=0 to get shape (12, n_points) if multiple points,
        # or (12, 1) if only one point, and then transpose to get (n_points, 12)
        reshaped_array = np.vstack(reshaped_).T

        result_df = pd.DataFrame(reshaped_array,
                                 columns=[f'Month_{i+1}' for i in range(12)])
    elif len(xr_data_points.points) > 1:
        reshaped_ = []
        for month in range(0, 12):
            month_ = climate_df[climate_df.time == month].drop(
                columns=['time'])
            reshaped_.append(month_.values.squeeze())

        result_df = pd.DataFrame(np.array(reshaped_).transpose(),
                                 columns=[f'Month_{i+1}' for i in range(12)])
    else:   
        raise ValueError("No points found in the radiation dataset.")

    # Set the new column names for the dataframe (normal year not hydrological)
    climate_var = 'pcsr'
    months_names = [f"_{month.lower()}" for month in month_abbr[1:]]
    result_df.columns = [
        f"{climate_var}{month_name}" for month_name in months_names
    ]
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
            col for col in df.columns if any(
                col.startswith(vo + '_') for vo in vois_climate)
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


def _load_datasets(climate_data: str, geopotential_data: str) -> tuple:
    """Load climate and geopotential datasets."""
    with xr.open_dataset(climate_data) as dataset_climate, xr.open_dataset(
            geopotential_data) as dataset_geopotential:
        return dataset_climate.load(), dataset_geopotential.load()


def _calculate_geopotential_height(ds_geopotential: xr.Dataset) -> xr.Dataset:
    """Calculate geopotential height in meters."""
    r_earth = 6367.47 * 10e3  # [m] (Grib1 radius)
    g = 9.80665  # [m/s^2]
    return ds_geopotential.assign(altitude_climate=lambda ds_geo: r_earth *
                                  (ds_geo.z / g) / (r_earth - (ds_geo.z / g)))


def _adjust_longitude(ds: xr.Dataset) -> xr.Dataset:
    """Adjust longitude coordinates to range from -180 to 180."""
    return ds.assign_coords(longitude=(((ds.longitude + 180) % 360) -
                                       180)).sortby("longitude")


def _crop_geopotential(ds: xr.Dataset, lat: xr.DataArray,
                       lon: xr.DataArray) -> xr.Dataset:
    """Crop geometric height to grid of climate data."""
    return ds.sel(longitude=lon, latitude=lat, method="nearest")


def _generate_climate_variable_names(ds_climate: xr.Dataset) -> list:
    """Generate list of climate variable names for one hydrological year."""
    climate_variables = list(ds_climate.keys())
    months_names = [
        f"_{month.lower()}" for month in month_abbr[10:] + month_abbr[1:10]
    ]
    return [
        f"{climate_var}{month_name}" for climate_var in climate_variables
        for month_name in months_names
    ]


def _create_date_range(year: int):
    """Create a date range for a given year."""
    if pd.isna(year):
        return None
    year = int(year)
    return pd.date_range(start=f"{year - 1}-09-01",
                         end=f"{year}-09-01",
                         freq="ME")


def _add_date_range(df: pd.DataFrame) -> pd.DataFrame:
    """Add date range to DataFrame."""
    df["range_date"] = df["YEAR"].apply(_create_date_range)
    return df


def _process_climate_data(ds_climate: xr.Dataset,
                          df: pd.DataFrame) -> pd.DataFrame:
    """Process climate data for all points and times."""

    # Create DataArrays for latitude and longitude
    lat_da = xr.DataArray(df["POINT_LAT"].values, dims="points")
    lon_da = xr.DataArray(df["POINT_LON"].values, dims="points")

    # Create a 2D array of dates ranges
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

    # Create a dataframe from the DataArray
    climate_df = (climate_data_points.to_dataframe().drop(
        columns=dropColumns).reset_index())

    # Drop columns
    climate_df = climate_df.drop(columns=["points", "time"])

    # Get the number of rows and columns
    num_rows, num_cols = climate_df.shape

    # Reshape the DataFrame to a 3D array (groups, 12, columns)
    reshaped_array = climate_df.to_numpy().reshape(-1, 12, num_cols)

    # Transpose and reshape to get the desired flattening effect
    result_array = reshaped_array.transpose(0, 2, 1).reshape(-1, 12 * num_cols)

    # Convert back to a DataFrame if needed
    result_df = pd.DataFrame(result_array)
    # Set the new column names for the dataframe (climate variables X months
    # of the hydrological year)
    result_df.columns = _generate_climate_variable_names(ds_climate)
    return result_df


def _process_altitude_data(ds_geopotential: xr.Dataset,
                           df: pd.DataFrame) -> pd.DataFrame:
    """Process altitude data for all points."""

    # 1. Create DataArrays for latitude and longitude
    lat_da = xr.DataArray(df["POINT_LAT"].values, dims="points")
    lon_da = xr.DataArray(df["POINT_LON"].values, dims="points")

    if 'valid_time' in ds_geopotential.dims:
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


def _combine_dataframes(df: pd.DataFrame, climate_df: pd.DataFrame,
                        altitude_df: pd.DataFrame) -> pd.DataFrame:
    """Combine DataFrames and add altitude data."""
    df = df.drop(columns=["range_date"]).reset_index(drop=True)
    climate_df = climate_df.reset_index(drop=True)
    altitude_df = altitude_df.drop(
        columns=["latitude", "longitude", "z"]).reset_index(drop=True)

    df = pd.concat([df, climate_df, altitude_df], axis=1)
    #df["ALTITUDE_CLIMATE"] = altitude_df.altitude_climate.values
    # rename column
    df.rename(columns={"altitude_climate": "ALTITUDE_CLIMATE"}, inplace=True)
    df.dropna(subset=["ALTITUDE_CLIMATE"], inplace=True)

    return df


def _calculate_elevation_difference(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the difference between geopotential height and stake measurement elevation."""
    df["ELEVATION_DIFFERENCE"] = df["POINT_ELEVATION"] - df["ALTITUDE_CLIMATE"]
    return df
