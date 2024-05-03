import xarray as xr
import numpy as np
import pandas as pd
import math

# File directory and names
file_dir = '.././Data/'
file_name_in = 'Files/Iceland_Stake_Data_T_Attributes.csv'
file_name_out = 'Files/Iceland_Stake_Data_Climate.csv'

# Read stake data
df = pd.read_csv(file_dir + file_name_in, index_col=False)

# Open climate datasets
# From https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means?tab=overview
# From https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels-monthly-means?tab=overview
with xr.open_dataset(file_dir + 'Climate/ERA5_monthly_averaged_climate_data.nc') as ds_climate, \
        xr.open_dataset(file_dir + 'Climate/ERA5_monthly_averaged_pressure_data.nc') as ds_geopotential:

    # Convert geopotential height to geometric height and add to dataset
    r_earth = 6378137.0
    g = 9.80665
    ds_geopotential_metric = ds_geopotential.assign(
        altitude_climate=lambda ds_geo: r_earth * (ds_geopotential.z / g) / (r_earth - (ds_geopotential.z / g))
    )

    # Reduce expver dimension
    ds_climate = ds_climate.reduce(np.nansum, 'expver')

    # Get latitude and longitude
    lat = ds_climate.latitude
    lon = ds_climate.longitude

    # Create list of climate name variables and months combined
    climate_vars = list(ds_climate.keys())
    months_names = ['_oct', '_nov', '_dec', '_jan', '_feb', '_mar', '_apr', '_may', '_jun', '_jul', '_aug', '_sep']
    monthly_climate_vars = [f'{climate_var}{month_name:02}' for climate_var in climate_vars for month_name in
                            months_names]

    # Initialize arrays for the climate variable per data point and altitude
    climate_per_point = np.full((len(df), len(monthly_climate_vars)), np.nan)
    altitude_per_point = np.full((len(df), 1), np.nan)

    stake_lat = df.lat.round(2)
    stake_lon = df.lon.round(2)
    stake_date = pd.to_datetime(df['d3'], format="%d/%m/%Y", errors='coerce')
    stake_year = np.array([date.year for date in stake_date])

    # Iterate through stake data, and get the climate variables and altitude for this point
    for idx, (lat, lon, year) in enumerate(zip(stake_lat, stake_lon, stake_year)):

        # Some years are float NaNs, these cannot be processed and therefore will be skipped
        if math.isnan(year):
            continue

        range_date = pd.date_range(start=str(int(year) - 1) + '-09-01',
                                   end=str(int(year)) + '-09-01', freq='ME')

        # Select climate data for the point, or the nearest point to it
        climate_data_point = ds_climate.sel(latitude=lat, longitude=lon, time=range_date, method='nearest')

        # Convert selected data to Dataframe and save it
        if climate_data_point.dims:
            climate_points = climate_data_point.to_dataframe().drop(columns=['latitude', 'longitude'])
            climate_per_point[idx, :] = climate_points.to_numpy().flatten(order='F')

            # Select altitude data
            altitude_point = ds_geopotential_metric.sel(latitude=lat, longitude=lon, method='nearest')
            altitude_per_point[idx] = altitude_point.altitude_climate.values[0][0]

    # Create DataFrames from arrays
    df_climate = pd.DataFrame(data=climate_per_point, columns=monthly_climate_vars)
    df_altitude = pd.DataFrame(data=altitude_per_point, columns=['altitude_climate'])

    # Concatenate DataFrames
    df_point_climate = pd.concat([df, df_climate, df_altitude], axis=1)

    # Write to CSV
    df_point_climate.to_csv(file_dir + file_name_out, index=False)
