"""
This code is inspired by the work of Kamilla Hauknes Sjursen

This script takes as input ERA5-Land monthly averaged climate data (pre-downloaded), and matches this with the locations
of the stake measurements. The climate features are training features for the machine-learning model. Important is that 
the climate data is already downloaded and saved in location: .././data/climate.

Depending on the amount of variables, and the temporal scale, downloads of the climate data can take up hours. 
Climate data can either be downloaded manually via the link below, or obtained via the script: 
get_ERA5_monthly_averaged_climate_data.py. This file should be first unzipped before running this script.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 04/06/2024
"""

import xarray as xr
import numpy as np
import pandas as pd
import math
import os

from dateutil import parser


# TODO: Put these URLs listed in this method in the documentation
def get_climate_features(df, output_fname, climate_data, geopotential_data, column_name_year):
    # Check if the climate input file exists
    if not os.path.exists(climate_data) and not os.path.exists(geopotential_data):
        raise FileNotFoundError(f'Either climate data or geopotential data, or both, do not exist')

    # Open climate datasets
    # Climate data retrieved from: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means?tab=overview
    with xr.open_dataset(climate_data) as ds_c, \
            xr.open_dataset(geopotential_data) as ds_g:

        ds_climate = ds_c.load()
        ds_geopotential = ds_g.load()

        # Convert geopotential height to geometric height and add to dataset
        r_earth = 6367.47 * 10e3  # [m] (Grib1 radius)
        g = 9.80665  # [m/s^2]
        ds_geopotential_metric = ds_geopotential.assign(
            altitude_climate=lambda ds_geo: r_earth * ((ds_geopotential.z / g) / (r_earth - (ds_geopotential.z / g)))
        )

        # Get latitude and longitude
        lat = ds_climate.latitude
        lon = ds_climate.longitude

        # Data retrieved from: https://ecmwf-projects.github.io/copernicus-training-c3s/reanalysis-climatology.html
        # Adjust longitude coordinates so that the coordinates range from -180 to 180, instead of 0 to 360
        ds_180 = ds_geopotential_metric.assign_coords(
            longitude=(((ds_geopotential_metric.longitude + 180) % 360) - 180)).sortby('longitude')

        ds_geopotential_cropped = ds_180.sel(longitude=lon, latitude=lat, method='nearest')

        # Reduce expver dimension
        ds_climate = ds_climate.reduce(np.nansum, 'expver')

        # Create list of climate name variables and months combined for one hydrological year
        climate_vars = list(ds_climate.keys())
        months_names = ['_oct', '_nov', '_dec', '_jan', '_feb', '_mar', '_apr', '_may', '_jun', '_jul', '_aug', '_sep']
        monthly_climate_vars = [f'{climate_var}{month_name:02}' for climate_var in climate_vars for month_name in
                                months_names]

        # Initialize arrays for the climate variable per data point and altitude
        climate_per_point = np.full((len(df), len(monthly_climate_vars)), np.nan)
        altitude_per_point = np.full((len(df), 1), np.nan)

        stake_lat = df.lat.round(2)
        stake_lon = df.lon.round(2)

        #TODO make this work for other data formats or just the years (YYYY) (int)
        stake_date = pd.to_datetime(df[column_name_year], format="%d/%m/%Y", errors='coerce')
        stake_year = np.array([date.year for date in stake_date])

        # Iterate through stake data, and get the climate variables and altitude for this point
        for idx, (lat, lon, year) in enumerate(zip(stake_lat, stake_lon, stake_year)):

            # Some years are float NaNs, these cannot be processed and therefore will be skipped
            if math.isnan(year):
                continue

            range_date = pd.date_range(
                start=str(int(year) - 1) + '-09-01',
                end=str(int(year)) + '-09-01', freq='ME'
            )

            # Select climate data for the point, or the nearest point to it in the range of the hydrological year
            climate_data_point = ds_climate.sel(latitude=lat, longitude=lon, time=range_date, method='nearest')

            # Convert selected data to Dataframe and save it
            if climate_data_point.dims:
                climate_points = climate_data_point.to_dataframe().drop(columns=['latitude', 'longitude'])
                climate_per_point[idx, :] = climate_points.to_numpy().flatten(order='F')

                # Select altitude data
                altitude_point = ds_geopotential_cropped.sel(latitude=lat, longitude=lon, method='nearest')
                altitude_per_point[idx] = altitude_point.altitude_climate.values[0]

        # Create DataFrames from arrays
        df_climate = pd.DataFrame(data=climate_per_point, columns=monthly_climate_vars)
        df_altitude = pd.DataFrame(data=altitude_per_point, columns=['altitude_climate'])

        # Concatenate DataFrames
        df_point_climate = pd.concat([df, df_climate, df_altitude], axis=1)

        # Drop records that do not have any geopotential height available
        df_point_climate.dropna(subset=['altitude_climate'], inplace=True)

        # Take the difference between the geopotential height and the elevation of the stake measurement
        df_point_climate['height_diff'] = df_point_climate['altitude_climate'] - df_point_climate['elevation']

        # Write to CSV
        df_point_climate.to_csv(output_fname, index=False)

        return df_point_climate
