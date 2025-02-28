import pandas as pd
import numpy as np
import pyproj
import xarray as xr
from scripts.wgs84_ch1903 import *
from scipy.spatial.distance import cdist
from pyproj import Transformer

from scripts.helpers import *



def processDatFileGLWMB(fileName, path_dat, path_csv):
    with open(path_dat + fileName + '.dat', 'r',
              encoding='latin-1') as dat_file:
        with open(path_csv + fileName + '.csv',
                  'w',
                  newline='',
                  encoding='latin-1') as csv_file:
            el_bands_ = []
            for num_rows, row in enumerate(dat_file):
                if num_rows == 0:
                    row = [value.strip() for value in row.split(';')]
                    num_el_bands = row[4]
                if num_rows == 1:
                    row = [value.strip() for value in row.split(';')]
                    # Add columns for each el band
                    # b_w_eb_i  :  n columns with area-mean winter balance of each elevation band  [mm w.e.]
                    # b_a_eb_i  :  n columns with area-mean annual balance of each elevation band  [mm w.e.]
                    # A_eb_i    :  n columns with area of each elevation band  [km2]
                    row += [
                        'b_w_eb_' + str(i) for i in range(int(num_el_bands))
                    ]
                    row += [
                        'b_a_eb' + str(i) for i in range(int(num_el_bands))
                    ]
                    row += ['A_eb_' + str(i) for i in range(int(num_el_bands))]
                    csv_file.write(','.join(row[:-1]) + '\n')
                if num_rows > 3:
                    row = [value.strip() for value in row.split(' ')]
                    # replace commas if there are any otherwise will create bug:
                    row = [value.replace(',', '-') for value in row]
                    # remove empty spaces
                    row = [i for i in row if i]
                    csv_file.write(','.join(row) + '\n')


# Checks for duplicate years for a stake
def remove_dupl_years(df_stake):
    all_years = []
    rows = []
    for row_nb in range(len(df_stake)):
        year = df_stake.date_fix0.iloc[row_nb].year
        if year not in all_years:
            all_years.append(year)
            rows.append(row_nb)
    return df_stake.iloc[rows]


def datetime_obj(value):
    date = str(value)
    year = date[:4]
    month = date[4:6]
    day = date[6:8]
    return pd.to_datetime(month + '-' + day + '-' + year)


def transformDates(df_or):
    """Some dates are missing in the original glamos data and need to be corrected.
    Args:
        df_or (pd.DataFrame): raw glamos dataframe
    Returns:
        pd.DataFrame: dataframe with corrected dates
    """
    df = df_or.copy()
    # Correct dates that have years:
    df.date0 = df.date0.apply(lambda x: datetime_obj(x))
    df.date1 = df.date1.apply(lambda x: datetime_obj(x))

    df['date_fix0'] = [np.nan for i in range(len(df))]
    df['date_fix1'] = [np.nan for i in range(len(df))]

    # transform rest of date columns who have missing years:
    for i in range(len(df)):
        year = df.date0.iloc[i].year
        df.date_fix0.iloc[i] = '10' + '-' + '01' + '-' + str(year)
        df.date_fix1.iloc[i] = '09' + '-' + '30' + '-' + str(year + 1)

    # hydrological dates
    df.date_fix0 = pd.to_datetime(df.date_fix0)
    df.date_fix1 = pd.to_datetime(df.date_fix1)

    # dates in wgms format:
    df['date0'] = df.date0.apply(lambda x: x.strftime('%Y%m%d'))
    df['date1'] = df.date1.apply(lambda x: x.strftime('%Y%m%d'))
    return df


def LV03toWGS84(df):
    """Converts from swiss data coordinate system to lat/lon/height
    Args:
        df (pd.DataFrame): data in x/y swiss coordinates
    Returns:
        pd.DataFrame: data in lat/lon/coords
    """
    converter = GPSConverter()
    lat, lon, height = converter.LV03toWGS84(df['x_pos'], df['y_pos'],
                                             df['z_pos'])
    df['lat'] = lat
    df['lon'] = lon
    df['height'] = height
    df.drop(['x_pos', 'y_pos', 'z_pos'], axis=1, inplace=True)
    return df


def latlon_to_laea(lat, lon):
    # Define the transformer: WGS84 to ETRS89 / LAEA Europe
    transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:3035")

    # Perform the transformation
    easting, northing = transformer.transform(lat, lon)
    return easting, northing


def closest_point(point, points):
    """ Find closest point from a list of points. """
    return points[cdist([point], points).argmin()]


def match_value(df, col1, x, col2):
    """ Match value x from col1 row to value in col2. """
    return df[df[col1] == x][col2].values[0]


def remove_close_points(df_gl):
    df_gl_cleaned = pd.DataFrame()
    for year in df_gl.YEAR.unique():
        for period in ['annual', 'winter']:
            df_gl_y = df_gl[(df_gl.YEAR == year) & (df_gl.PERIOD == period)]
            if len(df_gl_y) <= 1:
                df_gl_cleaned = pd.concat([df_gl_cleaned, df_gl_y])
                continue

            # Calculate distances to other points
            df_gl_y['x'], df_gl_y['y'] = latlon_to_laea(
                df_gl_y['POINT_LAT'], df_gl_y['POINT_LON'])

            distance = cdist(df_gl_y[['x', 'y']], df_gl_y[['x', 'y']], 'euclidean')

            # Merge close points
            merged_indices = set()
            for i in range(len(df_gl_y)):
                if i in merged_indices:
                    continue  # Skip already merged points

                # Find close points (distance < 10m)
                close_indices = np.where(distance[i, :] < 10)[0]
                close_indices = [idx for idx in close_indices if idx != i]

                if close_indices:
                    mean_MB = df_gl_y.iloc[close_indices + [i]].POINT_BALANCE.mean()

                    # Assign mean balance to the first point
                    df_gl_y.loc[df_gl_y.index[i], 'POINT_BALANCE'] = mean_MB

                    # Mark other indices for removal
                    merged_indices.update(close_indices)

            # Drop surplus points
            indices_to_drop = list(merged_indices)
            df_gl_y = df_gl_y.drop(df_gl_y.index[indices_to_drop])

            # Append cleaned DataFrame
            df_gl_cleaned = pd.concat([df_gl_cleaned, df_gl_y])

    # Final output
    df_gl_cleaned.reset_index(drop=True, inplace=True)
    points_dropped = len(df_gl) - len(df_gl_cleaned)
    print(f'Number of points dropped: {points_dropped}')
    return df_gl_cleaned if points_dropped > 0 else df_gl


def xarray_to_dataframe(data_array):
    # Flatten the DataArray (values) and extract x and y coordinates
    flattened_values = data_array.values.flatten()
    y_coords, x_coords = np.meshgrid(data_array.y.values,
                                     data_array.x.values,
                                     indexing='ij')

    # Flatten the coordinate arrays
    flattened_x = x_coords.flatten()
    flattened_y = y_coords.flatten()

    # Create a DataFrame with columns for x, y, and value
    df = pd.DataFrame({
        'x_pos': flattened_x,
        'y_pos': flattened_y,
        'value': flattened_values
    })
    df['z_pos'] = 0

    # Convert to lat/lon
    df = LV03toWGS84(df)

    return df


def load_grid_file(filepath):
    with open(filepath, 'r') as file:
        # Read metadata
        metadata = {}
        for _ in range(6):  # First 6 lines are metadata
            line = file.readline().strip().split()
            metadata[line[0].lower()] = float(line[1])

        # Get ncols from metadata to control the number of columns
        ncols = int(metadata['ncols'])
        nrows = int(metadata['nrows'])

        # Initialize an empty list to store rows of the grid
        data = []

        # Read the grid data line by line
        row_ = []
        for line in file:
            row = line.strip().split()
            if len(row_) < ncols:
                row_ += row
            if len(row_) == ncols:
                data.append([
                        np.nan
                        if float(x) == metadata['nodata_value'] else float(x)
                        for x in row_
                    ])
                # reset row_
                row_ = []

        # Convert list to numpy array
        grid_data = np.array(data)

        # Check that shape of grid data is correct
        assert grid_data.shape == (nrows, ncols)

    return metadata, grid_data

def convert_to_xarray(grid_data, metadata, num_months):
    # Extract metadata values
    ncols = int(metadata['ncols'])
    nrows = int(metadata['nrows'])
    xllcorner = metadata['xllcorner']
    yllcorner = metadata['yllcorner']
    cellsize = metadata['cellsize']

    # Create x and y coordinates based on the metadata
    x_coords = xllcorner + np.arange(ncols) * cellsize
    y_coords = yllcorner + np.arange(nrows) * cellsize

    time_coords = np.arange(num_months)
    
    if grid_data.shape != (num_months, nrows, ncols):
        raise ValueError(f"Expected grid_data shape ({num_months}, {nrows}, {ncols}), got {grid_data.shape}")
 
    # Create the xarray DataArray
    data_array = xr.DataArray(np.flip(grid_data, axis=1),
                              #grid_data,
                              dims=("time", "y", "x"),
                              coords={
                                  "time": time_coords,
                                  "y": y_coords,
                                  "x": x_coords
                              },
                              name="grid_data")
    return data_array



def transform_xarray_coords_lv03_to_wgs84(data_array):
    # Extract time, y, and x dimensions
    time_dim = data_array.coords['time']
    
    # Flatten the DataArray (values) and extract x and y coordinates for each time step
    flattened_values = data_array.values.reshape(-1)  # Flatten entire 3D array (time, y, x)
   
    # flattened_values = data_array.values.flatten()
    y_coords, x_coords = np.meshgrid(data_array.y.values,
                                     data_array.x.values,
                                     indexing='ij')

    # Flatten the coordinate arrays
    flattened_x = np.tile(x_coords.flatten(), len(time_dim))  # Repeat for each time step
    flattened_y = np.tile(y_coords.flatten(), len(time_dim))  # Repeat for each time step

    # Create a DataFrame with columns for x, y, and value
    df = pd.DataFrame({
        'x_pos': flattened_x,
        'y_pos': flattened_y,
        'value': flattened_values
    })
    df['z_pos'] = 0

    # Convert to lat/lon
    df = LV03toWGS84(df)

    # Transform LV03 to WGS84 (lat, lon)
    lon, lat = df.lon.values, df.lat.values

    # Reshape the flattened WGS84 coordinates back to the original grid shape (time, y, x)
    lon = lon.reshape((len(time_dim), *x_coords.shape))  # Shape: (time, y, x)
    lat = lat.reshape((len(time_dim), *y_coords.shape))  # Shape: (time, y, x)

    # Assign the 1D WGS84 coordinates for swapping
    lon_1d = lon[0, 0, :]  # Use the first time slice, and take x (lon) values
    lat_1d = lat[0, :, 0]  # Use the first time slice, and take y (lat) values

    # Assign the WGS84 coordinates back to the xarray
    data_array = data_array.assign_coords(lon=("x", lon_1d))  # Assign longitudes
    data_array = data_array.assign_coords(lat=("y", lat_1d))  # Assign latitudes

    # First, swap 'x' with 'lon' and 'y' with 'lat', keeping the time dimension intact
    data_array = data_array.swap_dims({'x': 'lon', 'y': 'lat'})
    
    # Reorder the dimensions to be (time, lon, lat)
    data_array = data_array.transpose("time", "lon", "lat")

    return data_array


def CleanWinterDates(df_raw):
    # For some winter measurements the FROM_DATE is the same year as the TO_DATE (even same date)
    # Correct it by setting it to beginning of hydrological year:
    for index, row in df_raw.iterrows():
        if row['PERIOD'] == 'winter':
            df_raw.loc[index, 'FROM_DATE'] = str(
                pd.to_datetime(row['TO_DATE'], format='%Y%m%d').year - 1) + '1001'
    for i, row in df_raw.iterrows():
        if pd.to_datetime(row['TO_DATE'], format='%Y%m%d').year - pd.to_datetime(
                row['FROM_DATE'], format='%Y%m%d').year != 1:
            # throw error if not corrected
            raise ValueError('Date correction failed:', row['GLACIER'], row['PERIOD'], row['FROM_DATE'],
                pd.to_datetime(row['FROM_DATE'], format='%Y%m%d').year,
                row['TO_DATE'],
                pd.to_datetime(row['TO_DATE'], format='%Y%m%d').year)
    return df_raw