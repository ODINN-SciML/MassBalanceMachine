import pandas as pd
import numpy as np
from datetime import datetime
from scipy.spatial.distance import cdist
import pyproj
import xarray as xr

from scripts.wgs84_ch1903 import *
from scripts.config_FR import *

def processDatFile(fileName, path_dat, path_csv):
    """
    Converts a `.dat` file into a `.csv` file.

    This function processes a `.dat` file located in a specified directory, 
    performs data cleaning and transformations, and saves the processed 
    content as a `.csv` file in another specified directory.

    Parameters:
        fileName (str): The name of the file (without extension) to be converted.
        path_dat (str): The directory path where the `.dat` file is located.
        path_csv (str): The directory path where the output `.csv` file will be saved.

    File Format Assumptions:
        - The `.dat` file uses a semicolon (`;`) as a delimiter in the second row.
        - Rows after the third row use spaces as a delimiter.
        - Empty spaces and commas in the data are cleaned:
            - Commas in data values are replaced with hyphens (`-`).
            - Empty strings are removed from the row.

    Processing Steps:
        1. The second row of the `.dat` file is treated as the header.
           - Values are stripped of whitespace and joined with a comma (`,`).
        2. Rows after the third row are treated as data rows.
           - Values are stripped of whitespace.
           - Commas within the data values are replaced with hyphens (`-`).
           - Empty values are removed.
        3. The processed rows are written to the `.csv` file.

    Encoding:
        - The function uses `latin-1` encoding to handle file reading and writing.

    Example:
        Given a `.dat` file "example.dat" with content:
        ```
        Header information (ignored)
        ;Col1;Col2;Col3;
        Another header (ignored)
        Data start
        Value1 Value2 Value3
        Value4 Value5,Value6
        ```
        The resulting "example.csv" will contain:
        ```
        Col1,Col2,Col3
        Value1,Value2,Value3
        Value4,Value5-Value6
        ```

    Notes:
        - The function assumes the file structure outlined above and may not work with different formats.
        - Ensure the provided paths end with a directory separator (`/` or `\\`) based on the operating system.

    """
    with open(path_dat + fileName + '.dat', 'r',
              encoding='latin-1') as dat_file:
        with open(path_csv + fileName + '.csv',
                  'w',
                  newline='',
                  encoding='latin-1') as csv_file:
            for num_rows, row in enumerate(dat_file):
                if num_rows == 1:
                    row = [value.strip() for value in row.split(';')]
                    csv_file.write(','.join(row) + '\n')
                if num_rows > 3:
                    row = [value.strip() for value in row.split(' ')]
                    # Replace commas if there are any, otherwise this will create a bug.
                    row = [value.replace(',', '-') for value in row]
                    # Remove empty spaces.
                    row = [i for i in row if i]
                    csv_file.write(','.join(row) + '\n')


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


def CleanWinterDates(df_raw):
    # For some winter measurements the FROM_DATE is the same year as the TO_DATE (even same date)
    # Correct it by setting it to beginning of hydrological year:
    for index, row in df_raw.iterrows():
        if row['PERIOD'] == 'winter':
            df_raw.loc[index, 'FROM_DATE'] = str(
                pd.to_datetime(row['TO_DATE'], format='%Y%m%d').year -
                1) + '1001'
    for i, row in df_raw.iterrows():
        if pd.to_datetime(row['TO_DATE'],
                          format='%Y%m%d').year - pd.to_datetime(
                              row['FROM_DATE'], format='%Y%m%d').year != 1:
            # throw error if not corrected
            raise ValueError(
                'Date correction failed:', row['GLACIER'], row['PERIOD'],
                row['FROM_DATE'],
                pd.to_datetime(row['FROM_DATE'],
                               format='%Y%m%d').year, row['TO_DATE'],
                pd.to_datetime(row['TO_DATE'], format='%Y%m%d').year)
    return df_raw


def check_multiple_rgi_ids(df):
    """
    Checks if any glacier is associated with more than one RGIId.
    """
    rgi_per_glacier = df.groupby('GLACIER')['RGIId'].nunique()
    glaciers_with_multiple_rgi = rgi_per_glacier[rgi_per_glacier > 1]
    if not glaciers_with_multiple_rgi.empty:
        print("Alert: The following glaciers have more than one RGIId:")
        print(glaciers_with_multiple_rgi)
    else:
        print("All glaciers are correctly associated with a single RGIId.")


def clean_rgi_ids(df):
    """
    Cleans and preprocesses RGI IDs for specific glaciers based on predefined rules.
    """
    corrections = {
        # Format: 'GLACIER': {'valid_rgi': 'RGI60-XX.XXXXX', 'action': 'drop|replace'}
        'albigna': {
            'valid_rgi': 'RGI60-11.02285',
            'action': 'drop'
        },
        'adler': {
            'valid_rgi': 'RGI60-11.02764',
            'action': 'drop'
        },
        'allalin': {
            'valid_rgi': 'RGI60-11.02704',
            'action': 'drop'
        },
        'basodino': {
            'valid_rgi': 'RGI60-11.01987',
            'action': 'drop'
        },
        'blauschnee': {
            'action': 'remove_glacier'
        },
        'corvatsch': {
            'valid_rgi': 'RGI60-11.01962',
            'action': 'drop'
        },
        'damma': {
            'valid_rgi': 'RGI60-11.01246',
            'action': 'drop'
        },
        'findelen': {
            'valid_rgi': 'RGI60-11.02773',
            'action': 'drop'
        },
        'hohlaub': {
            'valid_rgi': 'RGI60-11.02679',
            'action': 'drop'
        },
        'gries': {
            'valid_rgi': 'RGI60-11.01876',
            'action': 'drop'
        },
        'limmern': {
            'valid_rgi': 'RGI60-11.00918',
            'action': 'drop'
        },
        'ofental': {
            'action': 'remove_glacier'
        },
        'orny': {
            'valid_rgi': 'RGI60-11.02775',
            'action': 'replace'
        },
        'otemma': {
            'valid_rgi': 'RGI60-11.02801',
            'action': 'drop'
        },
        'plattalva': {
            'valid_rgi': 'RGI60-11.00892',
            'action': 'replace'
        },
        'plainemorte': {
            'valid_rgi': 'RGI60-11.02072',
            'action': 'drop'
        },
        'rhone': {
            'valid_rgi': 'RGI60-11.01238',
            'action': 'drop'
        },
        'sanktanna': {
            'valid_rgi': 'RGI60-11.01367',
            'action': 'drop'
        },
        'sexrouge': {
            'valid_rgi': 'RGI60-11.02244',
            'action': 'drop'
        },
        'silvretta': {
            'valid_rgi': 'RGI60-11.00804',
            'action': 'drop'
        },
        'tsanfleuron': {
            'valid_rgi': 'RGI60-11.02249',
            'action': 'drop'
        },
        'unteraar': {
            'action': 'remove_glacier'
        }
    }

    for glacier, details in corrections.items():
        if details['action'] == 'drop':
            df.drop(
                df[(df.GLACIER == glacier)
                   & (df.RGIId != details['valid_rgi'])].index,
                inplace=True,
            )
        elif details['action'] == 'replace':
            df.loc[df.GLACIER == glacier, 'RGIId'] = details['valid_rgi']
        elif details['action'] == 'remove_glacier':
            df.drop(df[df.GLACIER == glacier].index, inplace=True)

    return df


def remove_close_points(df_gl):
    df_gl_cleaned = pd.DataFrame()
    for year in df_gl.YEAR.unique():
        for period in ['annual', 'winter', 'summer']:
            df_gl_y = df_gl[(df_gl.YEAR == year) & (df_gl.PERIOD == period)]
            if len(df_gl_y) <= 1:
                df_gl_cleaned = pd.concat([df_gl_cleaned, df_gl_y])
                continue

            # Calculate distances to other points
            df_gl_y['x'], df_gl_y['y'] = latlon_to_laea(
                df_gl_y['POINT_LAT'], df_gl_y['POINT_LON'])

            distance = cdist(df_gl_y[['x', 'y']], df_gl_y[['x', 'y']],
                             'euclidean')

            # Merge close points
            merged_indices = set()
            for i in range(len(df_gl_y)):
                if i in merged_indices:
                    continue  # Skip already merged points

                # Find close points (distance < 10m)
                close_indices = np.where(distance[i, :] < 10)[0]
                close_indices = [idx for idx in close_indices if idx != i]

                if close_indices:
                    mean_MB = df_gl_y.iloc[close_indices +
                                           [i]].POINT_BALANCE.mean()

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


def latlon_to_laea(lat, lon):
    # Define the transformer: WGS84 to ETRS89 / LAEA Europe
    transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:3035")

    # Perform the transformation
    easting, northing = transformer.transform(lat, lon)
    return easting, northing


def check_point_ids_contain_glacier(dataframe):
    """
    Checks that each row's POINT_ID contains the name of the GLACIER.
    
    Parameters:
        dataframe (pd.DataFrame): A pandas DataFrame with columns "GLACIER" and "POINT_ID".
        
    Returns:
        bool: True if all rows satisfy the condition, False otherwise.
        pd.DataFrame: A DataFrame of rows where the condition is not met.
    """
    if 'GLACIER' not in dataframe.columns or 'POINT_ID' not in dataframe.columns:
        raise ValueError(
            "The dataframe must contain 'GLACIER' and 'POINT_ID' columns.")

    # Check condition
    invalid_rows = dataframe[~dataframe.apply(
        lambda row: row['GLACIER'] in row['POINT_ID'], axis=1)]

    # Report
    if invalid_rows.empty:
        print(
            "All POINT_IDs correctly contain their respective GLACIER names.")
        return True, None
    else:
        print(
            f"Found {len(invalid_rows)} rows where POINT_ID does not contain the GLACIER name."
        )
        return False, invalid_rows


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
        raise ValueError(
            f"Expected grid_data shape ({num_months}, {nrows}, {ncols}), got {grid_data.shape}"
        )

    # Create the xarray DataArray
    data_array = xr.DataArray(
        np.flip(grid_data, axis=1),
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
    flattened_values = data_array.values.reshape(
        -1)  # Flatten entire 3D array (time, y, x)

    # flattened_values = data_array.values.flatten()
    y_coords, x_coords = np.meshgrid(data_array.y.values,
                                     data_array.x.values,
                                     indexing='ij')

    # Flatten the coordinate arrays
    flattened_x = np.tile(x_coords.flatten(),
                          len(time_dim))  # Repeat for each time step
    flattened_y = np.tile(y_coords.flatten(),
                          len(time_dim))  # Repeat for each time step

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
    data_array = data_array.assign_coords(lon=("x",
                                               lon_1d))  # Assign longitudes
    data_array = data_array.assign_coords(lat=("y",
                                               lat_1d))  # Assign latitudes

    # First, swap 'x' with 'lon' and 'y' with 'lat', keeping the time dimension intact
    data_array = data_array.swap_dims({'x': 'lon', 'y': 'lat'})

    # Reorder the dimensions to be (time, lon, lat)
    data_array = data_array.transpose("time", "lon", "lat")

    return data_array

def get_geodetic_MB():
    """
    Reads and processes the geodetic mass balance dataset for Swiss glaciers.
    - Filters out invalid date entries.
    - Ensures Astart matches the year from date_start, and Aend matches the year from date_end.
    - Identifies duplicates based on (Astart, Aend) and keeps only the row where date_end is closest to the end of the month.
    
    Returns:
        pd.DataFrame: Processed geodetic mass balance data.
    """
    
    # Load necessary data
    glacier_ids = get_glacier_ids()
    data_glamos = pd.read_csv(path_PMB_GLAMOS_csv + 'CH_wgms_dataset_all.csv')
    
    # Read geodetic MB dataset
    geodetic_mb = pd.read_csv(path_geodetic_MB_glamos + 'dV_DOI2024_allcomb.csv')

    # Get RGI IDs for the glaciers
    rgi_gl = data_glamos.RGIId.unique()
    sgi_gl = [
        glacier_ids[glacier_ids['rgi_id.v6'] == rgi]['sgi-id'].values[0] for rgi in rgi_gl
    ]
    geodetic_mb = geodetic_mb[geodetic_mb['SGI-ID'].isin(sgi_gl)]

    # Add glacier_name to geodetic_mb based on SGI-ID
    glacier_names = [
        glacier_ids[glacier_ids['sgi-id'] == sgi_id].index[0]
        for sgi_id in geodetic_mb['SGI-ID'].values
    ]
    geodetic_mb['glacier_name'] = glacier_names

    # Replace 'claridenL' with 'clariden'
    geodetic_mb['glacier_name'] = geodetic_mb['glacier_name'].replace('claridenL', 'clariden')

    # Rename A_end to Aend
    geodetic_mb.rename(columns={'A_end': 'Aend'}, inplace=True)

    # Function to replace 9999 with September 30
    def fix_invalid_dates(date):
        date_str = str(date)
        if date_str.endswith('9999'):
            return f"{date_str[:4]}0930"  # Replace '9999' with '0930'
        return date_str

    # Apply the function to both columns
    geodetic_mb['date_start'] = geodetic_mb['date_start'].apply(fix_invalid_dates)
    geodetic_mb['date_end'] = geodetic_mb['date_end'].apply(fix_invalid_dates)
    
    # Convert to datetime format
    geodetic_mb['date_start'] = pd.to_datetime(geodetic_mb['date_start'], format='%Y%m%d', errors='coerce')
    geodetic_mb['date_end'] = pd.to_datetime(geodetic_mb['date_end'], format='%Y%m%d', errors='coerce')
    
    # Manually set Astart and Aend based on date_start and date_end
    geodetic_mb['Astart'] = geodetic_mb['date_start'].dt.year
    geodetic_mb['Aend'] = geodetic_mb['date_end'].dt.year

    return geodetic_mb


def get_glacier_ids():
    glacier_ids = pd.read_csv(path_glacier_ids, sep=',')
    glacier_ids.rename(columns=lambda x: x.strip(), inplace=True)
    glacier_ids.sort_values(by='short_name', inplace=True)
    glacier_ids.set_index('short_name', inplace=True)
    
    return glacier_ids