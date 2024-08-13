import pandas as pd
import numpy as np
import pyproj
from scripts.wgs84_ch1903 import *
from scipy.spatial.distance import cdist

# Converts .dat files to .csv
def processDatFile(fileName, path_dat, path_csv):
    with open(path_dat + fileName + '.dat', 'r', encoding='latin-1') as dat_file:
        with open(path_csv + fileName + '.csv', 'w', newline='', encoding='latin-1') as csv_file:
            for num_rows, row in enumerate(dat_file):
                if num_rows == 1:
                    row = [value.strip() for value in row.split(';')]
                    csv_file.write(','.join(row) + '\n')
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
    date  = str(value)
    year  = date[:4]
    month = date[4:6]
    day   = date[6:8]
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
    df.date_fix0  = pd.to_datetime(df.date_fix0)
    df.date_fix1  = pd.to_datetime(df.date_fix1)
    
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
    lat, lon, height = converter.LV03toWGS84(df['x_pos'], df['y_pos'], df['z_pos'])
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
                continue

            # Calculate distances to other points:
            df_gl_y['x'], df_gl_y['y'] = latlon_to_laea(
                df_gl_y['POINT_LAT'], df_gl_y['POINT_LON'])

            distance = cdist(df_gl_y[['x', 'y']], df_gl_y[['x', 'y']],
                             'euclidean')
            df_gl_y['point'] = [
                (x, y)
                for x, y in zip(df_gl_y['POINT_LAT'], df_gl_y['POINT_LON'])
            ]

            indices_to_merge = []
            for i in range(len(df_gl_y)):
                row = df_gl_y.iloc[i]
                # search points with a distance less than 10m
                index_closest = np.where(distance[i, :] < 10)[0]

                # if not just itself:
                if len(index_closest) > 1:
                    # save the indices to merge
                    indices_to_merge.append(index_closest)

            # Convert numpy arrays to tuples and use a set to remove duplicates
            unique_indices = list(set(tuple(row) for row in indices_to_merge))

            # Convert tuples back to numpy arrays
            unique_indices = [np.array(row) for row in unique_indices]

            # Remove surplus points:
            indices_to_drop = []
            for index in unique_indices:
                mean_MB = df_gl_y.iloc[index].POINT_BALANCE.mean()
                df_gl_y.iloc[index[0]]['POINT_BALANCE'] = mean_MB
                indices_to_drop.append(index[1:])

            if len(indices_to_drop) > 0:
                indices_to_drop = df_gl_y.index[np.concatenate(
                    indices_to_drop)]
                df_gl_y.drop(index=indices_to_drop, inplace=True)
                # print('{}: Dropped points: {}, {}'.format(period, len(indices_to_drop), list(indices_to_drop)))
            df_gl_cleaned = pd.concat([df_gl_cleaned, df_gl_y])
    print('Number of points dropped:', len(df_gl) - len(df_gl_cleaned))
    return df_gl_cleaned