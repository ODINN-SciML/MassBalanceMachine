import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS, Transformer


def transform_crs(df, *, from_crs):
    # Define ISN93 (RD New) and WGS84 coordinate reference systems
    dataset_crs = CRS.from_epsg(from_crs)  # Dataset CRS
    wgs84 = CRS.from_epsg(4326)  # WGS84 (EPSG:4326)

    transformer = Transformer.from_crs(dataset_crs, wgs84)

    # Function to transform coordinates from ISN93 to WGS84
    def transform_coordinates(lat, lon):
        lon_wgs84, lat_wgs84 = transformer.transform(lon, lat)
        return lat_wgs84, lon_wgs84

    # Apply transformation to the DataFrame
    df['POINT_LAT'], df['POINT_LON'] = zip(*df.apply(lambda x: transform_coordinates(x['POINT_LAT'], x['POINT_LON']), axis=1))

    return df


def select_smb_value(row, date_to_smb):
    """
    Selects the right surface mass balance type that matches the order of the date the measurement was taken
    i.e., the first date corresponds to the first measurement taken at the start of the winter (period), and thus
    we want to match this with the surface mass balance of the winter period. These periods can of course be arbitrary.
    """
    return row[date_to_smb.get(row['date'], np.nan)]


def transform_dates(row):
    """
    Makes three dates based on the available measurement dates provided in the dataset

    Args:
        row: of the dataframe we want to retrieve the measurement periods from

    Returns:
        A list of tuples, with each tuple being a measurement period
    """
    date1, date2, date3 = row
    return [(date1, date2), (date2, date3), (date1, date3)]


def reshape_dataset(df, date_columns, smb_columns, ids):
    # Transform dates
    transformed_dates = df[date_columns].apply(transform_dates, axis=1).explode()
    transformed_dates = pd.DataFrame(transformed_dates.tolist(), columns=['FROM_DATE', 'TO_DATE'])

    # Assign a unique ID to each row
    df['ID'] = np.arange(len(df))

    # Melt the dataframe
    df_melted = df.melt(
        id_vars=ids + smb_columns + ['ID'],
        value_vars=date_columns,
        var_name='date'
    ).drop(['value'], axis=1)

    # Sort the melted dataframe
    df_melted = df_melted.sort_values(by=['ID', 'date']).reset_index(drop=True)

    # Combine the melted dataframe with the transformed dates
    df_combined = pd.concat([df_melted, transformed_dates], axis=1)

    # Create date to SMB mapping
    date_to_smb = dict(zip(date_columns, smb_columns))

    # Apply SMB value selection
    df_combined['POINT_BALANCE'] = df_combined.apply(select_smb_value, axis=1, date_to_smb=date_to_smb)

    # Drop the original SMB columns and temporary columns
    df_combined.drop(['date', 'ID'] + smb_columns, axis=1, inplace=True)

    return df_combined


def convert_to_wgms(wgms_data_columns, data, date_columns, smb_columns):
    """
    In case the dataset has one record, that belongs to a stake, for multiple measurements in a single hydrological
    year, i.e., winter, summer, annual, this function converts this single record to an individual record for each of
    the measurements in that specific hydrological year. For each record, also an identifier is added. The identifier
    can be used to aggregate the different measurement, either monthly or seasonally.

    Assumed is that a record has the following sort of structure:
        Stake ID, ..., Measurement Date 1, Measurement Date 2, Measurement Date 3, ..., Winter SMB, Summer SMB, Annual SMB, ...

    After melting the record, the structure will be as follows:
        Stake ID, ..., Measurement FROM Date, TO DATE, Winter SMB, ...,
        Stake ID, ..., Measurement FROM Date, TO DATE, Summer SMB, ...,
        Stake ID, ..., Measurement FROM Date, TO DATE, Annual SMB, ...,

    Args:
        wgms_data_columns (dict): list of column names of wgms column names and corresponding data column names
        data (pandas dataframe): A dictionary with keys as column names and values the corresponding data for the new dataframe
        ids (list): A list containing the identifiers needed for melting the existing dataframe
        date_columns (list): A list containing the names of the columns corresponding to dates column names
        smb_columns (list): A list containing the names of the columns corresponding to Surface Mass Balance column names
    Returns:
        df (pandas dataframe): Dataframe with the raw data is melted to a wgms-like format so that a single stake can have
        multiple records that correspond to different measurement periods.
    """

    df_combined = reshape_dataset(data, date_columns, smb_columns, list(wgms_data_columns.keys())[:-3])

    df_combined = df_combined[list(wgms_data_columns.keys())].rename(columns=wgms_data_columns)

    # Convert and format the date columns
    df_combined['TO_DATE'] = pd.to_datetime(df_combined['TO_DATE'], errors='coerce', dayfirst=True).dt.strftime(
        '%Y%m%d')
    df_combined['FROM_DATE'] = pd.to_datetime(df_combined['FROM_DATE'], errors='coerce', dayfirst=True).dt.strftime(
        '%Y%m%d')

    # Replace NaT with empty string
    df_combined['TO_DATE'] = df_combined['TO_DATE'].apply(lambda x: x if pd.notna(x) else '')
    df_combined['FROM_DATE'] = df_combined['FROM_DATE'].apply(lambda x: x if pd.notna(x) else '')

    return df_combined


def get_rgi(df, gdf):
    # Convert the stake measurement points (given in longitude and latitude) in the DataFrame to GeoDataFrame,
    # using the column names for longitude and latitude similar as the WGMS dataset.
    geometry = [Point(lon, lat) for lon, lat in zip(df['POINT_LON'], df['POINT_LAT'])]
    points_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=gdf.crs)

    # Perform a spatial joint for all the stake measurements that are within a section of the icecap that is
    # associated with a RGIId.
    joined_df = gpd.sjoin(points_gdf, gdf, how="left", predicate="within", lsuffix="_left", rsuffix="_right")

    # Only keep the columns of the original dataframe and the RGIIds
    columns_to_keep = df.columns.values.tolist()
    columns_to_keep.append('RGIId')
    joined_df = joined_df[columns_to_keep]

    return joined_df
