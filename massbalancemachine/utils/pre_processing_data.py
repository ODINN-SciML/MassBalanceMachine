import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def convert_to_wgms(df, column_names_dates, column_names_SMB):
    """
    In case the dataset has one record, that belongs to a stake, for multiple measurements in a single hydrological
    year, i.e., winter, summer, annual, this function converts this single record to an individual record for each of
    the measurements in that specific hydrological year. For each record, also an identifier is added. The identifier
    can be used to aggregate the different measurement, either monthly or seasonally.

    Assumed is that a record has the following sort of structure:
        Stake ID, ..., Measurement Date 1, Measurement Date 2, Measurement Date 3, ..., Winter SMB, Summer SMB, Annual SMB, ...

    After melting the record, the structure will be as follows:
        Stake ID, ..., Measurement Date 1, Winter SMB, ..., BW
        Stake ID, ..., Measurement Date 2, Summer SMB, ..., BS
        Stake ID, ..., Measurement Date 3, Annual SMB, ..., BA

    Args:
        df (pandas dataframe): Dataframe containing records that have multiple measurements for a single stake
    Returns:
        df (pandas dataframe): Dataframe with the raw data is melted to a wgms-like format so that a single stake can have
        multiple records that correspond to different measurement periods.
    """

    return None


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
