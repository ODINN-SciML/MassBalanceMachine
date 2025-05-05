"""
This module contains functions to transform and reshape glaciological data for analysis and integration with the
WGMS (World Glacier Monitoring Service) dataset. The functions include coordinate transformation, selection and
reshaping of measurement periods, and spatial joining of glacier outlines.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 21/07/2024
"""

from typing import Any

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS, Transformer
import hashlib


def convert_to_wgs84(*, data: pd.DataFrame,
                     from_crs: str | int) -> pd.DataFrame:
    """
    Transform coordinates from a given CRS to WGS84.

    Args:
        data (pandas.DataFrame): DataFrame containing 'POINT_LAT' and 'POINT_LON' columns.
        from_crs (int): EPSG code of the source coordinate reference system.

    Returns:
        pandas.DataFrame: DataFrame with transformed coordinates in WGS84.
    """

    # Define FROM CRS and WGS84 coordinate reference systems
    dataset_crs = CRS.from_epsg(from_crs)  # Dataset CRS
    wgs84 = CRS.from_epsg(4326)  # WGS84 (EPSG:4326)

    transformer = Transformer.from_crs(dataset_crs, wgs84)

    # Function to transform coordinates from FROM_CRS to WGS84
    def transform_coordinates(lat, lon):
        lon_wgs84, lat_wgs84 = transformer.transform(lon, lat)
        return lat_wgs84, lon_wgs84

    # Apply transformation to the DataFrame
    data["POINT_LAT"], data["POINT_LON"] = zip(*data.apply(
        lambda x: transform_coordinates(x["POINT_LAT"], x["POINT_LON"]),
        axis=1))

    return data


def _select_smb_value(row: pd.Series, date_to_smb: dict) -> pd.Series:
    """
    Selects the right surface mass balance type that matches the order of the date the measurement was taken
    i.e., the first date corresponds to the first measurement taken at the start of the winter (period), and thus
    we want to match this with the surface mass balance of the winter period. These periods can of course be arbitrary.
    """
    return row[date_to_smb.get(row["date"], np.nan)]


def _transform_dates(row: pd.Series) -> "list[tuple[Any, Any]]":
    """
    Generate measurement periods from three dates.

    Args:
        row (pandas.Series): A row containing three date values.

    Returns:
        list: A list of tuples, with each tuple representing a measurement period (start_date, end_date).
    """
    date1, date2, date3 = row
    return [(date1, date2), (date2, date3), (date1, date3)]


def _reshape_dataset(df: pd.DataFrame, date_columns: "list[str]",
                     smb_columns: "list[str]", ids: "list[str]") -> pd.DataFrame:
    """
    Reshape the dataset to create individual records for each measurement period.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        date_columns (list): List of column names containing measurement dates.
        smb_columns (list): List of column names containing surface mass balance values.
        ids (list): List of column names to use as identifiers.

    Returns:
        pandas.DataFrame: Reshaped DataFrame with individual records for each measurement period.
    """

    # Transform dates
    transformed_dates = df[date_columns].apply(_transform_dates,
                                               axis=1).explode()
    transformed_dates = pd.DataFrame(transformed_dates.tolist(),
                                     columns=["FROM_DATE", "TO_DATE"])

    # Assign a unique ID to each row
    df["ID"] = np.arange(len(df))

    # Melt the dataframe
    df_melted = df.melt(id_vars=ids + smb_columns + ["ID"],
                        value_vars=date_columns,
                        var_name="date").drop(["value"], axis=1)

    # Sort the melted dataframe
    df_melted = df_melted.sort_values(by=["ID", "date"]).reset_index(drop=True)

    # Combine the melted dataframe with the transformed dates
    df_combined = pd.concat([df_melted, transformed_dates], axis=1)

    # Create date to SMB mapping
    date_to_smb = dict(zip(date_columns, smb_columns))

    # Apply SMB value selection
    df_combined["POINT_BALANCE"] = df_combined.apply(_select_smb_value,
                                                     axis=1,
                                                     date_to_smb=date_to_smb)

    # Drop the original SMB columns and temporary columns
    df_combined.drop(["date", "ID"] + smb_columns, axis=1, inplace=True)

    return df_combined


def convert_to_wgms(
    *,
    wgms_data_columns: dict,
    data: pd.DataFrame,
    date_columns: "list[str]",
    smb_columns: "list[str]",
) -> pd.DataFrame:
    """
    Convert dataset to WGMS-like format with individual records for each measurement period.

    Args:
        wgms_data_columns (dict): Mapping of WGMS column names to corresponding data column names.
        data (pandas.DataFrame): Input DataFrame containing the raw data.
        date_columns (list): List of column names containing measurement dates.
        smb_columns (list): List of column names containing surface mass balance values.

    Returns:
        pandas.DataFrame: DataFrame in WGMS-like format with individual records for each measurement period.
    """

    df_combined = _reshape_dataset(data, date_columns, smb_columns,
                                   list(wgms_data_columns.keys())[:-3])

    df_combined = df_combined[list(
        wgms_data_columns.keys())].rename(columns=wgms_data_columns)

    # Convert and format the date columns
    df_combined["TO_DATE"] = pd.to_datetime(
        df_combined["TO_DATE"], errors="coerce",
        dayfirst=True).dt.strftime("%Y%m%d")
    df_combined["FROM_DATE"] = pd.to_datetime(
        df_combined["FROM_DATE"], errors="coerce",
        dayfirst=True).dt.strftime("%Y%m%d")

    # Replace NaT with empty string
    df_combined["TO_DATE"] = df_combined["TO_DATE"].apply(
        lambda x: x if pd.notna(x) else "")
    df_combined["FROM_DATE"] = df_combined["FROM_DATE"].apply(
        lambda x: x if pd.notna(x) else "")

    return df_combined


def get_rgi(*, data: pd.DataFrame,
            glacier_outlines: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Assign RGI IDs to stake measurements based on their spatial location within glacier outlines.

    Args:
        data (pandas.DataFrame): DataFrame containing stake measurements with 'POINT_LON' and 'POINT_LAT' columns.
        glacier_outlines (geopandas.GeoDataFrame): GeoDataFrame containing glacier outlines with 'RGIId' column.

    Returns:
        geopandas.GeoDataFrame: DataFrame with original data and added 'RGIId' column for each stake measurement.
    """

    # Convert the stake measurement points (given in longitude and latitude) in the DataFrame to GeoDataFrame,
    # using the column names for longitude and latitude similar as the WGMS
    # dataset.
    geometry = [
        Point(lon, lat)
        for lon, lat in zip(data["POINT_LON"], data["POINT_LAT"])
    ]
    points_gdf = gpd.GeoDataFrame(data,
                                  geometry=geometry,
                                  crs=glacier_outlines.crs)

    # Perform a spatial joint for all the stake measurements that are within a section of the icecap that is
    # associated with a RGIId.
    joined_df = gpd.sjoin(
        points_gdf,
        glacier_outlines,
        how="left",
        predicate="within",
        lsuffix="_left",
        rsuffix="_right",
    )

    # Only keep the columns of the original dataframe and the RGIIds
    columns_to_keep = data.columns.values.tolist()
    columns_to_keep.append("RGIId")
    joined_df = joined_df[columns_to_keep]

    return joined_df

# Generate a unique glacier-wide ID
def get_hash(unique_string):
    unique_id = hashlib.md5(
        unique_string.encode()).hexdigest()[:10]  # Shortened hash
    return unique_id
