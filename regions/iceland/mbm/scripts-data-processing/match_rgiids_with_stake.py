"""
This script matches the stake measurement data with the RGI ID for each glacier on the icecap.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 04/06/2024
"""

import os
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

file_dir_in = '.././data/'
file_dir_out = '../data/stake-measurements-rgiid/'

# Data obtained from: https://nsidc.org/data/nsidc-0770/versions/6
file_path = os.path.join(file_dir_in, 'qgisv6/06_rgi60_Iceland.shp')

# Check if the directory exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f'{file_path} does not exist')

gdf = gpd.read_file(file_path)

# For all icecaps, match the RGIId to the stake measurements via their location
for icecap in glob.glob(file_dir_in + 'stake-measurements-merged/*.csv'):
    df = pd.read_csv(icecap)

    # Convert points DataFrame to GeoDataFrame
    geometry = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
    points_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=gdf.crs)

    # Perform a spatial joint for all the stake measurements that are within a section of the icecap that is associated with a RGIId
    joined_df = gpd.sjoin(points_gdf, gdf, how="left", predicate="within", lsuffix="_left", rsuffix="_right")

    # only keep the columns of the original df and the RGIId
    columns_to_keep = df.columns.values.tolist()
    columns_to_keep.append('RGIId')
    joined_df = joined_df[columns_to_keep]

    # Save DataFrame as CSV
    joined_df.to_csv(file_dir_out + icecap[-8:][:4] + '_RGIIDV6.csv', index=False)