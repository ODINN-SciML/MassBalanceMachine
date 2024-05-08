import os
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

file_dir_in = '.././data/'
file_dir_out = '../data/stake-measurements-rgiid/'

gdf = gpd.read_file(file_dir_in + 'qgisv6/06_rgi60_Iceland.shp')

# Check if the directory exists
if not os.path.exists(file_dir_out):
    # Create the directory if it doesn't exist
    os.makedirs(file_dir_out)

# For all icecaps, match the RGIId to the stake measurements via their location
for icecap in glob.glob(file_dir_in + 'stake-measurements-merged/*.csv'):
    df = pd.read_csv(icecap)

    # Convert points DataFrame to GeoDataFrame
    geometry = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
    points_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=gdf.crs)

    # Perform a spatial joint for all the stake measurements that are within a section of the icecap that is associated with a RGIId
    joined_df = gpd.sjoin(points_gdf, gdf, how="left", predicate="within", lsuffix="_left", rsuffix="_right")

    # Convert coordinates from ISN93 to wgs84
    # joined_gdf = joined_df.to_crs(epsg=4326)

    # only keep the columns of the original df and the RGIId
    columns_to_keep = list(df.columns.values)
    columns_to_keep.append('RGIId')
    joined_df = joined_df[columns_to_keep]

    # Save DataFrame as CSV
    joined_df.to_csv(file_dir_out + icecap[-8:][:4] + '_RGIIDV6.csv', index=False)