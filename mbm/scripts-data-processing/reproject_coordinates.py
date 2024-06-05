"""
This script transforms the any CRS to WGS84 CRS for the stake measurements in the region of interest.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 04/06/2024
"""

import pandas as pd
import os
import re
from pyproj import CRS, Transformer
from argparse import ArgumentParser

# Argument for the CRS of the stake measurements
parser = ArgumentParser()
parser.add_argument('-CRS', '--CRS', required=True, help='Provide the CRS for the stake measurements in the region of interest', type=str)

args = parser.parse_args()
RGI = args.CRS

if not CRS:
    raise ValueError("No CRS provided. Please provide the CRS for the stake measurements.") 

# File directory and names
file_dir = '.././data/'
file_name_in = 'region_stake_data.csv'
file_name_out = 'region_stake_data_reprojected.csv'

file_path = os.path.join(file_dir, file_name_in)

# Check if the directory exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f'{file_path} does not exist')

df = pd.read_csv(file_path)

# Define current CRS of the stake measurements in the region of interest and WGS84 coordinate reference systems
isn93 = CRS.from_epsg(CRS)   # CRS of the stake measurements in the region of interest
wgs84 = CRS.from_epsg(4326)  # WGS84 (EPSG:4326)

transformer = Transformer.from_crs(isn93, wgs84)

# Function to transform coordinates from ISN93 to WGS84
def transform_coordinates(lat, lon):
    lon_wgs84, lat_wgs84 = transformer.transform(lon, lat)
    return lat_wgs84, lon_wgs84

# Check if the dataset contains longitude and latitude columns, if not raise an exception
column_names = df.columns
lon_pattern = 'lon[a-zA-Z]*'
lat_pattern = 'lat[a-zA-Z]*'

# Initialize variables to store matching column names
matching_lon_column = None
matching_lat_column = None

# Check for matching column names
for col in column_names:
    if re.match(lon_pattern, col):
        matching_lon_column = col
        break  # Exit loop once a match is found
    if re.match(lat_pattern, col):
        matching_lat_column = col
        break  # Exit loop once a match is found

# Check if either lon or lat pattern is missing in the column names
if matching_lon_column is None or matching_lat_column is None:
    raise Exception("Either lon or lat pattern is missing in the column names.")

# Apply transformation to the DataFrame
df[matching_lat_column], df[matching_lon_column] = zip(*df.apply(lambda x: transform_coordinates(x[matching_lat_column], x[matching_lon_column]), axis=1))

df.to_csv(os.path.join(file_dir, file_name_out), index=False)
