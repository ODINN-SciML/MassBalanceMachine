import pandas as pd
from pyproj import CRS, Transformer

# File directory and names
file_dir = '.././data/'
file_name_in = '../data/files/Iceland_Stake_Data_Merged.csv'
file_name_out = '../data/files/Iceland_Stake_Data_Reprojected.csv'

df = pd.read_csv(file_dir + file_name_in)

# Define ISN93 (RD New) and WGS84 coordinate reference systems
isn93 = CRS.from_epsg(4659)  # ISN93 (RD New)
wgs84 = CRS.from_epsg(4326)  # WGS84 (EPSG:4326)

transformer = Transformer.from_crs(isn93, wgs84)


# Function to transform coordinates from ISN93 to WGS84
def transform_coordinates(lat, lon):
    lon_wgs84, lat_wgs84 = transformer.transform(lon, lat)
    return lat_wgs84, lon_wgs84


# Apply transformation to the DataFrame
df['lat'], df['lon'] = zip(*df.apply(lambda x: transform_coordinates(x['lat'], x['lon']), axis=1))

df.to_csv(file_dir + file_name_out, index=False)
