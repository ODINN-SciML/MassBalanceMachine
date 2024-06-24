"""
This script transforms the dataset to a WGMS file dataset format, with all the columns that are in the 
WGMS database. The dataset will be ready to be uploaded to the WGMS database. The WGMS ID will be assigned 
by WGMS. Depending on the data that is available for the stake measurements the columns are either copied,
or left blank. 

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 04/06/2024
"""

import pandas as pd
import os.path

# File directory
file_dir = '.././data/files/'
file_name_in = 'region_stake_data_climate.csv' # Input file is the output file of the get_climate_data.py script
file_name_out = 'region_stake_data_wgms.csv'

# File path for Iceland Stake Data
file_path = os.path.join(file_dir, file_name_in)

# Check if the directory exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f'{file_path} does not exist')

# Read Iceland Stake Data into DataFrame
df = pd.read_csv(file_path)

# Construct WGMS data dictionary with the appropriate columns that are required for a WGMS dataset
# Either copy the data from df to df_wgms by: COLUM_NAME: df['COLUMN_NAME'], or if not available: COLUMN_NAME: [None] * len(df)
wgms_data = {
    'POLITICAL_UNIT': ['LAND CODE'] * len(df),
    'NAME': df[''],
    'WGMS_ID': df[''],
    'YEAR': df[''],
    'POINT_ID': df[''],
    'FROM_DATE': df[''],
    'TO_DATE': df[''],
    'POINT_LAT': df[''],
    'POINT_LON': df[''],
    'POINT_ELEVATION': df[''],
    'POINT_BALANCE': df[''],
    'POINT_BALANCE_UNCERTAINTY': df[''],
    'DENSITY': df[''],
    'DENSITY_UNCERTAINTY': df[''],
    'BALANCE_CODE': df[''],
    'REMARKS': df[''],
}

# Create DataFrame for WGMS data
df_wgms = pd.DataFrame(data=wgms_data)

# Capitalize all the glacier names
df_wgms['NAME'] = df_wgms['NAME'].str.upper()

# Convert date columns to datetime format
df_wgms['TO_DATE'] = pd.to_datetime(df_wgms['TO_DATE'], errors='coerce', dayfirst=True)
df_wgms['FROM_DATE'] = pd.to_datetime(df_wgms['FROM_DATE'], errors='coerce', dayfirst=True)

# Apply lambda function to convert date format
df_wgms['TO_DATE'] = df_wgms['TO_DATE'].apply(lambda x: x.strftime('%Y%m%d') if pd.notna(x) else '')
df_wgms['FROM_DATE'] = df_wgms['FROM_DATE'].apply(lambda x: x.strftime('%Y%m%d') if pd.notna(x) else '')

# Save the new WGMS dataframe if it does not exist yet
file_wgms = os.path.join(file_dir, file_name_out)
if not os.path.isfile(file_wgms):
    df_wgms.to_csv(file_wgms, index=False)
