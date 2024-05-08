import pandas as pd
import os.path

# File directory
file_dir = '.././Data/Files/'
file_name_in = 'Iceland_Stake_Climate.csv'
file_name_out = 'Iceland_Stake_WGMS.csv'

# File path for Iceland Stake Data
file_path = os.path.join(file_dir, file_name_in)

# Read Iceland Stake Data into DataFrame
df = pd.read_csv(file_path)

# Construct WGMS data dictionary
wgms_data = {
    'POLITICAL_UNIT': ['IS'] * len(df),
    'NAME': df['Name'],
    'WGMS_ID': [None] * len(df),
    'YEAR': df['yr'],
    'POINT_ID': df['stake'],
    'FROM_DATE': df['d1'],
    'TO_DATE': df['d3'],
    'POINT_LAT': df['lat'],
    'POINT_LON': df['lon'],
    'POINT_ELEVATION': df['elevation'],
    'POINT_BALANCE': df['ba_stratigraphic'],
    'POINT_BALANCE_UNCERTAINTY': [None] * len(df),
    'DENSITY': df['rhow'],
    'DENSITY_UNCERTAINTY': [None] * len(df),
    'BALANCE_CODE': ['BA'] * len(df),
    'REMARKS': [''] * len(df),
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
