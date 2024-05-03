import pandas as pd

# Define the file directory and the input and output file names
file_dir = '.././Data/Files/'
file_name_in = 'Iceland_Stake_Data_Climate.csv'
file_name_out = 'Iceland_Stake_Data_Cleaned.csv'

df = pd.read_csv(file_dir + file_name_in)

# Drop records that do not have any climate variables or altitude
df.dropna(subset=['altitude_climate'], inplace=True)

# Drop redundant columns
columns_to_drop = ['bw_fld', 'bs_fld', 'ba_fld', 'rhow', 'rhos', 'ims', 'nswe', 'dw', 'imw', 'ds']

# Check if any column exists in the DataFrame
if any(col in df.columns for col in columns_to_drop):
    # Drop the specified columns
    df.drop(columns=columns_to_drop, inplace=True)

# Rename columns

df.to_csv(file_dir + file_name_out, index=False)