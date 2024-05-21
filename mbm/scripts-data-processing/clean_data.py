import pandas as pd

# Define the file directory and the input and output file names
file_dir = '../data/files/'
file_name_in = 'Iceland_Stake_Data_Climate.csv'
file_name_out = 'Iceland_Stake_Data_Cleaned.csv'

df = pd.read_csv(file_dir + file_name_in)

# Drop records that do not have any climate variables or altitude
df.dropna(subset=['altitude_climate'], inplace=True)

# Drop redundant columns
columns_to_drop = ['bw_floating_date', 'bs_floating_date', 'ba_floating_date', 'rhow', 'rhos', 'ims', 'nswe', 'dw', 'imw', 'ds']

for col in columns_to_drop:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# Rename columns

df.to_csv(file_dir + file_name_out, index=False)