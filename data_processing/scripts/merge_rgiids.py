import pandas as pd
import glob

# Directory containing RGIIDV6 CSV files
file_dir = '../data/stake-measurements-rgiid/'

# Find all CSV files in the directory
files = glob.glob(file_dir + '*.csv')

# Concatenate all CSV files into a single DataFrame
df = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)

# Write the merged DataFrame to a new CSV file
output_file = '../data/files/Iceland_Stake_Data_Merged.csv'
df.to_csv(output_file, sep=',', index=False)
