import pandas as pd
import glob

file_dir = '.././Data/'

# Directory containing RGIIDV6 CSV files
file_dir = file_dir + 'stake-measurements-rgiid/'

# Find all CSV files in the directory
files = glob.glob(file_dir + '*.csv')

# Concatenate all CSV files into a single DataFrame
df = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)

# Write the merged DataFrame to a new CSV file
output_file = file_dir + 'files/Iceland_Stake_Data_Merged.csv'
df.to_csv(output_file, sep=',', index=False)
