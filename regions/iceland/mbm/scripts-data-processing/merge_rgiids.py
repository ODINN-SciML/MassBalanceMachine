"""
This script merges all stake measurements, of all icecaps, into a single file.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 04/06/2024
"""

import pandas as pd
import glob

# Directory containing RGIIDV6 CSV files
file_dir = '.././data/stake-measurements-rgiid/'

# Find all CSV files in the directory
files = glob.glob(file_dir + '*.csv')

# Concatenate all CSV files into a single DataFrame
df = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)

# Write the merged DataFrame to a new CSV file
output_file = '.././data/files/Iceland_Stake_Data_Merged.csv'
df.to_csv(output_file, sep=',', index=False)
