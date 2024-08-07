"""
This script removes all records missing RGIIDs, and saves them in a separate file.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 04/06/2024
"""

import pandas as pd
import os

# File path for the stake data CSV file
csv_file = '.././data/files/Iceland_Stake_Data_Reprojected.csv'

# Check if the directory exists
if not os.path.exists(csv_file):
    raise FileNotFoundError(f'{csv_file} does not exist')

df = pd.read_csv(csv_file)

# Keep the stakes that are not assigned a glacier (NaN glacier IDs)
nan_glaciers = df[df['RGIId'].isna()]

# Filter out the stakes that are assigned a glacier (non-NaN glacier IDs)
df = df[df['RGIId'].notna()]

# File path for the new CSV file containing stakes without NaN glacier IDs
nan_glaciers_csv_file = '.././data/files/Iceland_Stake_Data_Nan_Glaciers.csv'

# Write the stake data with NaN glacier IDs to a new CSV file
nan_glaciers.to_csv(nan_glaciers_csv_file, index=False)

# Write the stake data without NaN glacier IDs back to the original CSV file
df.to_csv(csv_file, index=False)
