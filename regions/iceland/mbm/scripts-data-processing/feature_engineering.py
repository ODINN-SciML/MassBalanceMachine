"""
This script applies feature engineering by adding a new feature based on two existing features.
The new feature is the height difference between the elevation of the stake data and the geopotential height.
This feature will allow for downscaling the ERA5-Land data.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 04/06/2024
"""

import os
import pandas as pd

file_name = '.././data/files/Iceland_Stake_Data_Cleaned.csv'

# Check if the directory exists
if not os.path.exists(file_name):
    raise FileNotFoundError(f'{file_name} does not exist')

df = pd.read_csv(file_name)

# Take the difference between the geopotential height and the elevation of the stake measurement
df['height_diff'] = df['altitude_climate'] - df['elevation']

df.to_csv(file_name, index=False)