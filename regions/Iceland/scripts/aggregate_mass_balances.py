"""
This script aggregates the annual, summer, and winter mass balances per stake. The output
file of this script is stored as a csv file and used in QGIS to plot the cumulative mass balances
per glacier for all recorded measurements over time.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 04/06/2024
"""

import pandas as pd
import os

file_dir = '.././data/files/'

file_path = os.path.join(file_dir, 'Iceland_Stake_Data_Cleaned.csv')

# Check if the input file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f'{file_path} does not exist')

df = pd.read_csv(file_path)

# Group by 'stake' and calculate the cumulative sum for the 'annual and seasonal mass balances' columns
amb = df.groupby('stake')['ba_stratigraphic'].sum()
wmb = df.groupby('stake')['bw_stratigraphic'].sum()
smb = df.groupby('stake')['bs_stratigraphic'].sum()
lons = df.groupby('stake')['lon'].last()
lats = df.groupby('stake')['lat'].last()

# Make a new dataframe with all the aggregated mass balances and the stake name and their respective locations
aggregated_mass_balances = pd.concat([amb, wmb, smb, lons, lats], axis=1)

# Set the current index as a new column
aggregated_mass_balances.reset_index(inplace=True)

# Add a new default integer index
aggregated_mass_balances.reset_index(drop=True, inplace=True)

aggregated_mass_balances.to_csv(os.path.join(file_dir, 'aggregate_mass_balances.csv'), index=False)

