import pandas as pd

file_dir = '.././data/files/'

df = pd.read_csv(file_dir + 'Iceland_Stake_Data_Cleaned.csv')

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

aggregated_mass_balances.to_csv(file_dir + 'aggregate_mass_balances.csv', index=False)

