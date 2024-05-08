import pandas as pd

# File path for the stake data CSV file
csv_file = '.././Data/Files/Iceland_Stake_Data_Merged.csv'

df = pd.read_csv(csv_file)

# Keep the stakes that are not assigned a glacier (NaN glacier IDs)
nan_glaciers = df[df['RGIId'].isna()]

# Filter out the stakes that are assigned a glacier (non-NaN glacier IDs)
df = df[df['RGIId'].notna()]

# File path for the new CSV file containing stakes without NaN glacier IDs
nan_glaciers_csv_file = '.././Data/Iceland_Stake_Data_Nan_Glaciers.csv'

# Write the stake data with NaN glacier IDs to a new CSV file
nan_glaciers.to_csv(nan_glaciers_csv_file, index=False)

# Write the stake data without NaN glacier IDs back to the original CSV file
df.to_csv(csv_file, index=False)
