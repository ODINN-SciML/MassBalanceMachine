import glob
import os.path
import pandas as pd

# Define the path to the directory containing icecap stake data,
# and a path for the directory the merged files are saved
file_dir = '.././data/stake-measurements/'
merged_dir = '../data/stake-measurements-merged/'

icecaps = ['HOFS', 'LANG', 'VATN']


def read_files(icecap):
    # Create directory for merged files if it doesn't exist
    os.makedirs(merged_dir, exist_ok=True)

    file_path = os.path.join(file_dir, icecap, '*.txt')
    stake_files = glob.glob(file_path)

    # Read all stake data files and concatenate them into a single Dataframe
    df = pd.concat(
        (pd.read_csv(filename) for filename in stake_files),
        ignore_index=True
    )

    # Save the merged file as a csv, one for every icecap
    df.to_csv(os.path.join(merged_dir, f'{icecap}.csv'), sep=',', index=False)


# Iterate over each icecap and merge its stake data files
for ice_cap in icecaps:
    read_files(ice_cap)
