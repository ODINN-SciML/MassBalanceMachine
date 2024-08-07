"""
This script fetches all the stake measurement data from the three largest icecaps in Iceland and saves
the data in separate files.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 04/06/2024
"""

import os
import requests
import pandas as pd

def fetch_and_save_stake_data(icecap):
    # Define the paths to the directory containing icecap stakes and the directory for stake measurements
    icecap_dir = f'../data/stakes/{icecap}'
    stake_dir = '../data/stake-measurements/'

    # Check if the directory exists, if not, create it
    if not os.path.exists(icecap_dir):
        raise FileNotFoundError(f'{icecap_dir} does not exist')

    # Retrieve the stake data file associated with the specified icecap
    df = pd.read_csv(f'{icecap_dir}.txt', sep=',')

    # Iterate through the stakes associated with the specific icecap and retrieve data via API call
    for col in df.columns:
        url = f'https://joklavefsja-api.vedur.is/api/glaciers/v1/stake/{col}/measurements'
        headers = {'accept': 'text/csv'}

        response = requests.get(url, headers=headers)

        # Save each stake's measurements for the specific icecap
        stake_measurement_path = os.path.join(stake_dir, icecap)
        if not os.path.exists(stake_measurement_path):
            raise FileNotFoundError(f'{stake_measurement_path} does not exist')

        with open(os.path.join(stake_measurement_path, f'{col}.txt'), 'w') as f:
            f.write(response.text)

# List of icecaps to process
icecaps = ['HOFS', 'VATN', 'LANG']

# Process each icecap
for icecap in icecaps:
    fetch_and_save_stake_data(icecap)