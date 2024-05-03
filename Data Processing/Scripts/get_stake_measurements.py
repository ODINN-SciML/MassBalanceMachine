import os
import requests
import pandas as pd

# Define the path to the directory containing icecap stakes,
# and a path for the directory the stake measurements are saved
icecap_dir = '.././Data/Stakes/' + 'VATN' #HOFS, VATN, LANG
stake_dir = '.././Data/Stake Measurements/'

# Check if the directory exists, if not, create it
if not os.path.exists(icecap_dir):
    os.makedirs(icecap_dir)

# Retrieve the stake data file associated with the specified icecap
df = pd.read_csv(f'{icecap_dir}.txt', sep=', ')

# Iterate through the stakes associated with the specific icecap and retrieve data via API call
for col in df.columns:

    url = f'https://joklavefsja-api.vedur.is/api/glaciers/v1/stake/{col}/measurements'
    headers = {'accept': 'text/csv'}
    
    response = requests.get(url, headers=headers)

    # Save for each stake the measurements for the specific icecap
    with open(os.path.join(stake_dir, f'{icecap_dir}/{col}.txt'), 'w') as f:
        f.write(response.text)