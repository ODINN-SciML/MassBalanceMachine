"""
A script in the last step of the data-processing pipeline that cleans the data
(i.e., remove redundant columns and records with NaN values).

The script takes a list of columns as inputs, these are the columns that will be dropped.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 04/06/2024
"""

import os
import pandas as pd
from argparse import ArgumentParser

# Columns that are parsed as arguments and will be dropped from the dataset
parser = ArgumentParser()
parser.add_argument('-l', '--list', required=False, help='Provide the columns that are redundant', type=str, default=[])

args = parser.parse_args()
columns_to_drop = [item for item in args.list.split(',')]

# Define the file directory and the input and output file names
file_dir = '.././data/files/'
file_name_in = 'region_stake_data_climate.csv'
file_name_out = 'region_stake_data_cleaned.csv'

full_path = os.path.join(file_dir, file_name_in)

# Check if the input file exists
if not os.path.exists(full_path):
    raise FileNotFoundError(f'{full_path} does not exist')

df = pd.read_csv(full_path)

# Drop records that do not have any geopotential height available
df.dropna(subset=['altitude_climate'], inplace=True)

# Drop other redundant columns
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

df.to_csv(os.path.join(file_dir, file_name_out), index=False)

