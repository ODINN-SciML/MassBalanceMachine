"""
This dataset class is part of the massbalancemachine package and is designed for user data processing.
After cleaning and processing, the dataset becomes ready for use in the model training and evaluation pipeline.

Users provide stake measurement data, which must include at least the latitude and longitude coordinates,
surface mass balance (either seasonal or annual), the start and end dates of the measurement, and a label
indicating whether the measurement is seasonal or annual. Additional topographical and meteorological features,
as well as RGI IDs (if not initially available), are then added to the dataset. Finally, the dataset is converted
to a monthly resolution.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 04/07/2024
"""

import os

from get_climate_data import get_climate_features
from get_topo_data import get_topo_features
from transform_to_monthly import convert_to_monthly


class Dataset:
    """
    A dataset class that retrieves both the climate and topography data add them to the dataset and
    transforms them to WGMS format.

    Attributes:
        data (pandas dataframe): A pandas dataframe containing the raw data
        dir_path (string): Path to the directory containing the raw data, and save intermediate results
        RGIIds (pandas column): A pandas column containing the RGI IDs of the raw data
    """
    def __init__(self, *, data=None, data_path='', column_name_RGIIDs='RGIId'):
        self.data = self.check_dates(data=data.copy())
        self.dir = data_path
        self.RGIIds = data[column_name_RGIIDs]

    @staticmethod
    def check_dates(data):
        """
            Cleans the input DataFrame by removing rows with missing or invalid dates.

            This method performs the following operations on the input DataFrame:
            1. Drops rows where the 'FROM_DATE' or 'TO_DATE' columns have missing (NaN) values.
            2. Drops rows where the 'FROM_DATE' or 'TO_DATE' columns have invalid dates (specifically, where the day is represented as '99').

            Args:
                data (pd.DataFrame): The input DataFrame containing date columns 'FROM_DATE' and 'TO_DATE'.

            Returns:
                pd.DataFrame: The cleaned DataFrame with invalid or incomplete date records removed.
        """
        # Drop data records that do not have a From or To data available
        data = data.dropna(subset=['FROM_DATE', 'TO_DATE'])
        # Drop data records that have an invalid From or To data (e.g., day=99)
        data = data[~data['FROM_DATE'].str.endswith('99')]
        data = data[~data['TO_DATE'].str.endswith('99')]

        return data

    def get_topo_features(self, voi):
        """
        Fetches all the topographical data, for a list of variables of interest, using OGGM for the specified RGI IDs

        Args:
            voi (string): A string containing the variables of interest
        """
        output_fname = os.path.join(self.dir, 'region_topographical_features.csv')

        self.data = get_topo_features(self.data, output_fname, voi, self.RGIIds)

    def get_climate_features(self, output_fname, climate_data, geopotential_data, column_name_year):
        """
        By specifying which source of reanalysis to use (e.g. ERA5, W5E5, MeteoSwiss…) it fetches all the climate
        data, for a list of variables of interest, for the specified RGI IDs.

        Args:
            output_fname (string): the name of the output file containing the raw data with topographical data
            climate_data (netCDF4): A netCDF4 file containing the climate data for the region of interest
            geopotential_data (netCDF4): A netCDF4 file containing the geopotential data
            col_name_yr (string): A string containing the variable name of the column of the last measurement in the hydrological year
            column_name_year (string/int): A string or int of the hydrological year, either just the year: YYYY, or any other format DD-MM-YYYY
        """
        output_fname = os.path.join(self.dir, output_fname)

        self.data = get_climate_features(self.data, output_fname, climate_data, geopotential_data, column_name_year)

    def convert_to_monthly(self, output_fname, vois_columns_climate, vois_topo_columns, smb_column_names, column_name_year):
        """
        Converts the climate and topographical data to monthly data

        Args:
            output_fname:
            vois_columns_climate:
            vois_topo_columns:
            smb_column_names:
            column_name_year:
        """
        output_fname = os.path.join(self.dir, output_fname)
        self.data = convert_to_monthly(self.data, output_fname, vois_columns_climate, vois_topo_columns, smb_column_names, column_name_year)