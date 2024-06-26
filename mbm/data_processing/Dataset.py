import os

from get_climate_data import get_climate_features
from get_topo_data import get_topo_features
from transform_to_monthly import convert_to_monthly


class Dataset:
    """
    A dataset class that retrieves both the climate and topography data add them to the dataset and
    transforms them to WGMS format.

    Attributes:
        df (pandas dataframe): A pandas dataframe containing the raw data
        RGIIds (pandas column): A pandas column containing the RGI IDs of the raw data
        dir_path (string): Path to the directory containing the raw data, and save intermediate results
        region_ID (string): A string containing the region of interest ID of the raw data
    """
    def __init__(self, df, column_name_RGI, dir_path, region_ID):
        self.df = df.copy()
        self.RGIIds = df[column_name_RGI]
        self.dir = dir_path
        self.region_ID = region_ID

    def get_topo_features(self, output_fname, voi):
        """
        Fetches all the topographical data, for a list of variables of interest, using OGGM for the specified RGI IDs

        Args:
            output_fname (string): the name of the output file containing the raw data with topographical data
            voi (string): A string containing the variables of interest

        Returns:
            topo_features (pandas dataframe): A pandas dataframe containing the topographical features and the raw data
        """
        output_fname = os.path.join(self.dir, output_fname)

        self.df = get_topo_features(self.df, output_fname, voi, self.RGIIds)

    def get_climate_features(self, output_fname, climate_data, geopotential_data, column_name_year):
        """
        By specifying which source of reanalysis to use (e.g. ERA5, W5E5, MeteoSwiss…) it fetches all the climate
        data, for a list of variables of interest, for the specified RGI IDs.

        Args:
            climate_data (netCDF4): A netCDF4 file containing the climate data for the region of interest
            geopotential_data (netCDF4): A netCDF4 file containing the geopotential data
            col_name_yr (string): A string containing the variable name of the column of the last measurement in the hydrological year
            column_name_year (string/int): A string or int of the hydrological year, either just the year: YYYY, or any other format DD-MM-YYYY
        """
        output_fname = os.path.join(self.dir, output_fname)

        self.df = get_climate_features(self.df, output_fname, climate_data, geopotential_data, column_name_year)


    def convert_to_monthly(self, output_fname, vois_columns_climate, vois_topo_columns, smb_column_names, column_name_year):
        output_fname = os.path.join(self.dir, output_fname)
        self.df = convert_to_monthly(self.df, output_fname, vois_columns_climate, vois_topo_columns, smb_column_names, column_name_year)