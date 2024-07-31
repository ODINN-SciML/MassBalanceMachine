"""
The Dataset class is part of the massbalancemachine package and is designed for user data processing, preparing
the data for the model training and testing.

Users provide stake measurement data, which must be in a WGMS-like format (if not, please see the data preprocessing
notebook).

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 21/07/2024
"""

import os
import logging
import pandas as pd
from get_climate_data import get_climate_features
from get_topo_data import get_topographical_features
from transform_to_monthly import transform_to_monthly

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Dataset:
    """
    A dataset class that retrieves both the climate and topography data, adds them to the dataset and
    transforms the data to a monthly time resolution.

    Attributes:
        data (pd.DataFrame): A pandas dataframe containing the raw data
        region (str): The name of the region, for saving the files accordingly
        data_dir (str): Path to the directory containing the raw data, and save intermediate results
        RGIIds (pd.Series): Series of RGI IDs from the data
    """

    def __init__(self, *, data: pd.DataFrame, region_name: str,
                 data_path: str):
        self.data = self._clean_data(data=data.copy())
        self.region = region_name
        self.data_dir = data_path
        self.RGIIds = self.data["RGIId"]

    def get_topo_features(self, *, vois: list[str]) -> None:
        """
        Fetches all the topographical data, for a list of variables of interest, using OGGM for the specified RGI IDs

        Args:
            vois (list[str]): A string containing the topographical variables of interest
        """
        output_fname = self._get_output_filename("topographical_features")
        self.data = get_topographical_features(self.data, output_fname, vois,
                                               self.RGIIds)

    def get_climate_features(self, *, climate_data: str,
                             geopotential_data: str) -> None:
        """
        Fetches all the climate data, for a list of variables of interest, for the specified RGI IDs.

        Args:
            climate_data (str): A netCDF-3 file location containing the climate data for the region of interest
            geopotential_data (str): A netCDF-3 file location containing the geopotential data
        """
        output_fname = self._get_output_filename("climate_features")
        self.data = get_climate_features(self.data, output_fname, climate_data,
                                         geopotential_data)

    def convert_to_monthly(self, *, vois_climate: list[str],
                           vois_topographical: list[str]) -> None:
        """
        Converts a variable period for the SMB target data measurement to a monthly time resolution.

        Args:
            vois_climate (list[str]): variables of interest from the climate data
            vois_topographical (list[str]): variables of interest from the topographical data
        """
        output_fname = self._get_output_filename("monthly_dataset")
        self.data = transform_to_monthly(self.data, vois_climate,
                                         vois_topographical, output_fname)

    def _get_output_filename(self, feature_type: str) -> str:
        """
        Generates the output filename for a given feature type.

        Args:
            feature_type (str): The type of feature (e.g., "topographical_features", "climate_features", "monthly")

        Returns:
            str: The full path to the output file
        """
        return os.path.join(self.data_dir, f"{self.region}_{feature_type}.csv")

    @staticmethod
    def _clean_data(data: pd.DataFrame) -> pd.DataFrame:
        """Cleans the input Dataframe by removing rows with missing or invalid dates, and
        removes rows with invalid POINT_BALANCE values."""
        # Fist, check if there are any faulty dates, if there are throw these out.
        corrected_data = Dataset._check_dates(data=data)

        # Second, drop all records without a POINT_BALANCE (e.g. NaN)
        corrected_data.dropna(subset=['POINT_BALANCE'], inplace=True)

        return corrected_data

    @staticmethod
    def _check_dates(data: pd.DataFrame) -> pd.DataFrame:
        """Cleans the input DataFrame by removing rows with missing or invalid dates."""
        required_columns = ["FROM_DATE", "TO_DATE"]
        Dataset._validate_columns(data, required_columns)

        try:
            data = Dataset._remove_missing_dates(data, required_columns)
            data = Dataset._convert_and_filter_dates(data)
            return data
        except Exception as e:
            logging.error(f"Error processing dates: {e}")
            raise

    @staticmethod
    def _validate_columns(data: pd.DataFrame,
                          required_columns: list[str]) -> None:
        """Validates that all required columns are present in the DataFrame."""
        if not all(col in data.columns for col in required_columns):
            logging.error(
                f"Missing one of the required columns: {required_columns}")
            raise KeyError(
                f"Required columns {required_columns} are not present in the DataFrame."
            )

    @staticmethod
    def _remove_missing_dates(data: pd.DataFrame,
                              date_columns: list[str]) -> pd.DataFrame:
        """Removes rows with missing dates from the DataFrame."""
        return data.dropna(subset=date_columns)

    @staticmethod
    def _convert_and_filter_dates(data: pd.DataFrame) -> pd.DataFrame:
        """Converts date columns to string and filters out invalid dates."""
        data = data.astype({"FROM_DATE": str, "TO_DATE": str})
        return data[~data["FROM_DATE"].str.endswith("99")
                    & ~data["TO_DATE"].str.endswith("99")]
