"""
The Dataset class is part of the massbalancemachine package and is designed for user data processing, preparing
the data for the model training and testing.

Users provide stake measurement data, which must be in a WGMS-like format (if not, please see the data preprocessing
notebook).

Date Created: 21/07/2024
"""

import os
from typing import Union, Callable

import numpy as np
import xarray as xr
import oggm
import config
import logging
import pandas as pd
import torch
from skorch.helper import SliceDataset

from get_climate_data import get_climate_features, retrieve_clear_sky_rad, smooth_era5land_by_mode
from get_topo_data import get_topographical_features, get_glacier_mask
from transform_to_monthly import transform_to_monthly
from create_glacier_grid import create_glacier_grid_RGI

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Dataset:
    """
    A dataset class that retrieves both the climate and topography data, adds them to the dataset and
    transforms the data to a monthly time resolution.

    Attributes:
        cfg (config.Config): Configuration instance.
        data (pd.DataFrame): A pandas dataframe containing the raw data
        region (str): The name of the region, for saving the files accordingly
        data_dir (str): Path to the directory containing the raw data, and save intermediate results
        RGIIds (pd.Series): Series of RGI IDs from the data
    """

    def __init__(self, *, cfg: config.Config, data: pd.DataFrame,
                 region_name: str, data_path: str):
        self.cfg = cfg
        self.data = self._clean_data(data=data.copy())
        self.region = region_name
        self.data_dir = data_path
        self.RGIIds = self.data["RGIId"]

    def get_topo_features(self,
                          *,
                          vois: "list[str]",
                          custom_working_dir: str = '') -> None:
        """
        Fetches all the topographical data, for a list of variables of interest, using OGGM for the specified RGI IDs

        Args:
            vois (list[str]): A string containing the topographical variables of interest
            custom_working_dir (str, optional): The path to the custom working directory for OGGM data. Default to ''
        """
        output_fname = self._get_output_filename("topographical_features")
        self.data = get_topographical_features(self.data, output_fname, vois,
                                               self.RGIIds, custom_working_dir,
                                               self.cfg)

    def get_climate_features(self,
                             *,
                             climate_data: str,
                             geopotential_data: str,
                             change_units: bool = False,
                             smoothing_vois: dict = None) -> None:
        """
        Fetches all the climate data, for a list of variables of interest, for the specified RGI IDs.

        Args:
            climate_data (str): A netCDF-3 file location containing the climate data for the region of interest
            geopotential_data (str): A netCDF-3 file location containing the geopotential data
            change_units (bool, optional): A boolean indicating whether to change the units of the climate data. Default to False.
            smoothing_vois (dict, optional): A dictionary containing the variables of interest for smoothing climate artifacts. Default to None.
        """
        output_fname = self._get_output_filename("climate_features")

        smoothing_vois = smoothing_vois or {}  # Safely default to empty dict
        vois_climate = smoothing_vois.get('vois_climate')
        vois_other = smoothing_vois.get('vois_other')

        self.data = get_climate_features(self.data, output_fname, climate_data,
                                         geopotential_data, change_units,
                                         vois_climate, vois_other)

    def get_potential_rad(self, path_to_direct):
        """Fetches monthly clear sky radiation data for each glacier in the dataset.
        Args:
            path_to_direct (str): path to the directory containing the direct radiation data
        """
        df = self.data.copy()
        glaciers = df['GLACIER'].unique()
        df_concat = pd.DataFrame()

        for glacierName in glaciers:
            df_glacier = df[df['GLACIER'] == glacierName]
            path_to_file = path_to_direct + f'xr_direct_{glacierName}.zarr'
            df_glacier = retrieve_clear_sky_rad(df_glacier, path_to_file)
            df_concat = pd.concat([df_concat, df_glacier], axis=0)

        # reset index
        df_concat.reset_index(drop=True, inplace=True)
        self.data = df_concat

    def remove_climate_artifacts(self, vois_climate: str,
                                 vois_other: str) -> None:
        """For big glaciers covered by more than one ERA5-Land grid cell, the
        climate data is the one with the most data points. This function smooths the
        climate data by taking the mode of the data for each grid cell. 

        Args:
            vois_climate (str): A string containing the climate variables of interest
        """
        self.data = smooth_era5land_by_mode(self.data, vois_climate,
                                            vois_other)

    def convert_to_monthly(self,
                           *,
                           vois_climate: "list[str]",
                           vois_topographical: "list[str]",
                           meta_data_columns: "list[str]" = None) -> None:
        """
        Converts a variable period for the SMB target data measurement to a monthly time resolution.

        Args:
            vois_climate (list[str]): variables of interest from the climate data
            vois_topographical (list[str]): variables of interest from the topographical data
            meta_data_columns (list[str]): metadata columns
        """
        if meta_data_columns is None:
            meta_data_columns = self.cfg.metaData
        output_fname = self._get_output_filename("monthly_dataset")
        self.data = transform_to_monthly(self.data, meta_data_columns,
                                         vois_climate, vois_topographical,
                                         output_fname)

    def get_glacier_mask(
        self,
        custom_working_dir: str = ''
    ) -> "tuple[xr.Dataset, tuple[np.array, np.array], oggm.GlacierDirectory]":
        """Creates an xarray that contains different variables from OGGM,
            mapped over the glacier outline. The glacier mask is also returned.

        Args:
            custom_working_dir (str, optional): working directory for the OGGM data. Defaults to ''.
        Returns:
            ds (xr.Dataset): the glacier data from OGGM masked over the glacier outline
            glacier_indices (np.array): indices of glacier pixels in OGGM grid
            gdir (oggm.GlacierDirectory): the OGGM glacier directory

        """
        ds, glacier_indices, gdir = get_glacier_mask(self.data,
                                                     custom_working_dir,
                                                     self.cfg)
        return ds, glacier_indices, gdir

    def create_glacier_grid_RGI(self,
                                custom_working_dir: str = '') -> pd.DataFrame:
        """Creates a dataframe with the glacier grid data from RGI v.6,
            which contains the glacier data from OGGM mapped over the glacier outline in yearly format.

        Args:
            custom_working_dir (str, optional): working directory for the OGGM data. Defaults to ''.

        Returns:
            df_grid (pd.DataFrame): yearly dataframe with the glacier grid data.
        """
        # Get glacier mask from OGGM
        ds, glacier_indices, gdir = get_glacier_mask(self.data,
                                                     custom_working_dir,
                                                     self.cfg)
        # years_stake = self.data['YEAR'].unique()

        # Fixed time range because we want the grid from the beginning
        # of climate data to end (not just when there are stake measurements)
        years = range(1951, 2024)
        rgi_gl = self.data['RGIId'].unique()[0]
        df_grid = create_glacier_grid_RGI(ds, years, glacier_indices, gdir,
                                          rgi_gl)
        return df_grid

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
        # Fist, check if there are any faulty dates, if there are throw these
        # out.
        corrected_data = Dataset._check_dates(data=data)

        # Second, drop all records without a POINT_BALANCE (e.g. NaN)
        corrected_data.dropna(subset=["POINT_BALANCE"], inplace=True)

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
                          required_columns: "list[str]") -> None:
        """Validates that all required columns are present in the DataFrame."""
        if not all(col in data.columns for col in required_columns):
            logging.error(
                f"Missing one of the required columns: {required_columns}")
            raise KeyError(
                f"Required columns {required_columns} are not present in the DataFrame."
            )

    @staticmethod
    def _remove_missing_dates(data: pd.DataFrame,
                              date_columns: "list[str]") -> pd.DataFrame:
        """Removes rows with missing dates from the DataFrame."""
        return data.dropna(subset=date_columns)

    @staticmethod
    def _convert_and_filter_dates(data: pd.DataFrame) -> pd.DataFrame:
        """Converts date columns to string and filters out invalid dates."""
        data = data.astype({"FROM_DATE": str, "TO_DATE": str})
        return data[~data["FROM_DATE"].str.endswith("99")
                    & ~data["TO_DATE"].str.endswith("99")]


class AggregatedDataset(torch.utils.data.Dataset):
    """
    A dataset class that groups together all entries based on their IDs. The number
    of features of each element of the dataset is equal to the true number of
    features times the maximum number of elements that are assigned to the same ID
    in the original dataset. The size of the aggregated dataset is equal to the
    number of unique IDs.

    Attributes:
        cfg (config.Config): Configuration instance.
        features (np.ndarray): A numpy like array containing the features. Shape
            should be (nbPoints, nbFeatures).
        metadata (np.ndarray): A numpy like array containing the meta data. Shape
            should be (nbPoints, nbMetadata). Used for example to retrieve the ID
            of each stake measurement.
        metadataColumns (list): List containing the labels of each metadata column.
            If not specified, metadata fields of the configuration instance are used.
        targets (np.ndarray, optional): A numpy like array containing the targets.
    """

    def __init__(self,
                 cfg: config.Config,
                 features: np.ndarray,
                 metadata: np.ndarray,
                 metadataColumns: "list[str]" = None,
                 targets: np.ndarray = None) -> None:
        self.cfg = cfg
        self.features = features
        self.metadata = metadata
        self.metadataColumns = metadataColumns or self.cfg.metaData
        self.targets = targets
        self.ID = np.array([
            self.metadata[i][self.metadataColumns.index('ID')]
            for i in range(len(self.metadata))
        ])
        self.uniqueID = np.unique(self.ID)
        self.maxConcatNb = max(
            [len(np.argwhere(self.ID == id)[:, 0]) for id in self.uniqueID])
        self.nbFeatures = self.features.shape[1]
        self.norm = Normalizer({k: cfg.bnds[k] for k in cfg.featureColumns})

    def mapSplitsToDataset(
        self, splits: "list[tuple[np.ndarray, np.ndarray]]"
    ) -> "list[tuple[np.ndarray, np.ndarray]]":
        """
        Maps split indices (usually the result of DataLoader.get_cv_split) to the
        indices used by the AggregatedDataset class.

        Attributes:
            splits (list of tuple): List containing the splits indices for the cross
                validation groups

        Returns:
            list[tuple[np.ndarray, np.ndarray]]: List with the same number of tuples
                as the input. Each tuple contains numpy arrays which provide the
                corresponding indices the cross validation should use according to
                the input splits variable.
        """
        ret = []
        for split in splits:
            t = []
            for e in split:
                uniqueSelectedId = np.unique(self.ID[e])
                ind = np.argwhere(
                    self.uniqueID[None, :] == uniqueSelectedId[:, None])[:, 1]
                assert all(uniqueSelectedId == self.uniqueID[ind])
                t.append(ind)
            ret.append(tuple(t))
        return ret

    def __len__(self) -> int:
        return len(self.uniqueID)

    def _getInd(self, index):
        ind = np.argwhere(self.ID == self.uniqueID[index])[:, 0]
        months = self.metadata[ind][:, self.metadataColumns.index('MONTHS')]
        numMonths = [self.cfg.month_abbr_hydr[m] for m in months]
        ind = ind[np.argsort(
            numMonths)]  # Sort ind to get monthly data in chronological order
        return ind

    def __getitem__(self, index: int) -> tuple:
        ind = self._getInd(index)
        f = self.features[ind][:, :]
        f = self.norm.normalize(f)
        fpad = np.empty((self.maxConcatNb, self.nbFeatures))
        fpad.fill(np.nan)
        fpad[:f.shape[0], :] = f
        fpad = fpad.reshape(-1)
        if self.targets is None:
            return (fpad, )
        else:
            t = self.targets[ind][:]
            tpad = np.empty(self.maxConcatNb)
            tpad.fill(np.nan)
            tpad[:t.shape[0]] = t
            return fpad, tpad

    def indexToId(self, index):
        """Maps an index of the dataset to the ID of the stake measurement."""
        return self.uniqueID[index]

    def indexToMetadata(self, index):
        ind = self._getInd(index)
        return self.metadata[ind][:, :]


class Normalizer:
    """
    A normalizer class to normalize data based on lower and upper bounds.

    Attributes:
        bnds (dict of tuple): Dictionary where each key is the name of a feature and
            the two values in the tuple are respectively the lower and upper bounds.
    """

    def __init__(self, bnds: "dict[str, tuple[float, float]]") -> None:
        assert not np.isnan(list(bnds.values())).any(), "Bounds contain NaNs"
        self.bnds = bnds

    def _norm(self, data, lower_bnd, upper_bnd):
        return (data - lower_bnd) / (upper_bnd - lower_bnd)

    def _unorm(self, data, lower_bnd, upper_bnd):
        return data * (upper_bnd - lower_bnd) + lower_bnd

    def _map(self, x: Union[dict, torch.Tensor, np.ndarray],
             fct: Callable) -> Union[dict, torch.Tensor, np.ndarray]:
        if isinstance(x, dict):
            z = {}
            for k in x:
                z[k] = fct(x[k], self.bnds[k][0], self.bnds[k][1])
            return z
        elif isinstance(x, (torch.Tensor, np.ndarray)):
            assert x.shape[-1] == len(self.bnds), f"Size of the input to normalize is {x.shape} and it doesn't match the number of bounds defined in the Normalizer object which is {len(self.bnds)}"
            z = torch.zeros_like(x) if isinstance(
                x, torch.Tensor) else np.zeros_like(x)
            for i, k in enumerate(self.bnds):
                z[..., i] = fct(x[..., i], self.bnds[k][0], self.bnds[k][1])
            return z
        else:
            raise NotImplementedError(f"Type {type(x)} is not supported yet")

    def normalize(
        self, x: Union[dict, torch.Tensor, np.ndarray]
    ) -> Union[dict, torch.Tensor, np.ndarray]:
        """Normalize data."""
        return self._map(x, self._norm)

    def unnormalize(
        self, x: Union[dict, torch.Tensor, np.ndarray]
    ) -> Union[dict, torch.Tensor, np.ndarray]:
        """Unnormalize data, the opposite operation of normalize"""
        return self._map(x, self._unorm)


class SliceDatasetBinding(Dataset):
    def __init__(self, X:SliceDataset, y:SliceDataset=None) -> None:
        """
        Binding to a SliceDataset that allows providing training and validation
        datasets through the train_split argument of CustomNeuralNetRegressor.

        Arguments:
            X (SliceDataset): Features defined through a SliceDataset.
            y (SliceDataset): Targets defined through a SliceDataset.
        """
        assert isinstance(X, SliceDataset), "X must be a SliceDataset instance"
        assert y is None or isinstance(y, SliceDataset), "y must be a SliceDataset instance"
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        # If y is None, just return X
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]