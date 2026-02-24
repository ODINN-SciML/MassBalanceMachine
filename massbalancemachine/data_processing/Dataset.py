"""
The Dataset class is part of the massbalancemachine package and is designed for user data processing, preparing
the data for the model training and testing.

Users provide stake measurement data, which must be in a WGMS-like format (if not, please see the data preprocessing
notebook).

Date Created: 21/07/2024
"""

import os

import numpy as np
import xarray as xr
import oggm
import config
import logging
import pandas as pd
import torch
from skorch.helper import SliceDataset
from torch.utils.data import DataLoader, Subset
from torch.utils.data import WeightedRandomSampler, SubsetRandomSampler

import random as rd
from typing import Union, Callable, Dict, List, Optional, Tuple
from collections import Counter
from tqdm import tqdm

from data_processing.get_climate_data import (
    get_climate_features_,
    retrieve_clear_sky_rad,
    smooth_era5land_by_mode,
)
from data_processing.get_topo_data import get_topographical_features, get_glacier_mask
from data_processing.transform_to_monthly import transform_to_monthly
from data_processing.glacier_utils import create_glacier_grid_RGI
from data_processing.climate_data_download import (
    download_climate_ERA5,
    path_climate_data,
)
from data_processing.utils import _rebuild_month_index, _compute_head_tail_pads_from_df
import config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Dataset:
    """
    A dataset class that retrieves both the climate and topography data, adds them to the dataset and
    transforms the data to a monthly time resolution.

    Attributes:
        cfg (config.Config): Configuration instance.
        data (pd.DataFrame): A pandas dataframe containing the raw data
        region (str): The name of the region, for saving the files accordingly
        region_id (str): The region ID, for saving the files accordingly and eventually downloading them if needed
        data_dir (str): Path to the directory containing the raw data, and save intermediate results
        RGIIds (pd.Series): Series of RGI IDs from the data
        months_tail_pad (list of str): Months to pad the start of the hydrological year
        months_head_pad (list of str): Months to pad the end of the hydrological year
    """

    def __init__(
        self,
        cfg: config.Config,
        data: pd.DataFrame,
        region_name: str,
        region_id: int,
        data_path: str,
        months_tail_pad=None,  #: List[str] = ['aug_', 'sep_'], # before 'oct'
        months_head_pad=None,  #: List[str] = ['oct_'], # after 'sep'
    ):
        self.cfg = cfg
        self.data = self._clean_data(data=data.copy())
        self.region = region_name
        self.region_id = region_id
        self.data_dir = data_path
        self.RGIIds = self.data["RGIId"]
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

        # Padding to allow for flexible month ranges (customize freely)
        assert (months_head_pad is None) == (
            months_tail_pad is None
        ), "If any of months_head_pad or months_tail_pad is provided, the other variable must also be provided."
        if months_head_pad is None and months_tail_pad is None:
            months_head_pad, months_tail_pad = _compute_head_tail_pads_from_df(
                self.data
            )
        self.months_head_pad = months_head_pad
        self.months_tail_pad = months_tail_pad

    def get_topo_features(self, vois: list[str], custom_working_dir: str = "") -> None:
        """
        Fetches all the topographical data, for a list of variables of interest, using OGGM for the specified RGI IDs

        Args:
            vois (list[str]): A string containing the topographical variables of interest
            custom_working_dir (str, optional): The path to the custom working directory for OGGM data. Default to ''
        """
        output_fname = self._get_output_filename("topographical_features")
        self.data = get_topographical_features(
            self.data, output_fname, vois, self.RGIIds, custom_working_dir, self.cfg
        )

    def get_climate_features(
        self,
        climate_data: str = None,
        geopotential_data: str = None,
        change_units: bool = False,
        smoothing_vois: dict = None,
    ) -> None:
        """
        Fetches all the climate data, for a list of variables of interest, for the specified RGI IDs.

        Args:
            climate_data (str): A netCDF-3 file location containing the climate data for the region of interest.
                Defaults to `None` which automatically downloads the file if necessary using the CDSAPI.
            geopotential_data (str): A netCDF-3 file location containing the geopotential data.
                Defaults to `None` which automatically downloads the file if necessary using the CDSAPI.
            change_units (bool, optional): A boolean indicating whether to change the units of the climate data. Default to False.
            smoothing_vois (dict, optional): A dictionary containing the variables of interest for smoothing climate artifacts. Default to None.
        """
        output_fname = self._get_output_filename("climate_features")

        smoothing_vois = smoothing_vois or {}  # Safely default to empty dict
        vois_climate = smoothing_vois.get("vois_climate")
        vois_other = smoothing_vois.get("vois_other")

        local_path = path_climate_data(self.region_id)
        assert (climate_data is None) == (
            geopotential_data is None
        ), "When climate_data is provided, geopotential_data should also be provided."
        if climate_data is None:
            climate_data = local_path + "era5_monthly_averaged_data.nc"
            geopotential_data = local_path + "era5_geopotential_pressure.nc"
            if not (os.path.isfile(climate_data) and os.path.isfile(geopotential_data)):
                download_climate_ERA5(self.region_id)

        self.data = get_climate_features_(
            self.data,
            output_fname,
            climate_data,
            geopotential_data,
            change_units,
            self.months_tail_pad,
            self.months_head_pad,
            vois_climate,
            vois_other,
        )

    def get_potential_rad(self, path_to_direct: str) -> None:
        """
        Fetch monthly clear-sky radiation for each glacier and add padded-month
        columns according to self.months_tail_pad / self.months_head_pad.

        Parameters
        ----------
        path_to_direct : str
            Directory containing 'xr_direct_{GLACIER}.zarr' files.
        """
        df = self.data.copy()
        glaciers = df["GLACIER"].unique()
        chunks = []

        for glacier in glaciers:
            sub = df[df["GLACIER"] == glacier].copy()
            zarr_path = os.path.join(path_to_direct, f"xr_direct_{glacier}.zarr")
            # User-provided function that merges clear-sky rad into sub:
            sub = retrieve_clear_sky_rad(sub, zarr_path)
            chunks.append(sub)

        df_concat = pd.concat(chunks, axis=0, ignore_index=True)

        # Create padded month columns for potential radiation (pcsr)
        df_concat = self._copy_padded_month_columns(df_concat, prefixes=("pcsr",))

        # Save back
        self.data = df_concat

    def remove_climate_artifacts(self, vois_climate: str, vois_other: str) -> None:
        """For big glaciers covered by more than one ERA5-Land grid cell, the
        climate data is the one with the most data points. This function smooths the
        climate data by taking the mode of the data for each grid cell.

        Args:
            vois_climate (str): A string containing the climate variables of interest
        """
        self.data = smooth_era5land_by_mode(self.data, vois_climate, vois_other)

    def convert_to_monthly(
        self,
        vois_climate: list[str],
        vois_topographical: list[str],
        meta_data_columns: list[str] = None,
    ) -> None:
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
        self.data = transform_to_monthly(
            self.data, meta_data_columns, vois_climate, vois_topographical, output_fname
        )

    def get_glacier_mask(
        self, custom_working_dir: str = ""
    ) -> tuple[xr.Dataset, tuple[np.array, np.array], oggm.GlacierDirectory]:
        """Creates an xarray that contains different variables from OGGM,
            mapped over the glacier outline. The glacier mask is also returned.

        Args:
            custom_working_dir (str, optional): working directory for the OGGM data. Defaults to ''.
        Returns:
            ds (xr.Dataset): the glacier data from OGGM masked over the glacier outline
            glacier_indices (np.array): indices of glacier pixels in OGGM grid
            gdir (oggm.GlacierDirectory): the OGGM glacier directory

        """
        ds, glacier_indices, gdir = get_glacier_mask(
            self.data, custom_working_dir, self.cfg
        )
        return ds, glacier_indices, gdir

    def create_glacier_grid_RGI(self, custom_working_dir: str = "") -> pd.DataFrame:
        """Creates a dataframe with the glacier grid data from RGI v.6,
            which contains the glacier data from OGGM mapped over the glacier outline in yearly format.

        Args:
            custom_working_dir (str, optional): working directory for the OGGM data. Defaults to ''.

        Returns:
            df_grid (pd.DataFrame): yearly dataframe with the glacier grid data.
        """
        # Get glacier mask from OGGM
        ds, glacier_indices, gdir = get_glacier_mask(
            self.data, custom_working_dir, self.cfg
        )
        # years_stake = self.data['YEAR'].unique()

        # Fixed time range because we want the grid from the beginning
        # of climate data to end (not just when there are stake measurements)
        years = range(1951, 2024)
        rgi_gl = self.data["RGIId"].unique()[0]
        df_grid = create_glacier_grid_RGI(ds, years, glacier_indices, gdir, rgi_gl)
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

    def _copy_padded_month_columns(
        self, df: pd.DataFrame, prefixes=("pcsr",), overwrite: bool = False
    ) -> pd.DataFrame:
        """
        For each padding token (e.g. '_aug_', '_sep_', 'oct_'),
        create a new column like 'pcsr__aug_' by copying from the base column
        'pcsr_aug'. Works for any variable names given in `prefixes`.
        """
        df = df.copy()
        padded_tokens = list(self.months_tail_pad) + list(self.months_head_pad)

        if not padded_tokens:
            return df  # nothing to do

        for token in padded_tokens:
            base = token.strip("_")  # e.g. '_aug_' -> 'aug', 'oct_' -> 'oct'
            for pref in prefixes:
                src = f"{pref}_{base}"
                dst = f"{pref}_{token}"
                if (dst in df.columns) and not overwrite:
                    continue
                if src in df.columns:
                    df[dst] = df[src].values
                else:
                    df[dst] = np.nan
        return df

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
    def _validate_columns(data: pd.DataFrame, required_columns: list[str]) -> None:
        """Validates that all required columns are present in the DataFrame."""
        if not all(col in data.columns for col in required_columns):
            logging.error(f"Missing one of the required columns: {required_columns}")
            raise KeyError(
                f"Required columns {required_columns} are not present in the DataFrame."
            )

    @staticmethod
    def _remove_missing_dates(
        data: pd.DataFrame, date_columns: list[str]
    ) -> pd.DataFrame:
        """Removes rows with missing dates from the DataFrame."""
        return data.dropna(subset=date_columns)

    @staticmethod
    def _convert_and_filter_dates(data: pd.DataFrame) -> pd.DataFrame:
        """Converts date columns to string and filters out invalid dates."""
        data = data.astype({"FROM_DATE": str, "TO_DATE": str})
        return data[
            ~data["FROM_DATE"].str.endswith("99") & ~data["TO_DATE"].str.endswith("99")
        ]


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

    def __init__(
        self,
        cfg: config.Config,
        features: np.ndarray,
        metadata: np.ndarray,
        months_head_pad: list[str],
        months_tail_pad: list[str],
        metadataColumns: list[str] = None,
        targets: np.ndarray = None,
    ) -> None:
        self.cfg = cfg
        self.features = features
        self.metadata = metadata
        self.metadataColumns = metadataColumns or self.cfg.metaData
        self.targets = targets

        assert len(self.features) > 0, "The features variable is empty."

        _, self.month_pos = _rebuild_month_index(months_head_pad, months_tail_pad)

        self.ID = np.array(
            [
                self.metadata[i][self.metadataColumns.index("ID")]
                for i in range(len(self.metadata))
            ]
        )
        self.uniqueID = np.unique(self.ID)
        self.maxConcatNb = max(
            [len(np.argwhere(self.ID == id)[:, 0]) for id in self.uniqueID]
        )
        self.nbFeatures = self.features.shape[1]
        self.nbMetadata = self.metadata.shape[1]
        self.norm = Normalizer({k: cfg.bnds[k] for k in cfg.featureColumns})

    def mapSplitsToDataset(
        self, splits: list[tuple[np.ndarray, np.ndarray]]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Maps split indices (usually the result of DataLoader.get_cv_split) to the
        indices used by the AggregatedDataset class.

        Args:
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
                ind = np.argwhere(self.uniqueID[None, :] == uniqueSelectedId[:, None])[
                    :, 1
                ]
                assert all(uniqueSelectedId == self.uniqueID[ind])
                t.append(ind)
            ret.append(tuple(t))
        return ret

    def __len__(self) -> int:
        return len(self.uniqueID)

    def _getInd(self, index):
        ind = np.argwhere(self.ID == self.uniqueID[index])[:, 0]
        months = self.metadata[ind][:, self.metadataColumns.index("MONTHS")]
        numMonths = [self.month_pos[m] for m in months]
        ind = ind[
            np.argsort(numMonths)
        ]  # Sort ind to get monthly data in chronological order
        return ind

    def __getitem__(self, index: int) -> tuple:
        ind = self._getInd(index)
        f = self.features[ind][:, :]
        f = self.norm.normalize(f)
        fpad = np.empty((self.maxConcatNb, self.nbFeatures))  # Features padded
        fpad.fill(np.nan)
        fpad[: f.shape[0], :] = f
        fpad = fpad.reshape(-1)
        if self.targets is None:
            return (fpad,)
        else:
            t = self.targets[ind][:]
            tpad = np.empty(self.maxConcatNb)  # Target padded
            tpad.fill(np.nan)
            tpad[: t.shape[0]] = t
            m = self.metadata[ind][:, :]
            mpad = np.empty(
                (self.maxConcatNb, self.nbMetadata), dtype=self.metadata.dtype
            )  # Metadata padded
            mpad.fill(np.nan)
            mpad[: m.shape[0], :] = m
            mpad = mpad.reshape(-1)
            return fpad, tpad, mpad

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

    def __init__(self, bnds: dict[str, tuple[float, float]]) -> None:
        assert not np.isnan(list(bnds.values())).any(), "Bounds contain NaNs"
        self.bnds = bnds

    def _norm(self, data, lower_bnd, upper_bnd):
        return (data - lower_bnd) / (upper_bnd - lower_bnd)

    def _unorm(self, data, lower_bnd, upper_bnd):
        return data * (upper_bnd - lower_bnd) + lower_bnd

    def _map(
        self, x: Union[dict, torch.Tensor, np.ndarray], fct: Callable
    ) -> Union[dict, torch.Tensor, np.ndarray]:
        if isinstance(x, dict):
            z = {}
            for k in x:
                z[k] = fct(x[k], self.bnds[k][0], self.bnds[k][1])
            return z
        elif isinstance(x, (torch.Tensor, np.ndarray)):
            assert x.shape[-1] == len(
                self.bnds
            ), f"Size of the input to normalize is {x.shape} and it doesn't match the number of bounds defined in the Normalizer object which is {len(self.bnds)}"
            z = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
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

    def __init__(
        self,
        X: SliceDataset,
        y: SliceDataset = None,
        M: SliceDataset = None,
        metadataColumns: list[str] = None,
    ) -> None:
        """
        Binding to a SliceDataset that allows providing training and validation
        datasets through the train_split argument of CustomNeuralNetRegressor.

        Arguments:
            X (SliceDataset): Features defined through a SliceDataset.
            y (SliceDataset): Targets defined through a SliceDataset.
            M (SliceDataset): Metadata defined through a SliceDataset.
        """
        assert isinstance(X, SliceDataset), "X must be a SliceDataset instance"
        assert y is None or isinstance(
            y, SliceDataset
        ), "y must be a SliceDataset instance"
        assert M is None or isinstance(
            M, SliceDataset
        ), "M must be a SliceDataset instance"
        assert (M is None) == (
            metadataColumns is None
        ), "If M or metadataColumns is provided, the other variable must be provided too."
        self.X = X
        self.y = y
        self.M = M
        self.metadataColumns = metadataColumns

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]

    def getMetadata(self, idx):
        return self.M[idx]


# ---------- LSTM Dataset ----------
class MBSequenceDataset(Dataset):
    """
    Dataset for glacier mass-balance sequences.
    Provides:
      - MBSequenceDataset.from_dataframe(...) -> builds sequences from a tidy monthly table
      - Scaling helpers (fit_scalers / transform_inplace / set_scalers_from)
      - Access to .keys [(GLACIER, YEAR, ID, PERIOD)] aligned with row order
    """

    # ---------- Constructors ----------
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        monthly_cols: List[str],
        static_cols: List[str],
        show_progress: bool = True,
        expect_target: bool = True,
        months_tail_pad=None,
        months_head_pad=None,
        probe_cols: Optional[List[str]] = None,
        normalize_target: bool = True,
    ) -> "MBSequenceDataset":
        """
        Build a dataset directly from a monthly table.

        Assumes MONTHS are already normalized to {'oct','nov','dec','jan','feb','mar','apr','may','jun','jul','aug','sep'}.
        Required columns: GLACIER, YEAR, ID, PERIOD, MONTHS, monthly_cols, static_cols,
        and POINT_BALANCE if expect_target=True.
        """

        # Padding to allow for flexible month ranges (customize freely)
        assert (months_head_pad is None) == (
            months_tail_pad is None
        ), "If any of months_head_pad or months_tail_pad is provided, the other variable must also be provided."

        try:
            if months_head_pad is None and months_tail_pad is None:
                months_head_pad, months_tail_pad = _compute_head_tail_pads_from_df(df)
        except AttributeError as e:
            raise ValueError(
                "Could not compute months_head_pad / months_tail_pad from dataframe. Please provide them explicitly as arguments in function from_dataframe."
            ) from e

        month_list, month_pos = _rebuild_month_index(months_head_pad, months_tail_pad)

        pos_map = {k: v - 1 for k, v in month_pos.items()}  # token -> 0-based index
        T = int(len(month_list))  # max sequence length

        data_dict = cls._build_sequences(
            df=df,
            monthly_cols=monthly_cols,
            static_cols=static_cols,
            pos_map=pos_map,
            T=T,
            show_progress=show_progress,
            expect_target=expect_target,
            # NEW:
            probe_cols=probe_cols,
        )
        return cls(data_dict, normalize_target=normalize_target)

    def make_loaders(
        self,
        train_idx,
        val_idx,
        batch_size_train: int = 64,
        batch_size_val: int = 158,
        seed: int = 42,
        fit_and_transform: bool = True,
        shuffle_train: bool = True,
        drop_last_train: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        use_weighted_sampler: bool = False,
        verbose: bool = True,
    ):
        """
        Split this dataset into train/val, (optionally) fit+apply scalers on TRAIN,
        and return DataLoaders plus the split indices.

        Parameters
        ----------
        use_weighted_sampler : bool, default False
            If True, uses WeightedRandomSampler for the training DataLoader to
            balance winter/annual samples.

        Returns
        -------
        train_dl, val_dl, train_idx, val_idx
        """

        # Reproducible sampling
        g = torch.Generator()
        g.manual_seed(seed)
        self.seed_all(seed)

        def _seed_worker(worker_id):
            # Set seed for Python and NumPy in each worker
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            rd.seed(worker_seed)
            torch.manual_seed(worker_seed)

        if fit_and_transform:
            self.fit_scalers(train_idx)
            self.transform_inplace()
            for p in self.probe_names:
                v = self.probes[p].detach().cpu().numpy()
                print(p, "nanmean=", np.nanmean(v), "nanstd=", np.nanstd(v))

        train_ds = Subset(self, train_idx)
        val_ds = Subset(self, val_idx)

        if use_weighted_sampler:
            # Compute weights to balance winter/annual samples
            iw = self.iw[train_idx].numpy().astype(bool)
            ia = self.ia[train_idx].numpy().astype(bool)
            n_w, n_a = int(iw.sum()), int(ia.sum())

            # If one class is missing, WeightedRandomSampler may produce all-zero weights
            # (e.g. n_w=0 -> w_a=0 and if all samples are annual => sum(weights)=0).
            if (n_w == 0) or (n_a == 0):
                if verbose:
                    print(
                        f"Weighted sampler disabled (one class missing): "
                        f"{n_w} winter | {n_a} annual. Using shuffle instead."
                    )
                train_dl = DataLoader(
                    train_ds,
                    batch_size=batch_size_train,
                    shuffle=shuffle_train,
                    drop_last=drop_last_train,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    worker_init_fn=_seed_worker,
                    generator=g,
                )
            else:
                # Upweight the minority class (annual) relative to winter
                w_w = 1.0
                w_a = n_w / n_a  # > 0 because both are > 0

                sample_weights = np.where(ia, w_a, w_w).astype(np.float32)

                # Safety: ensure valid distribution
                sw_sum = float(sample_weights.sum())
                if not np.isfinite(sw_sum) or sw_sum <= 0.0:
                    if verbose:
                        print(
                            "Weighted sampler disabled (invalid weights distribution). "
                            "Using shuffle instead."
                        )
                    train_dl = DataLoader(
                        train_ds,
                        batch_size=batch_size_train,
                        shuffle=shuffle_train,
                        drop_last=drop_last_train,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        worker_init_fn=_seed_worker,
                        generator=g,
                    )
                else:
                    sample_weights = torch.from_numpy(sample_weights)

                    sampler = WeightedRandomSampler(
                        sample_weights,
                        num_samples=len(sample_weights),
                        replacement=True,
                        generator=g,
                    )

                    train_dl = DataLoader(
                        train_ds,
                        batch_size=batch_size_train,
                        sampler=sampler,
                        drop_last=drop_last_train,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        worker_init_fn=_seed_worker,
                        generator=g,
                    )
        else:
            train_dl = DataLoader(
                train_ds,
                batch_size=batch_size_train,
                shuffle=shuffle_train,
                drop_last=drop_last_train,
                num_workers=num_workers,
                pin_memory=pin_memory,
                worker_init_fn=_seed_worker,
                generator=g,
            )

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size_val,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=_seed_worker,
            generator=g,
        )

        # ---- Sanity check printout ----
        n_w_tr, n_a_tr = int(self.iw[train_idx].sum()), int(self.ia[train_idx].sum())
        n_w_va, n_a_va = int(self.iw[val_idx].sum()), int(self.ia[val_idx].sum())
        if verbose:
            print(f"Train counts: {n_w_tr} winter | {n_a_tr} annual")
            print(f"Val   counts: {n_w_va} winter | {n_a_va} annual")

        return train_dl, val_dl

    @staticmethod
    def seed_all(seed=None):
        # Python
        rd.seed(seed)
        # NumPy
        np.random.seed(seed)
        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # cuDNN deterministic kernels
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Forbid nondeterministic ops (warn if an op has no deterministic impl)
        torch.use_deterministic_algorithms(True, warn_only=True)

        # Setting CUBLAS environment variable (helps in newer versions)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

    @staticmethod
    def make_test_loader(
        ds_test: "MBSequenceDataset",
        ds_train: "MBSequenceDataset",
        *,
        seed,
        batch_size: int = 158,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        """
        Copy TRAIN scalers to TEST, transform TEST in-place, and return a DataLoader.
        """
        g = torch.Generator()
        g.manual_seed(seed)

        ds_test.normalize_target = ds_train.normalize_target  # <--- propagate flag
        ds_test.set_scalers_from(ds_train)
        ds_test.transform_inplace()

        def _seed_worker(worker_id):
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            rd.seed(worker_seed)

        test_dl = DataLoader(
            ds_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=_seed_worker,
            generator=g,
        )
        return test_dl

    @staticmethod
    def _stack(a: List[np.ndarray]) -> np.ndarray:
        return np.stack(a, axis=0) if len(a) else np.empty((0,))

    @staticmethod
    def _build_sequences(
        df: pd.DataFrame,
        monthly_cols: List[str],
        static_cols: List[str],
        pos_map: Dict[str, int],
        T: int,
        show_progress: bool = True,
        expect_target: bool = True,
        # NEW:
        probe_cols: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:

        probe_cols = list(probe_cols) if probe_cols else []
        # monthly inputs remain ONLY the trained inputs:
        #   X_monthly is built from monthly_cols (unchanged)
        # we will aggregate probe_cols to parallel (T,) arrays, one per col

        req = {"GLACIER", "YEAR", "ID", "PERIOD", "MONTHS", *monthly_cols, *static_cols}
        if expect_target:
            req |= {"POINT_BALANCE"}
        # Require probe cols to be present in df (but not part of inputs)
        req |= set(probe_cols)

        missing = req - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")

        df = df.copy()
        df["PERIOD"] = df["PERIOD"].astype(str).str.strip().str.lower()
        df["MONTHS"] = df["MONTHS"].astype(str).str.strip().str.lower()

        X_monthly, X_static = [], []
        mask_valid, mask_w, mask_a = [], [], []
        y, is_winter, is_annual, keys = [], [], [], []

        groups = list(df.groupby(["GLACIER", "YEAR", "ID", "PERIOD"], sort=False))
        iterator = tqdm(groups, desc="Building sequences") if show_progress else groups

        agg_cols = (
            monthly_cols
            + static_cols
            + probe_cols
            + (["POINT_BALANCE"] if expect_target else [])
        )

        probe_buffers = {p: [] for p in probe_cols}  # will hold list of (T,) per sample
        for (g, yr, mid, per), sub in iterator:
            # average duplicates within the same month token
            subm = sub.groupby("MONTHS", as_index=False)[agg_cols].mean(
                numeric_only=True
            )

            # (T, Fm) matrix + valid mask
            mat = np.zeros((T, len(monthly_cols)), dtype=np.float32)
            mv = np.zeros(T, dtype=np.float32)

            # Initialize probe arrays per sample
            probe_per_sample = {
                p: np.full(T, np.nan, dtype=np.float32) for p in probe_cols
            }

            for _, r in subm.iterrows():
                m = str(r["MONTHS"]).strip().lower()
                if m not in pos_map:
                    raise ValueError(
                        f"Unexpected month token '{m}'. Expected one of {list(pos_map.keys())}."
                    )
                pos = int(pos_map[m])  # 0-based

                # fill inputs
                mat[pos, :] = r[monthly_cols].to_numpy(np.float32)
                mv[pos] = 1.0

                # fill probes for this month
                for p in probe_cols:
                    probe_per_sample[p][pos] = np.float32(r[p])

            # static features
            s = subm.iloc[0][static_cols].to_numpy(np.float32)

            # target
            target = float(subm["POINT_BALANCE"].mean()) if expect_target else np.nan

            # Currently this function doesn't work without POINT_BALANCE,
            # so we set a dummy value for mask logic below
            # Need to fix this at some point...
            if not expect_target:
                subm["POINT_BALANCE"] = 0.0  # dummy for mask logic below

            # ---- per-sample seasonal masks (flexible windows) ----
            per_l = str(per).strip().lower()

            # ---- Seasonal loss masks ----
            loss_mask = np.zeros(T)
            for _, r in subm.iterrows():
                if not np.isnan(r["POINT_BALANCE"]):
                    loss_mask[pos_map[r["MONTHS"]]] = 1

            if per_l == "winter":
                mw_sample = loss_mask
                ma_sample = np.zeros(T)
            elif per_l == "annual":
                mw_sample = np.zeros(T)
                ma_sample = loss_mask

            # append once per group
            X_monthly.append(mat)
            X_static.append(s)
            mask_valid.append(mv)
            mask_w.append(mw_sample)
            mask_a.append(ma_sample)
            y.append(target)
            is_winter.append(per_l == "winter")
            is_annual.append(per_l == "annual")
            keys.append((g, int(yr), int(mid), per_l))

            for p in probe_cols:
                probe_buffers[p].append(probe_per_sample[p])

        def stack(a):
            return np.stack(a, axis=0) if len(a) else np.empty((0,))

        data_dict = dict(
            X_monthly=stack(X_monthly),
            X_static=stack(X_static),
            mask_valid=stack(mask_valid),
            mask_w=stack(mask_w),
            mask_a=stack(mask_a),
            y=np.asarray(y, dtype=np.float32),
            is_winter=np.asarray(is_winter, dtype=bool),
            is_annual=np.asarray(is_annual, dtype=bool),
            keys=keys,
        )

        # add probes
        for p in probe_cols:
            data_dict[f"probe__{p}"] = stack(probe_buffers[p])  # shape (B,T)

        # Key uniqueness check
        if len(keys) != len(set(keys)):
            from collections import Counter

            dupes = [k for k, c in Counter(keys).items() if c > 1]
            raise ValueError(f"Found {len(dupes)} duplicate keys, e.g. {dupes[:5]}")

        return data_dict

    # ---------- Torch Dataset API ----------

    def __init__(self, data_dict: Dict[str, np.ndarray], normalize_target: bool = True):
        self.normalize_target = normalize_target

        # raw numpy -> tensors
        self.Xm = torch.from_numpy(data_dict["X_monthly"]).float()  # (B,T,Fm)
        self.Xs = torch.from_numpy(data_dict["X_static"]).float()  # (B,Fs)
        self.mv = torch.from_numpy(data_dict["mask_valid"]).float()  # (B,T)
        self.mw = torch.from_numpy(data_dict["mask_w"]).float()  # (B,T)
        self.ma = torch.from_numpy(data_dict["mask_a"]).float()  # (B,T)
        self.y = torch.from_numpy(data_dict["y"]).float()  # (B,)
        self.iw = torch.from_numpy(data_dict["is_winter"]).bool()  # (B,)
        self.ia = torch.from_numpy(data_dict["is_annual"]).bool()  # (B,)
        self.keys = data_dict.get("keys", [])

        # scalers (set by fit_scalers or set_scalers_from)
        self.month_mean: Optional[torch.Tensor] = None
        self.month_std: Optional[torch.Tensor] = None
        self.static_mean: Optional[torch.Tensor] = None
        self.static_std: Optional[torch.Tensor] = None
        self.y_mean: Optional[torch.Tensor] = None
        self.y_std: Optional[torch.Tensor] = None

        # --- Probes (optional) ---
        self.probe_names: List[str] = []
        self.probes: Dict[str, torch.Tensor] = {}
        self.probe_mean: Dict[str, Optional[torch.Tensor]] = {}
        self.probe_std: Dict[str, Optional[torch.Tensor]] = {}

        for k, v in data_dict.items():
            if isinstance(k, str) and k.startswith("probe__"):
                pname = k.split("probe__")[1]
                self.probe_names.append(pname)
                self.probes[pname] = torch.from_numpy(v).float()  # (B,T)

    def __len__(self) -> int:
        return self.Xm.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        batch = {
            "x_m": self.Xm[idx],
            "x_s": self.Xs[idx],
            "mv": self.mv[idx],
            "mw": self.mw[idx],
            "ma": self.ma[idx],
            "y": self.y[idx],
            "iw": self.iw[idx],
            "ia": self.ia[idx],
        }
        for pname in self.probe_names:
            batch[pname] = self.probes[pname][idx]  # (T,)
        return batch

    # ---------- Scaling helpers ----------
    def fit_scalers(self, idx_train: np.ndarray) -> None:
        """Fit scalers on TRAIN subset only."""
        # monthly features: mean/std over valid months
        Xm = self.Xm[idx_train].numpy()  # (N,T,Fm)
        Mv = self.mv[idx_train].numpy()  # (N,T)
        mask3 = Mv[..., None]  # (N,T,1)
        num = (Xm * mask3).sum(axis=(0, 1))  # (Fm,)
        den = mask3.sum(axis=(0, 1))  # (Fm,)
        month_mean = num / np.maximum(den, 1e-8)
        var = (((Xm - month_mean) * mask3) ** 2).sum(axis=(0, 1)) / np.maximum(
            den, 1e-8
        )
        month_std = np.sqrt(np.maximum(var, 1e-8))

        # static features: simple mean/std per feature
        Xs = self.Xs[idx_train].numpy()
        static_mean = Xs.mean(axis=0)
        static_std = np.sqrt(np.maximum(Xs.var(axis=0), 1e-8))

        # target scaler
        y = self.y[idx_train].numpy()
        y_mean = float(np.mean(y))
        y_std = float(np.sqrt(max(np.var(y), 1e-8)))

        if self.normalize_target:
            self.y_mean = torch.tensor(y_mean, dtype=torch.float32)
            self.y_std = torch.tensor(y_std, dtype=torch.float32)
        else:
            # disable normalization
            self.y_mean = torch.tensor(0.0, dtype=torch.float32)
            self.y_std = torch.tensor(1.0, dtype=torch.float32)

        # store as tensors
        self.month_mean = torch.from_numpy(month_mean).float()
        self.month_std = torch.from_numpy(month_std).float()
        self.static_mean = torch.from_numpy(static_mean).float()
        self.static_std = torch.from_numpy(static_std).float()

        # Fitting probes (NaN-safe, masked by mv)
        Mv = self.mv[idx_train].numpy().astype(bool)
        self.probe_mean, self.probe_std = {}, {}
        for pname in self.probe_names:
            P = self.probes[pname][idx_train].numpy()
            valid = Mv & ~np.isnan(P)
            den = valid.sum()
            if den == 0:
                mu, std = 0.0, 1.0
            else:
                vals = P[valid]
                mu = float(vals.mean())
                var = float(((vals - mu) ** 2).mean())
                std = float(np.sqrt(max(var, 1e-8)))
            self.probe_mean[pname] = torch.tensor(mu, dtype=torch.float32)
            self.probe_std[pname] = torch.tensor(std, dtype=torch.float32)

    def transform_inplace(self) -> None:
        """Apply standardization to Xm, Xs, y using fitted scalers."""
        assert (
            self.month_mean is not None and self.month_std is not None
        ), "Call fit_scalers or set_scalers_from first."
        assert (
            self.static_mean is not None and self.static_std is not None
        ), "Call fit_scalers or set_scalers_from first."
        assert (
            self.y_mean is not None and self.y_std is not None
        ), "Call fit_scalers or set_scalers_from first."

        self.Xm = (self.Xm - self.month_mean) / self.month_std
        self.Xs = (self.Xs - self.static_mean) / self.static_std

        # Conditionally normalize target
        if self.normalize_target:
            self.y = (self.y - self.y_mean) / self.y_std

        # Transform probes
        for pname in self.probe_names:
            assert (
                pname in self.probe_mean and pname in self.probe_std
            ), f"Missing scaler for probe '{pname}'. Call fit_scalers/set_scalers_from first."
            self.probes[pname] = (
                self.probes[pname] - self.probe_mean[pname]
            ) / self.probe_std[pname]

    def set_scalers_from(self, other: "MBSequenceDataset") -> None:
        """Copy fitted scalers from another dataset (usually the train dataset)."""
        self.month_mean = other.month_mean.clone()
        self.month_std = other.month_std.clone()
        self.static_mean = other.static_mean.clone()
        self.static_std = other.static_std.clone()
        self.y_mean = other.y_mean.clone()
        self.y_std = other.y_std.clone()
        self.normalize_target = other.normalize_target

        # Copy probe scalers
        self.probe_mean = {k: v.clone() for k, v in other.probe_mean.items()}
        self.probe_std = {k: v.clone() for k, v in other.probe_std.items()}

    def _clone_untransformed_dataset(
        ds_src: "MBSequenceDataset",
    ) -> "MBSequenceDataset":
        # Require the source to be pristine (not standardized yet)
        if any(
            x is not None
            for x in (ds_src.month_mean, ds_src.static_mean, ds_src.y_mean)
        ):
            raise ValueError(
                "ds_pristine was already transformed. Build a fresh pristine dataset once and keep it read-only."
            )

        data_dict = dict(
            X_monthly=ds_src.Xm.detach().cpu().numpy().copy(),
            X_static=ds_src.Xs.detach().cpu().numpy().copy(),
            mask_valid=ds_src.mv.detach().cpu().numpy().copy(),
            mask_w=ds_src.mw.detach().cpu().numpy().copy(),
            mask_a=ds_src.ma.detach().cpu().numpy().copy(),
            y=ds_src.y.detach().cpu().numpy().copy(),
            is_winter=ds_src.iw.detach().cpu().numpy().copy(),
            is_annual=ds_src.ia.detach().cpu().numpy().copy(),
            keys=list(ds_src.keys),
        )

        # NEW: copy all probe arrays if present
        if hasattr(ds_src, "probe_names"):
            for pname in ds_src.probe_names:
                data_dict[f"probe__{pname}"] = (
                    ds_src.probes[pname].detach().cpu().numpy().copy()
                )
        return MBSequenceDataset(data_dict, normalize_target=ds_src.normalize_target)

    # ---------- Utilities ----------
    @staticmethod
    def _clone_for_permutation(ds_src: "MBSequenceDataset") -> "MBSequenceDataset":
        """
        Clone a *transformed* dataset for safe permutation.

        Keeps fitted scalers and normalized tensors intact.
        """
        data_dict = dict(
            X_monthly=ds_src.Xm.detach().cpu().numpy().copy(),
            X_static=ds_src.Xs.detach().cpu().numpy().copy(),
            mask_valid=ds_src.mv.detach().cpu().numpy().copy(),
            mask_w=ds_src.mw.detach().cpu().numpy().copy(),
            mask_a=ds_src.ma.detach().cpu().numpy().copy(),
            y=ds_src.y.detach().cpu().numpy().copy(),
            is_winter=ds_src.iw.detach().cpu().numpy().copy(),
            is_annual=ds_src.ia.detach().cpu().numpy().copy(),
            keys=list(ds_src.keys),
        )

        ds_new = MBSequenceDataset(data_dict, normalize_target=ds_src.normalize_target)

        # copy fitted scalers
        ds_new.month_mean = ds_src.month_mean.clone()
        ds_new.month_std = ds_src.month_std.clone()
        ds_new.static_mean = ds_src.static_mean.clone()
        ds_new.static_std = ds_src.static_std.clone()
        ds_new.y_mean = ds_src.y_mean.clone()
        ds_new.y_std = ds_src.y_std.clone()

        return ds_new

    @staticmethod
    def split_indices(
        n: int, val_ratio: float = 0.2, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = max(1, int(n * (1 - val_ratio)))
        return idx[:cut], idx[cut:]


# --------- Transfer Learning Dataset Wrapper ---------
class MBSequenceDatasetTL(Dataset):
    """
    Thin wrapper around MBSequenceDataset that injects domain labels (SOURCE_CODE/domain_id)
    into each batch item, without changing MBSequenceDataset.
    """

    def __init__(self, base_ds, source_codes, domain_vocab=None):
        """
        base_ds: MBSequenceDataset
        source_codes: list[str] length == len(base_ds) (one per sequence)
        domain_vocab: optional dict[str,int] OR list[str]
            - dict: {"CH":0,"NOR":1,...}
            - list: ["CH","NOR",...]
        """
        self.base = base_ds

        if len(source_codes) != len(base_ds):
            raise ValueError(
                f"source_codes length {len(source_codes)} != len(base_ds) {len(base_ds)}"
            )
        self.source_codes = list(source_codes)

        if domain_vocab is not None and not isinstance(domain_vocab, dict):
            # accept list/tuple as vocab
            domain_vocab = {sc: i for i, sc in enumerate(list(domain_vocab))}

        self.domain_vocab = domain_vocab

        if self.domain_vocab is not None:
            dom = []
            for sc in self.source_codes:
                if sc not in self.domain_vocab:
                    raise KeyError(f"SOURCE_CODE '{sc}' missing from domain_vocab")
                dom.append(int(self.domain_vocab[sc]))
            self.domain_id = dom  # store as plain ints
        else:
            self.domain_id = None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        b = dict(self.base[idx])
        b["source_code"] = self.source_codes[idx]
        if self.domain_id is not None:
            b["domain_id"] = self.domain_id[idx]  # python int => collates cleanly
        return b
