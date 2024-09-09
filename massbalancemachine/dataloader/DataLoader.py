"""
The DataLoader class is part of the massbalancemachine package and is designed for handling dataset operations
for the different models that are available in the massbalancemachine package. It provides functionality for train-test splitting and cross-validation, specifically
tailored for glacier mass balance datasets.

Users can load their data into this class to prepare it for model training and testing. The class uses pandas
for data manipulation and scikit-learn for splitting operations.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 24/07/2024
"""

from typing import Any, Iterator, Tuple, List

import config

import numpy as np
import pandas as pd

from numpy import ndarray
from sklearn.model_selection import GroupKFold, KFold, GroupShuffleSplit


class DataLoader:
    """
    A class for loading and preprocessing glacier surface mass balance data for machine learning tasks.

    This class provides methods for splitting data into train and test sets, and creating
    cross-validation splits. It's designed to work with pandas DataFrames and maintains
    consistency across different splits by using iterators and seeds.

    Attributes:
        data (pd.DataFrame): The input dataset.
        n_splits (int): Number of splits for cross-validation.
        random_seed (int): Seed for random operations to ensure reproducibility.
        test_size (float): Proportion of the dataset to include in the test split.
        cv_split (tuple[list[tuple[ndarray, ndarray]]]): Stores cross-validation split information.
        train_indices (Iterator): Iterator for training data indices.
        test_indices (Iterator): Iterator for test data indices.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        meta_data_columns: List[str] = None,
        random_seed: int = None,
    ):
        """
        Initialize the DataLoader with the provided dataset.
        Args:
            data (pd.DataFrame): The input dataset to be processed.
            meta_data_columns (list): List of columns that contain metadata.
            random_seed (int): Seed for random operations to ensure reproducibility.
        """
        self.data = data
        self.n_splits = config.N_SPLITS
        self.random_seed = config.SEED
        self.test_size = config.TEST_SIZE
        self.cv_split = None
        self.train_indices = None
        self.test_indices = None
        self.meta_data_columns = config.META_DATA

    def set_train_test_split(
        self,
        *,
        test_size: float = None,
        # random_seed: int = None,
        shuffle: bool = True,
    ) -> Tuple[Iterator[Any], Iterator[Any]]:
        """
        Split the dataset into training and testing sets.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            shuffle (bool): Whether to shuffle the data before splitting.

        Returns:
            Tuple[Iterator[Any], Iterator[Any]]: Iterators for training and testing indices.
        """

        # Save the test size and random seed as attributes of the dataloader
        # object
        if test_size is not None:
            self.test_size = test_size

        # Create a train test set based on indices, not the actual data

        # Split data so that years of stakes are in the same group
        # I.e, one year of a stake is not split amongst test and train set

        # From the data get the features, targets, and glacier IDS
        X, y, glacier_ids, stake_meas_id = self._prepare_data_for_cv(
            self.data, self.meta_data_columns
        )
        gss = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=self.random_seed
        )
        train_indices, test_indices = next(gss.split(X, y, stake_meas_id))

        # Check that the intersection train and test ids is empty
        train_stake_meas_id = stake_meas_id[train_indices]
        test_stake_meas_id = stake_meas_id[test_indices]
        assert len(np.intersect1d(train_stake_meas_id, test_stake_meas_id)) == 0

        # Make it iterators and set as an attribute of the class
        self.train_indices = train_indices
        self.test_indices = test_indices

        return iter(self.train_indices), iter(self.test_indices)

    def set_custom_train_test_indices(
        self, train_indices: np.array, test_indices: np.array
    ):
        """
        Function to set custom training and testing indices.

        Args:
            train_indices (list): List of training indices.
            test_indices (list): List of testing indices.
        """
        self.train_indices = train_indices
        self.test_indices = test_indices

    def get_cv_split(
        self, *, n_splits: int = 5, type_fold: str = "random"
    ) -> "list[tuple[ndarray, ndarray]]":
        """
        Create a cross-validation split of the training data.

        This method orchestrates the process of creating a cross-validation split,
        using either a random or group-based strategy. For groups, it uses scikit-learn's GroupKFold to ensure that data from the same glacier is not split
        across different folds.

        Args:
            n_splits (int): Number of splits for cross-validation.
            type_fold (str): Type of cross-validation fold. Options are 'random', 'group-rgi', or 'group-meas-id'.

        Returns:
            tuple[list[tuple[ndarray, ndarray]]]: A dictionary containing glacier IDs and CV split information.

        Raises:
            ValueError: If train_iterator is None (i.e., if set_train_test_split hasn't been called).
        """

        # Save the number of splits as an attribute of this class
        self.n_splits = n_splits

        # Check if there is already a train iterator, this is needed to make
        # the splits for CV
        self._validate_train_iterator()

        # Based on the indices of the data, obtain the actual data
        train_data = self._get_train_data()

        # From the training data get the features, targets, and glacier IDS
        X, y, glacier_ids, stake_meas_id = self._prepare_data_for_cv(
            train_data, self.meta_data_columns
        )

        # Create the cross validation splits
        splits = self._create_group_kfold_splits(
            X, y, glacier_ids, stake_meas_id, type_fold
        )
        self.cv_split = splits

        return self.cv_split

    def get_train_test_indices(self):
        """Return the train and test indices."""
        return self.train_indices, self.test_indices

    def _validate_train_iterator(self) -> None:
        """Validate that the train_iterator has been set."""
        if self.train_indices is None:
            raise ValueError("train_iterator is None. Call set_train_test_split first.")

    def _get_train_data(self) -> pd.DataFrame:
        """Retrieve the training data using the train_iterator."""
        train_indices = self.train_indices
        return self.data.iloc[train_indices]

    @staticmethod
    def _prepare_data_for_cv(
        train_data: pd.DataFrame, meta_data_columns: list
    ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, np.array]:
        """Prepare the training data for cross-validation."""
        X = train_data.drop(meta_data_columns, axis=1)
        y = train_data["POINT_BALANCE"]
        glacier_ids = train_data["RGIId"].values
        stake_meas_id = train_data["ID"].values  # unique value per stake measurement
        return X, y, glacier_ids, stake_meas_id

    def _create_group_kfold_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        glacier_ids: np.ndarray,
        stake_meas_id: np.ndarray,
        type_fold: str,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create KFold splits based on the specified fold type."""
        fold_types = {
            "group-rgi": (GroupKFold, glacier_ids),
            "group-meas-id": (GroupKFold, stake_meas_id),
            "random": (KFold, None),
        }

        FoldClass, groups = fold_types.get(type_fold, (KFold, None))

        kf = FoldClass(n_splits=self.n_splits)

        if isinstance(kf, KFold) and type_fold == "random":
            kf.shuffle = True
            kf.random_state = self.random_seed

        split_args = [X, y, groups] if groups is not None else [X, y]
        return list(kf.split(*split_args))