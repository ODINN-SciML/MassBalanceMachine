"""
@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 04/06/2024
"""

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from massbalancemachine.data_processing import Dataset


class DataLoader:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.cv_split = None
        self.train_iterator = None
        self.test_iterator = None

    def set_train_test_split(self, *, test_size=.3, random_seed=42, shuffle=True):
        # Retrieve the data from the massbalancemachine dataset
        df = self.dataset.data

        # Get the indices of the data, that will be split
        indices = np.arange(len(df))

        # Split the dataset for training and testing
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_seed,
            shuffle=shuffle
        )

        # Create iterators
        train_iterator = iter(train_indices)
        test_iterator = iter(test_indices)

        # Set the instance parameters to their new values
        self.train_iterator = train_iterator
        self.test_iterator = test_iterator

    def get_cv_split(self, *, n_splits=5):
        # Retrieve the data from the massbalancemachine dataset
        indices_train_data = list(self.train_iterator)
        df = self.dataset.data.iloc[indices_train_data]

        years = df['YEAR']

        # Select features for training
        df_X = df.drop(['YEAR', 'POINT_BALANCE', 'RGIId'], axis=1)

        # Select the targets for training
        df_y = df[['POINT_BALANCE']]

        # Get arrays of features+metadata and targets
        X, y = df_X.values, df_y.values

        # Get glacier IDs from training dataset (in the order of which they appear in training dataset).
        # gp_s is an array with shape equal to the shape of X_train_s and y_train_s.
        glacier_ids = df['ID'].values

        # Use GroupKFold for splitting
        group_kf_s = GroupKFold(n_splits=n_splits)

        # Split into folds according to group by glacier ID
        splits = group_kf_s.split(X, y, glacier_ids)

        cv_split = {
            'glacier_ids': glacier_ids,
            'random_n_folds': splits,
            'space': self.__class__._get_custom_folds(df_X, 'space'),
            'time': self.__class__._get_custom_folds(df_X, 'time', years),
            'spacetime': self.__class__._get_custom_folds(df_X, 'spacetime',years)
        }

        self.cv_split = cv_split

    @staticmethod
    def _get_custom_folds(df, type_split, n_folds=5, years=None):
        X = None

        # Match the type of folds are desired
        match type_split:
            case 'space':
                X = df[['POINT_LAT', 'POINT_LON']]
            case 'time':
                X = years
            case 'spacetime':
                X = pd.concat([df, years], axis=1)

        # Perform clustering with DBSCAN
        epsilon = 0.01
        min_samples = 5
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        df['cluster_labels'] = dbscan.fit_predict(X)

        # Filter out noise points (cluster label -1 indicates noise in DBSCAN)
        df = df[df['cluster_labels'] != -1]

        # Create the folds
        n_folds = n_folds
        df['fold'] = -1
        stratified_kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        return stratified_kf
