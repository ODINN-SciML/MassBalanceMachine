import numpy as np
import pandas as pd

from typing import Tuple
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split


def create_train_test_data(df: pd.DataFrame, test_size=.3, random_seed=42, shuffle=True) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Divide the dataset into 70/30 split for training and testing
    train_data, test_data = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        shuffle=shuffle
    )

    return train_data, test_data


def create_train_test_split(df: pd.DataFrame, n_splits=5) -> tuple[
    pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list]:
    # Select features for training
    df_X_features = df.drop(['YEAR', 'POINT_BALANCE'], axis=1)

    # Select the targets for training
    df_y_labels = df[['POINT_BALANCE']]

    # Get arrays of features+metadata and targets
    X_features, y_labels = df_X_features.values, df_y_labels.values

    # Get glacier IDs from training dataset (in the order of which they appear in training dataset).
    # gp_s is an array with shape equal to the shape of X_train_s and y_train_s.
    glacier_ids = df['ID'].values

    # Use GroupKFold for splitting
    group_kf_s = GroupKFold(n_splits=n_splits)

    # Split into folds according to group by glacier ID
    splits = list(group_kf_s.split(X_features, y_labels, glacier_ids))

    return df_X_features, df_y_labels, X_features, y_labels, splits
