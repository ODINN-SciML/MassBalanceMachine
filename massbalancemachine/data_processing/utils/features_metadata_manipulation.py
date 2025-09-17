from typing import Tuple
import numpy as np
import pandas as pd
from config import Config


def create_features_metadata(
    cfg: Config,
    X: pd.DataFrame,
    meta_data_columns: list = None,
) -> Tuple[np.array, np.ndarray]:
    """
    Split the input DataFrame into features and metadata.

    Args:
        cfg (mbm.Config): MBM configuration instance.
        X (pd.DataFrame): The input DataFrame containing both features and metadata.
        meta_data_columns (list): The metadata columns to be extracted. If not
            specified, metadata fields of the configuration instance will be used.

    Returns:
        tuple: A tuple containing:
            - features (array-like): The feature values.
            - metadata (array-like): The metadata values.
    """
    meta_data_columns = meta_data_columns or cfg.metaData
    feature_columns = cfg.featureColumns

    # Extract metadata and features
    metadata = X[meta_data_columns].values
    features = X[feature_columns].values

    return features, metadata
