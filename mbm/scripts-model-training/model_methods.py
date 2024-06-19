"""
This script contains a custom XGBoost regressor that incorporates metadata into the learning process,
along with functions for training and evaluating the model using a custom mean squared error function.

The custom loss function is by Kamilla Hauknes Sjursen.

@Author: Kamilla Heuknes Sjursen and adapted by Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 04/06/2024
"""

import xgboost as xgb
import numpy as np
from sklearn.utils.validation import check_is_fitted


def custom_mse_metadata(y_true, y_pred, metadata):
    """
    Custom Mean Squared Error (MSE) objective function for evaluating monthly predictions
    with respect to seasonally or annually aggregated observations.

    Parameters
    ----------
    y_true : numpy.ndarray
        True (aggregated) values for each instance.
    y_pred : numpy.ndarray
        Predicted values.
    metadata : numpy.ndarray
        Metadata for each instance. The first column represents the group ID for aggregation.

    Returns
    -------
    gradients : numpy.ndarray
        The gradient of the loss with respect to y_pred.
    hessians : numpy.ndarray
        The hessian of the loss with respect to y_pred, filled with ones for MSE.
    """
    # Initialize empty arrays for gradient and hessian
    gradients = np.zeros_like(y_pred)
    hessians = np.ones_like(y_pred)  # Ones in case of mse

    # Unique aggregation groups based on the aggregation ID
    unique_ids = np.unique(metadata[:, 0])

    # Loop over each unique ID to aggregate accordingly
    for uid in unique_ids:
        # Find indexes for the current aggregation group
        indexes = metadata[:, 0] == uid

        # Aggregate y_pred for the current group
        y_pred_agg = np.sum(y_pred[indexes])

        # True value is the same repeated value for the group, so we can use the mean
        y_true_mean = np.mean(y_true[indexes])

        # Compute gradients for the group based on the aggregated prediction
        gradient = y_pred_agg - y_true_mean
        gradients[indexes] = gradient

    return gradients, hessians



def get_ytrue_y_pred_agg(y_true, y_pred, X):
    """
    Get aggregated true and predicted values based on unique IDs in the metadata.

    Parameters
    ----------
    y_true : numpy.ndarray
        True target values.
    y_pred : numpy.ndarray
        Predicted target values.
    X : numpy.ndarray
        Input data with features and metadata.

    Returns
    -------
    numpy.ndarray
        Aggregated true values.
    numpy.ndarray
        Aggregated predicted values.
    """
    metadata = X[:, -3:]
    unique_ids = np.unique(metadata[:, 0])
    y_pred_agg_all = []
    y_true_mean_all = []

    # Loop over each unique ID to calculate MSE
    for uid in unique_ids:
        # Indexes for the current ID
        indexes = metadata[:, 0] == uid
        # Aggregate y_pred for the current ID
        y_pred_agg = np.sum(y_pred[indexes])
        y_pred_agg_all.append(y_pred_agg)
        # True value is the mean of true values for the group
        y_true_mean = np.mean(y_true[indexes])
        y_true_mean_all.append(y_true_mean)

    y_pred_agg_all_arr = np.array(y_pred_agg_all)
    y_true_mean_all_arr = np.array(y_true_mean_all)

    return y_true_mean_all_arr, y_pred_agg_all_arr


def get_aggregated_predictions(X, y, splits, model, season_month=12):
    """
    Get aggregated predictions and true values for each split.

    Parameters
    ----------
    X : numpy.ndarray
        Input data with features and metadata.
    y : numpy.ndarray
        True target values.
    splits : list of tuples
        List of (train_index, test_index) tuples for splitting the data.
    model : CustomXGBoostRegressor
        The model to be trained and used for prediction.
    season_month : int, optional
        The month to filter the data by, default is 12.

    Returns
    -------
    numpy.ndarray
        Aggregated true values.
    numpy.ndarray
        Aggregated predicted values.
    """
    y_pred_list = []
    y_true_list = []
    unique_ids_list = []

    for train_index, test_index in splits:
        # Split the data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Filter by season
        mask = X_test[:, -2] == season_month
        X_filtered = X_test[mask]
        y_true_filtered = y_test[mask]
        y_pred_filtered = y_pred[mask]

        # Extract metadata (assumed last three columns) and unique IDs
        metadata = X_filtered[:, -3:]
        unique_ids = np.unique(metadata[:, 0])

        for uid in unique_ids:
            # Indexes for the current ID
            indexes = metadata[:, 0] == uid

            # Aggregate predictions and true values for the current ID
            y_pred_agg = np.sum(y_pred_filtered[indexes])
            y_true_mean = np.mean(y_true_filtered[indexes])

            y_pred_list.append(y_pred_agg)
            y_true_list.append(y_true_mean)
            unique_ids_list.append(uid)

    y_pred_agg_all = np.array(y_pred_list)
    y_true_mean_all = np.array(y_true_list)

    return y_true_mean_all, y_pred_agg_all