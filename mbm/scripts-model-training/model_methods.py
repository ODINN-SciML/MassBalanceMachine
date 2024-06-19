"""
This script contains functions training the XGBoost model.

The custom loss function is by Kamilla Hauknes Sjursen

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 04/06/2024
"""

import xgboost as xgb
import numpy as np
import sklearn

from sklearn.utils.validation import check_is_fitted


def train_xgb_model(X, y, idc_list, params, scorer='neg_mean_squared_error', return_train=True):
    # Define model object
    xgb_model = xgb.XGBRegressor(tree_method='hist')

    # Set up grid search
    clf = sklearn.model_selection.GridSearchCV(
        xgb_model,
        params,
        cv=idc_list,    # Int or iterator (default for int is k_fold)
        verbose=True,   # Controls number of messages
        n_jobs=4,       # No. of parallel jobs
        scoring=scorer, # Can use multiple metrics
        refit=True,     # Default True. For multiple metric evaluation, refit must be str denoting scorer to be used
                        # to find the best parameters for refitting the estimator.
        return_train_score=return_train  # Default False. If False, cv_results_ will not include training scores.
    )

    # Fit model to folds
    clf.fit(X, y)

    # Model object with the best fitted parameters (** to unpack parameter dict)
    fitted_model = xgb.XGBRegressor(**clf.best_params_)

    # Obtain the cross-validation score for each splot with the fitted model
    cvl = sklearn.model_selection.cross_val_score(fitted_model, X, y, cv=idc_list, scoring='neg_mean_squared_error')

    return clf, fitted_model, cvl


# Custom objective function scikit learn api with metadata, to be used with custom XGBRegressor class

def custom_mse_metadata(y_true, y_pred, metadata):
    """
    Custom Mean Squared Error (MSE) objective function for evaluating monthly predictions with respect to
    seasonally or annually aggregated observations.

    For use in cases where predictions are done on a monthly timescale and need to be aggregated to be
    compared with the true aggregated seasonal or annual value. Aggregations are performed according to a
    unique ID provided by metadata. The function computes gradients and hessians
    used in gradient boosting methods, specifically for use with the XGBoost library's custom objective
    capabilities.

    Parameters
    ----------
    y_true : numpy.ndarray
        True (seasonally or annually aggregated) values for each instance. For a unique ID,
        values are repeated n_months times across the group, e.g. the annual mass balance for a group
        of 12 monthly predictions with the same unique ID is repeated 12 times. Before calculating the
        loss, the mean over the n unique IDs is taken.

    y_pred : numpy.ndarray
        Predicted monthly values. These predictions will be aggregated according to the
        unique ID before calculating the loss, e.g. 12 monthly predictions with the same unique ID is
        aggregated for evaluation against the true annual value.

    metadata : numpy.ndarray
        An ND numpy array containing metadata for each monthly prediction. The first column is mandatory
        and represents the ID of the aggregated group to which each instance belongs. Each group identified
        by a unique ID will be aggregated together for the loss calculation. The following columns in the
        metadata can include additional information for each instance that may be useful for tracking or further
        processing but are not used in the loss calculation, e.g. number of months to be aggregated or the name
        of the month.

        ID (column 0): An integer that uniquely identifies the group which the instance belongs to.

    Returns
    -------
    gradients : numpy.ndarray
        The gradient of the loss with respect to the predictions y_pred. This array has the same shape
        as y_pred.

    hessians : numpy.ndarray
        The second derivative (hessian) of the loss with respect to the predictions y_pred. For MSE loss,
        the hessian is constant and thus this array is filled with ones, having the same shape as y_pred.
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


# Get true values (means) and predicted values (aggregates)

def get_ytrue_y_pred_agg(y_true, y_pred, X):
    # Extract the metadata
    metadata = X[:, -3:]  # Assuming last three columns are the metadata
    unique_ids = np.unique(metadata[:, 0])  # Assuming ID is the first column
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