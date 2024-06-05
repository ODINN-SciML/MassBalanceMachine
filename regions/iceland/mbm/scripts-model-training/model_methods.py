"""
This script contains functions training the XGBoost model.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 04/06/2024
"""

import xgboost as xgb
import numpy as np
import sklearn


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


def get_prediction_per_season(X_train_all, y_train_all, splits_all, best_model, months=12):
    y_pred_list = []
    y_test_list = []

    for train_index, test_index in splits_all:
        # Loops over n_splits iterations and gets train and test splits in each fold
        X_train, X_test = X_train_all[train_index], X_train_all[test_index]
        y_train, y_test = y_train_all[train_index], y_train_all[test_index]

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        indices = np.argwhere((X_test == months))[:, 0]
        y_test_crop = y_test[indices]
        y_pred_crop = y_pred[indices]

        y_test_list.extend(y_test_crop)
        y_pred_list.extend(y_pred_crop)

    # Arrays of predictions and observations for each fold
    y_test_season = np.hstack([*y_test_list])
    y_pred_season = np.hstack([*y_pred_list])

    return y_test_season, y_pred_season