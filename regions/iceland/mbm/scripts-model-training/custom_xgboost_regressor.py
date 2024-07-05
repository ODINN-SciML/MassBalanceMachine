import numpy as np

from xgboost import XGBRegressor
from sklearn.utils.validation import check_is_fitted

from model_methods import *


class CustomXGBoostRegressor(XGBRegressor):
    """
    CustomXGBRegressor is an extension of the XGBoost regressor that incorporates additional metadata into the learning process. The estimator
    is tailored to handle training datasets where the last three columns are metadata rather than features.

    The metadata is utilized in a custom mean squared error function. This function calculates gradients and hessians incorporating metadata,
    allowing the model to learn from both standard feature data and additional information provided as metadata.

    The custom objective closure captures metadata along with the target values and predicted values to compute the gradients and hessians needed
    for the XGBoost training process.

    The class contains a custom score function (custom mse) that is used in GridSearchCV to evaluate validation performance for each fold.
    This is the default scorer for the class.

    Parameters inherited from XGBRegressor are customizable and additional parameters can be passed via kwargs, which will be handled by the
    XGBRegressor's __init__ method.

    Examples
    --------
    # >>> model = CustomXGBRegressor(n_estimators=500, learning_rate=0.05)
    # >>> model.fit(X_train, y_train)  # X_train includes metadata as the last 3 columns
    # >>> predictions = model.predict(X_test)  # X_test includes metadata as the last 3 columns

    Note: CustomXGBRegressor requires a custom MSE function, `custom_mse_metadata`, which computes the gradient and hessian using additional metadata.
    """

    def __init__(self, metadata_shape, **kwargs):
        super(CustomXGBoostRegressor, self).__init__(**kwargs)
        self.metadata_shape = metadata_shape

    def fit(self, X, y, **fit_params):
        # Split features from metadata
        metadata, features = X[:, -self.metadata_shape:], X[:, :-self.metadata_shape]

        # Define closure that captures metadata for use in custom objective
        def custom_objective(y_true, y_pred):
            return custom_mse_metadata(y_true, y_pred, metadata)

        # Set custom objective
        self.set_params(objective=custom_objective)

        # Call fit method from parent class (XGBRegressor)
        super().fit(features, y, **fit_params)

        return self

    def predict(self, X):
        # Check if the model is fitted
        check_is_fitted(self)

        features = X[:, :-self.metadata_shape]

        return super().predict(features)

    def score(self, X, y):
        y_pred = self.predict(X)

        metadata, features = X[:, -self.metadata_shape:], X[:, :-self.metadata_shape]

        all_pred_agg = []
        all_true_mean = []

        unique_ids = np.unique(metadata[:, 0])  # ID is first column of metadata

        # Loop over each unique ID to aggregate/get mean
        for uid in unique_ids:
            indexes = metadata[:, 0] == uid

            # Aggregate predictions for the current ID
            y_pred_agg = np.sum(y_pred[indexes])

            # Get mean of true values for the current ID
            y_true_mean = np.mean(y[indexes])

            all_pred_agg.append(y_pred_agg)
            all_true_mean.append(y_true_mean)

            # mse += (y_pred_agg - y_true_mean) ** 2

        all_pred_agg = np.array(all_pred_agg)
        all_true_mean = np.array(all_true_mean)

        # Compute mse
        mse = ((all_pred_agg - all_true_mean) ** 2).mean()

        return -mse  # Return negative because GridSearchCV maximizes score
