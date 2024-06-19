"""
CustomXGBRegressor extends the XGBoost regressor to incorporate additional metadata in the learning process. The last three columns of the training dataset are treated as metadata rather than features.

The custom mean squared error function (custom_mse_metadata) calculates gradients and hessians using both feature data and metadata, enhancing the model's learning capabilities. This custom objective captures metadata along with target and predicted values for gradient and hessian computation during training.

The class includes a custom score function for evaluating validation performance in GridSearchCV. Parameters from XGBRegressor are customizable, and additional parameters can be passed via kwargs.
Examples

# >>> model = CustomXGBRegressor(n_estimators=500, learning_rate=0.05)
# >>> model.fit(X_train, y_train)  # X_train includes metadata as the last 3 columns
# >>> predictions = model.predict(X_test)  # X_test includes metadata as the last 3 columns

Note: CustomXGBRegressor requires a custom MSE function, custom_mse_metadata, for gradient and hessian computation using metadata.

@Author: Kamilla Haukens Sjursen, and adapted by Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 19/06/2024
"""
import numpy as np
from xgboost import XGBRegressor
from sklearn.utils.validation import check_is_fitted
from model_methods import *

class CustomXGBoostRegressor(XGBRegressor):
    """
    A custom XGBoost Regressor that incorporates additional metadata during training and scoring.
    
    Attributes:
        metadata_shape (int): The number of metadata columns in the input data.
    """

    def __init__(self, metadata_shape, **kwargs):
        """
        Initializes the CustomXGBoostRegressor with metadata shape and any additional keyword arguments.

        Args:
            metadata_shape (int): The number of metadata columns in the input data.
            **kwargs: Additional keyword arguments to pass to the XGBRegressor constructor.
        """
        super(CustomXGBoostRegressor, self).__init__(**kwargs)
        self.metadata_shape = metadata_shape

    def fit(self, X, y, **fit_params):
        """
        Fits the model using the provided features and target values.
        
        Args:
            X (np.ndarray): The input data, including both features and metadata.
            y (np.ndarray): The target values.
            **fit_params: Additional fitting parameters to pass to the fit method of XGBRegressor.
        
        Returns:
            self: The fitted estimator.
        """
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
        """
        Predicts target values for the given input data.
        
        Args:
            X (np.ndarray): The input data, including both features and metadata.
        
        Returns:
            np.ndarray: The predicted target values.
        """
        # Check if the model is fitted
        check_is_fitted(self)

        # Extract features from the input data
        features = X[:, :-self.metadata_shape]

        return super().predict(features)

    def score(self, X, y):
        """
        Computes the negative mean squared error of the predictions.
        
        Args:
            X (np.ndarray): The input data, including both features and metadata.
            y (np.ndarray): The true target values.
        
        Returns:
            float: The negative mean squared error of the aggregated predictions.
        """
        # Get predictions
        y_pred = self.predict(X)

        # Split metadata and features
        metadata, features = X[:, -self.metadata_shape:], X[:, :-self.metadata_shape]

        all_pred_agg = []
        all_true_mean = []

        unique_ids = np.unique(metadata[:, 0])  # ID is the first column of metadata

        # Loop over each unique ID to aggregate predictions and true values
        for uid in unique_ids:
            indexes = metadata[:, 0] == uid

            # Aggregate predictions for the current ID
            y_pred_agg = np.sum(y_pred[indexes])

            # Get mean of true values for the current ID
            y_true_mean = np.mean(y[indexes])

            all_pred_agg.append(y_pred_agg)
            all_true_mean.append(y_true_mean)

        all_pred_agg = np.array(all_pred_agg)
        all_true_mean = np.array(all_true_mean)

        # Compute mean squared error
        mse = ((all_pred_agg - all_true_mean) ** 2).mean()

        return -mse  # Return negative MSE because GridSearchCV maximizes score