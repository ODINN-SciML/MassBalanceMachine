"""
This code is taken, and refactored, and inspired from the work performed by: Kamilla Hauknes Sjursen

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 18/07/2024
"""

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.utils.validation import check_is_fitted

from massbalancemachine.dataloader import DataLoader
from massbalancemachine.models.Model import Model


class CustomXGBoostRegressor(XGBRegressor, Model):
    """
    A custom XGBoost regressor that extends the XGBRegressor class.

    This class implements a custom objective function and scoring method
    that takes into account metadata for each stake measurement.
    As the dataset has a monthly resolution, multiple records belong to one time
    period and should therefore take be into account when evaluating the score/loss.
    """

    def __init__(self, **kwargs):
        """
        Initialize the CustomXGBoostRegressor.

        Args:
            **kwargs: Keyword arguments to be passed to the parent XGBRegressor class.
        """
        super(XGBRegressor).__init__(**kwargs)
        super(Model).__init__(self)

    def fit(self, X, y, **fit_params):
        """
        Fit the model to the training data.

        This method overrides the parent class fit method to incorporate
        a custom objective function that uses metadata.

        Args:
            X (pd.DataFrame): The input features including metadata columns.
            y (array-like): The target values.
            **fit_params: Additional parameters to be passed to the parent fit method.

        Returns:
            self: The fitted estimator.
        """

        # Separate the features from the metadata provided in the dataset
        features, metadata = self._create_features_metadata(X)

        # Define closure that captures metadata for use in custom objective
        def custom_objective(y_true, y_pred):
            return self._custom_mse_metadata(y_true, y_pred, metadata)

        # Set custom objective
        self.set_params(objective=custom_objective)

        # Call fit method from parent class (XGBRegressor)
        super(XGBRegressor).fit(features, y, **fit_params)

        return self

    def score(self, X, y):
        """
        Compute the mean squared error of the model on the given test data and labels.

        Args:
            X (pd.DataFrame): The input features including metadata columns.
            y (array-like): The true labels.

        Returns:
            float: The negative mean squared error (for compatibility with sklearn's GridSearchCV).
        """

        # Separate the features from the metadata provided in the dataset
        features, metadata = self._create_features_metadata(X)

        # Make a prediction based on the features available in the dataset
        y_pred = self.predict(features)

        # Get the aggregated predictions and the mean score based on the true labels, and predicted labels
        # based on the metadata.
        y_pred_agg, y_true_mean, _, _ = self._create_metadata_scores(metadata, y, y_pred)

        # Calculate MSE
        mse = ((y_pred_agg - y_true_mean) ** 2).mean()

        return -mse  # Return negative because GridSearchCV maximizes score

    def predict(self, features):
        """
        Predict using the fitted model.

        Args:
            features (pd.DataFrame): The input features.

        Returns:
            array-like: The predicted values.
        """
        # Check if the model is fitted
        check_is_fitted(self)

        return super().predict(features)

    def perform_gridsearch(self, dataloader: DataLoader, random=True, loss='reg:squarederror', score='reg:squarederror',
                           **params):
        pass

    def monthly_loss(self, metric='MSE'):
        pass

    @staticmethod
    def _create_features_metadata(X):
        """
        Split the input DataFrame into features and metadata.

        Args:
            X (pd.DataFrame): The input DataFrame containing both features and metadata.

        Returns:
            tuple: A tuple containing:
                - features (array-like): The feature values.
                - metadata (array-like): The metadata values.
        """
        # Split features from metadata
        metadata_columns = ['POINT_ID', 'ID', 'N_MONTHS', 'MONTH']
        # Get feature columns by subtracting metadata columns from all columns
        feature_columns = X.columns.difference(metadata_columns)

        # Convert feature_columns to a list (if needed)
        feature_columns = list(feature_columns)

        # Extract metadata and features
        metadata = X[metadata_columns[1:]].values  # Exclude 'POINT_ID'
        features = X[feature_columns].values

        return features, metadata

    @staticmethod
    def _custom_mse_metadata(y_true, y_pred, metadata):
        """
        Compute custom gradients and hessians for the MSE loss, taking into account metadata.

        Args:
            y_true (array-like): The true target values.
            y_pred (array-like): The predicted values.
            metadata (array-like): The metadata for each data point.

        Returns:
            tuple: A tuple containing:
                - gradients (array-like): The computed gradients.
                - hessians (array-like): The computed hessians.
        """
        # Initialize gradients and hessians
        gradients = np.zeros_like(y_pred)
        hessians = np.ones_like(y_pred)

        # Get the aggregated predictions and the mean score based on the true labels, and predicted labels
        # based on the metadata.
        y_pred_agg, y_true_mean, grouped_ids, df_metadata = CustomXGBoostRegressor._create_metadata_scores(metadata, y_true, y_pred)

        # Compute gradients
        gradients_agg = y_pred_agg - y_true_mean

        # Create a mapping from ID to gradient
        gradient_map = dict(zip(grouped_ids.groups.keys(), gradients_agg))

        # Assign gradients to corresponding indices
        df_metadata['gradient'] = df_metadata['ID'].map(gradient_map)
        gradients[df_metadata.index] = df_metadata['gradient'].values

        return gradients, hessians

    @staticmethod
    def _create_metadata_scores(metadata, y1, y2):
        """
        Create aggregated scores based on metadata.

        Args:
            metadata (array-like): The metadata for each data point.
            y1 (array-like): The first set of values (typically true values).
            y2 (array-like): The second set of values (typically predicted values).

        Returns:
            tuple: A tuple containing:
                - y_pred_agg (array-like): Aggregated predictions.
                - y_true_mean (array-like): Mean of true values.
                - grouped_ids (pd.GroupBy): Grouped data by ID.
                - df_metadata (pd.DataFrame): DataFrame of metadata.
        """
        df_metadata = pd.DataFrame(metadata, columns=['ID', 'N_MONTHS', 'MONTH'])

        # Aggregate y_pred and y_true for each group
        grouped_ids = df_metadata.assign(y_true=y1, y_pred=y2).groupby('ID')
        y_pred_agg = grouped_ids['y_pred'].sum().values
        y_true_mean = grouped_ids['y_true'].mean().values

        return y_pred_agg, y_true_mean, grouped_ids, df_metadata
