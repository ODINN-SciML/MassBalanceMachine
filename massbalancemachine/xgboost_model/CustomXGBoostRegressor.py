import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.utils.validation import check_is_fitted


class CustomXGBoostRegressor(XGBRegressor):
    def __init__(self, **kwargs):
        super(CustomXGBoostRegressor, self).__init__(**kwargs)

    def fit(self, X, y, **fit_params):
        # Split features from metadata
        metadata_columns = ['POINT_ID', 'ID', 'N_MONTHS', 'MONTH']
        # Get feature columns by subtracting metadata columns from all columns
        feature_columns = X.columns.difference(metadata_columns)

        # Convert feature_columns to a list (if needed)
        feature_columns = list(feature_columns)

        # Extract metadata and features
        metadata = X[metadata_columns[1:]].values  # Exclude 'POINT_ID'
        features = X[feature_columns].values

        # Define closure that captures metadata for use in custom objective
        def custom_objective(y_true, y_pred):
            return self._custom_mse_metadata(y_true, y_pred, metadata)

        # Set custom objective
        self.set_params(objective=custom_objective)

        # Call fit method from parent class (XGBRegressor)
        super().fit(features, y, **fit_params)

        return self

    def score(self, X, y):
        metadata_columns = ['POINT_ID', 'ID', 'N_MONTHS', 'MONTH']
        # Get feature columns by subtracting metadata columns from all columns
        feature_columns = X.columns.difference(metadata_columns)

        # Convert feature_columns to a list (if needed)
        feature_columns = list(feature_columns)

        # Extract metadata and features
        metadata = X[metadata_columns[1:]]  # Exclude 'POINT_ID'
        features = X[feature_columns].values

        y_pred = self.predict(features)

        y_pred_agg, y_true_mean, _, _ = self._create_metadata_scores(metadata, y, y_pred)

        # Calculate MSE
        mse = ((y_pred_agg - y_true_mean) ** 2).mean()

        return -mse  # Return negative because GridSearchCV maximizes score

    def predict(self, features):
        # Check if the model is fitted
        check_is_fitted(self)

        return super().predict(features)

    @staticmethod
    def _custom_mse_metadata(y_true, y_pred, metadata):
        # Initialize gradients and hessians
        gradients = np.zeros_like(y_pred)
        hessians = np.ones_like(y_pred)

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
        df_metadata = pd.DataFrame(metadata, columns=['ID', 'N_MONTHS', 'MONTH'])

        # Aggregate y_pred and y_true for each group
        grouped_ids = df_metadata.assign(y_true=y1, y_pred=y2).groupby('ID')
        y_pred_agg = grouped_ids['y_pred'].sum().values
        y_true_mean = grouped_ids['y_true'].mean().values

        return y_pred_agg, y_true_mean, grouped_ids, df_metadata
