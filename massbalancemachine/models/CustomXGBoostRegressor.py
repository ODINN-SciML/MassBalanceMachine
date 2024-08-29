"""
This code is taken, and refactored, and inspired from the work performed by: Kamilla Hauknes Sjursen

The CustomXGBoostRegressor class inherits from the XGBoost Regressor and is adapted to account for a monthly resolution.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 09/08/2024
"""

from typing import Union, Dict, Tuple
from pathlib import Path
from contextlib import contextmanager

import dill
import config

import numpy as np
import pandas as pd
import cupy as cp

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error


class CustomXGBoostRegressor(XGBRegressor):
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
            meta_data_columns (list): The metadata columns of the dataset.
            **kwargs: Keyword arguments to be passed to the parent XGBRegressor class.
        """
        super().__init__(**kwargs)
        self.param_search = None

    def gridsearch(
        self,
        parameters: Dict[str, Union[list, np.ndarray]],
        splits: Dict[str, Union[list, np.ndarray]],
        features: pd.DataFrame,
        targets: np.ndarray,
        num_jobs: int = None,
    ) -> None:
        """
        Perform a grid search for hyperparameter tuning.

        This method uses GridSearchCV to exhaustively search through a specified parameter grid.

        Args:
            parameters (dict): A dictionary of parameters to search over.
            splits (tuple[list[tuple[ndarray, ndarray]]]): A dictionary containing cross-validation split information.
            features (pandas.DataFrame): The input features for training.
            targets (array-like): The target values for training.
            num_jobs (int, optional): The number of jobs to run in parallel. -1 means using all processors. Defaults to -1.

        Sets:
            self.param_search (GridSearchCV): The fitted GridSearchCV object.
        """

        clf = GridSearchCV(
            estimator=self,
            param_grid=parameters,
            cv=splits,
            verbose=1,
            n_jobs=config.NUM_JOBS,
            scoring=None,  # Uses default in CustomXGBRegressor()
            refit=True,
            error_score="raise",
            return_train_score=True,
        )

        clf.fit(features, targets)
        self.param_search = clf

    def randomsearch(
        self,
        parameters: Dict[str, Union[list, np.ndarray]],
        n_iter: int,
        splits: Dict[str, Union[list, np.ndarray]],
        features: pd.DataFrame,
        targets: np.ndarray,
        num_jobs: int = None,
        random_seed: int = None,
    ) -> None:
        """
        Perform a randomized search for hyperparameter tuning.

        This method uses RandomizedSearchCV to search a subset of the specified parameter space.

        Args:
            parameters (dict): A dictionary of parameters and their distributions to sample from.
            n_iter (int): Number of parameter settings that are sampled.
            splits (tuple[list[tuple[ndarray, ndarray]]]): A dictionary containing cross-validation split information.
            features (pandas.DataFrame): The input features for training.
            targets (array-like): The target values for training.
            num_jobs (int, optional): The number of jobs to run in parallel. -1 means using all processors. Defaults to -1.
            random_seed (int, optional): Random seed for reproducibility. Defaults to config.SEED.

        Sets:
            self.param_search (RandomizedSearchCV): The fitted RandomizedSearchCV object.
        """

        clf = RandomizedSearchCV(
            estimator=self,
            param_distributions=parameters,
            n_iter=n_iter,
            cv=splits,
            verbose=1,
            n_jobs=config.NUM_JOBS,
            scoring=None,  # Uses default in CustomXGBRegressor()
            refit=True,
            error_score="raise",
            return_train_score=True,
            random_state=config.SEED,
        )

        clf.fit(features, targets)

        self.param_search = clf

    def fit(
        self, X: pd.DataFrame, y: np.array, **fit_params
    ) -> "CustomXGBoostRegressor":
        """
        Fit the model to the training data.

        This method overrides the parent class fit method to incorporate
        a custom objective function that uses metadata.

        Args:
            **kwargs:
            X (pd.DataFrame): The input features including metadata columns.
            y (array-like): The target values.
            **fit_params: Additional parameters to be passed to the parent fit method.

        Returns:
            self: The fitted estimator.
        """

        # Separate the features from the metadata provided in the dataset
        features, metadata = self._create_features_metadata(X, config.META_DATA)

        # If running on GPU need to be converted to cupy
        if "cuda" in self.get_params()["device"]:
            features = cp.array(features)
            y = cp.array(y)

        # Define closure that captures metadata for use in custom objective
        def custom_objective(y_true, y_pred):
            return self._custom_mse_metadata(y_true, y_pred, metadata, config.META_DATA)

        # Set custom objective
        self.set_params(objective=custom_objective)

        # Call the fit function from the XGBoost library with the custom
        # objective function
        super().fit(features, y, **fit_params)

        return self

    def score(self, X: pd.DataFrame, y: np.array) -> float:
        """
        Compute the mean squared error of the model on the given test data and labels.

        Args:
            X (pd.DataFrame): The input features including metadata columns.
            y (array-like): The true labels.

        Returns:
            float: The negative mean squared error (for compatibility with sklearn's GridSearchCV).
        """

        # Separate the features from the metadata provided in the dataset
        features, metadata = self._create_features_metadata(X, config.META_DATA)

        # If running on GPU need to be converted to cupy
        if "cuda" in self.get_params()["device"]:
            features = cp.array(features)

        # Make a prediction based on the features available in the dataset
        y_pred = self.predict(features)

        # Get the aggregated predictions and the mean score based on the true labels, and predicted labels
        # based on the metadata.
        y_pred_agg, y_true_mean, _, _ = self._create_metadata_scores(
            metadata, y, y_pred, config.META_DATA
        )

        # Calculate MSE
        mse = ((y_pred_agg - y_true_mean) ** 2).mean()

        return -mse  # Return negative because GridSearchCV maximizes score

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict using the fitted model.

        Args:
            features (pd.DataFrame): The input features.

        Returns:
            array-like: The predicted values (in monthly format).
        """
        # Check if the model is fitted
        check_is_fitted(self)

        return super().predict(features)

    def evalMetrics(
        self, metadata: np.array, y_pred: np.array, y_target: np.array
    ) -> Tuple[float, float, float]:
        """
        Compute three evaluation metrics of the model on the given test data and labels.

        Args:
            y_target (array-like): The true target values.
            y_pred (array-like): The predicted values.
            metadata (np.array): The metadata values.

        Returns:
            Tuple[float, float, float]: The mean squared error, root mean squared error and mean absolute error.
        """

        # Get the aggregated predictions and the mean score based on the true labels, and predicted labels
        # based on the metadata.
        y_pred_agg, y_true_mean, _, _ = self._create_metadata_scores(
            metadata, y_target, y_pred, meta_data_columns=config.META_DATA
        )

        mse = mean_squared_error(y_true_mean, y_pred_agg)
        rmse = root_mean_squared_error(y_true_mean, y_pred_agg)
        mae = mean_absolute_error(y_true_mean, y_pred_agg)

        return mse, rmse, mae

    def aggrPredict(
        self, metadata: np.array, meta_data_columns: list, features: pd.DataFrame
    ) -> np.ndarray:
        """
        Makes predictions in aggregated format using the fitted model.
        Args:
            features (pd.DataFrame): The input features.
            meta_data_columns (list[str]): The metadata columns.
            metadata (np.array): The metadata values.

        Returns:
            array-like: The predicted values (in monthly format).
        """
        # Check if the model is fitted
        check_is_fitted(self)

        # Predictions in monthly format
        y_pred = super().predict(features)

        # Aggregate to meas ID level (annual or seasonal, etc.)
        df_metadata = pd.DataFrame(metadata, columns=meta_data_columns)

        # Aggregate y_pred and y_true for each group
        grouped_ids = df_metadata.assign(y_pred=y_pred).groupby("ID")
        y_pred_agg = grouped_ids["y_pred"].sum().values

        return y_pred_agg

    @classmethod
    @contextmanager
    def model_file(cls, fname: str, mode: str):
        """Context manager to handle model file and directory operations"""
        models_dir = Path("./models")
        # Check if the directory already exists
        models_dir.mkdir(exist_ok=True)
        file_path = models_dir / fname
        try:
            with open(file_path, mode) as f:
                yield f
        except IOError:
            print(f"Error accessing file: {file_path}")
            raise

    def save_model(self, fname: str) -> None:
        """Save a grid search or randomized search CV instance to a file"""
        with self.model_file(fname, "wb") as f:
            dill.dump(self.param_search, f)

    @classmethod
    def load_model(cls, fname: str) -> GridSearchCV | RandomizedSearchCV:
        """Load a grid search or randomized search CV instance from a file"""
        with cls.model_file(fname, "rb") as f:
            return dill.load(f)  # returns the grid search instance (self.param_search)

    @staticmethod
    def _create_features_metadata(
        X: pd.DataFrame, meta_data_columns: list
    ) -> Tuple[np.array, np.ndarray]:
        """
        Split the input DataFrame into features and metadata.

        Args:
            X (pd.DataFrame): The input DataFrame containing both features and metadata.
            meta_data_columns (list): The metadata columns to be extracted.

        Returns:
            tuple: A tuple containing:
                - features (array-like): The feature values.
                - metadata (array-like): The metadata values.
        """
        # Split features from metadata
        # Get feature columns by subtracting metadata columns from all columns
        feature_columns = X.columns.difference(meta_data_columns)

        # remove POINT_BALANCE and YEAR from feature columns
        feature_columns = feature_columns.drop(
            ["POINT_BALANCE", "YEAR", "POINT_LAT", "POINT_LON"]
        )

        # Convert feature_columns to a list (if needed)
        feature_columns = list(feature_columns)

        # Extract metadata and features
        metadata = X[meta_data_columns].values
        features = X[feature_columns].values

        return features, metadata

    @staticmethod
    def _custom_mse_metadata(
        y_true: np.array, y_pred: np.array, metadata: np.array, meta_data_columns: list
    ) -> Tuple[np.array, np.array]:
        """
        Compute custom gradients and hessians for the MSE loss, taking into account metadata.

        Args:
            y_true (array-like): The true target values.
            y_pred (array-like): The predicted values.
            metadata (array-like): The metadata for each data point.

        Returns:
            tuple: A tuple containing:
                - gradients (array-like): The computed gradients.
                - hessians (array-like): The second derivative (hessian) of the loss with respect to
                the predictions y_pred. For MSE loss,
                the hessian is constant and thus this array is filled with ones, having the same shape as y_pred.
        """
        # Initialize gradients and hessians
        gradients = np.zeros_like(y_pred)
        hessians = np.ones_like(y_pred)

        # Get the aggregated predictions and the mean score based on the true labels, and predicted labels
        # based on the metadata.
        y_pred_agg, y_true_mean, grouped_ids, df_metadata = (
            CustomXGBoostRegressor._create_metadata_scores(
                metadata, y_true, y_pred, meta_data_columns
            )
        )

        # Compute gradients
        gradients_agg = y_pred_agg - y_true_mean

        # Create a mapping from ID to gradient
        gradient_map = dict(zip(grouped_ids.groups.keys(), gradients_agg))

        # Assign gradients to corresponding indices
        df_metadata["gradient"] = df_metadata["ID"].map(gradient_map)
        gradients[df_metadata.index] = df_metadata["gradient"].values

        return gradients, hessians

    @staticmethod
    def _create_metadata_scores(
        metadata: np.array, y1: np.array, y2: np.array, meta_data_columns: list
    ) -> Tuple[
        np.array, np.array, pd.core.groupby.generic.DataFrameGroupBy, pd.DataFrame
    ]:
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
        df_metadata = pd.DataFrame(metadata, columns=meta_data_columns)

        # Aggregate y_pred and y_true for each group
        grouped_ids = df_metadata.assign(y_true=y1, y_pred=y2).groupby("ID")
        y_pred_agg = grouped_ids["y_pred"].sum().values
        y_true_mean = grouped_ids["y_true"].mean().values

        return y_pred_agg, y_true_mean, grouped_ids, df_metadata
