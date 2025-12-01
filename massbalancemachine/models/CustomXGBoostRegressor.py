"""
This code is taken, and refactored, and inspired from the work performed by: Kamilla Hauknes Sjursen

The CustomXGBoostRegressor class inherits from the XGBoost Regressor and is adapted to account for a monthly resolution.

Date Created: 09/08/2024
"""

from typing import Union, Dict, Tuple
from pathlib import Path
from contextlib import contextmanager

import dill
import config

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils.validation import check_is_fitted
import data_processing
import metrics

try:
    import cupy as cp
except ImportError:
    pass


class CustomXGBoostRegressor(XGBRegressor):
    """
    A custom XGBoost regressor that extends the XGBRegressor class.

    This class implements a custom objective function and scoring method
    that takes into account metadata for each stake measurement.
    As the dataset has a monthly resolution, multiple records belong to one time
    period and should therefore take be into account when evaluating the score/loss.
    """

    def __init__(self, cfg: config.Config, **kwargs):
        """
        Initialize the CustomXGBoostRegressor.

        Args:
            cfg (config.Config): Configuration instance.
            **kwargs: Keyword arguments to be passed to the parent XGBRegressor class.
        """
        super().__init__(**kwargs)
        self.cfg = cfg
        self.param_search = None

    def gridsearch(
        self,
        parameters: Dict[str, Union[list, np.ndarray]],
        splits: Dict[str, Union[list, np.ndarray]],
        features: pd.DataFrame,
        targets: np.ndarray,
    ) -> None:
        """
        Perform a grid search for hyperparameter tuning.

        This method uses GridSearchCV to exhaustively search through a specified parameter grid.

        Args:
            parameters (dict): A dictionary of parameters to search over.
            splits (tuple[list[tuple[ndarray, ndarray]]]): A dictionary containing cross-validation split information.
            features (pandas.DataFrame): The input features for training.
            targets (array-like): The target values for training.

        Sets:
            self.param_search (GridSearchCV): The fitted GridSearchCV object.
        """

        clf = GridSearchCV(
            estimator=self,
            param_grid=parameters,
            cv=splits,
            verbose=1,
            n_jobs=self.cfg.numJobs,
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

        Sets:
            self.param_search (RandomizedSearchCV): The fitted RandomizedSearchCV object.
        """
        clf = RandomizedSearchCV(
            estimator=self,
            param_distributions=parameters,
            n_iter=n_iter,
            cv=splits,
            verbose=1,
            n_jobs=self.cfg.numJobs,
            scoring=None,  # Uses default in CustomXGBRegressor()
            refit=True,
            error_score="raise",
            return_train_score=True,
            random_state=self.cfg.seed,
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
        features, metadata = data_processing.utils.create_features_metadata(self.cfg, X)

        # If running on GPU need to be converted to cupy
        if "cuda" in self.get_params()["device"]:
            features = cp.array(features)
            y = cp.array(y)

        # Define closure that captures metadata for use in custom objective
        def custom_objective(y_true, y_pred):
            return self._custom_mse_metadata(
                y_true, y_pred, metadata, self.cfg.metaData
            )

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
        features, metadata = data_processing.utils.create_features_metadata(self.cfg, X)

        # If running on GPU need to be converted to cupy
        if "cuda" in self.get_params()["device"]:
            features = cp.array(features)

        # Make a prediction based on the features available in the dataset
        y_pred = self.predict(features)

        # Get the aggregated predictions and the mean score based on the true labels, and predicted labels
        # based on the metadata.
        y_pred_agg, y_true_mean, _, _ = self._create_metadata_scores(
            metadata, y, y_pred, self.cfg.metaData
        )

        # Calculate score
        mse = ((y_pred_agg - y_true_mean) ** 2).mean()

        if self.cfg.loss == "MSE":
            return -mse  # Return negative because GridSearchCV maximizes score
        else:
            raise ValueError(f"Loss function {self.cfg.loss} not supported.")

    def score_geod(self, X: pd.DataFrame, y: np.array, periods: list) -> float:
        """
        Compute the mean squared error of the model on the given geodetic data and target MB.

        Args:
            X (pd.DataFrame): The input features of whole glacier grid including metadata columns.
            y (array-like): The true geodetic mass balance.

        Returns:
            float: The negative mean squared error (for compatibility with sklearn's GridSearchCV).
        """

        # Separate the features from the metadata provided in the dataset
        features_grid, metadata_grid = data_processing.utils.create_features_metadata(
            self.cfg, X
        )

        # If running on GPU need to be converted to cupy
        if "cuda" in self.get_params()["device"]:
            features_grid = cp.array(features_grid)

        # Make a prediction based on the features available in the dataset
        y_pred_grid = self.predict(features_grid)

        y_pred_agg, _ = self._create_metadata_scores_geod(
            metadata_grid, y_pred_grid, self.cfg.metaData, periods
        )

        # Compute Mean Squared Error, ignoring NaNs in case of missing data
        mse = np.nanmean((y_pred_agg - y) ** 2)

        # Return negative MSE for GridSearchCV compatibility
        if self.cfg.loss == "MSE":
            return -mse
        else:
            raise ValueError(f"Loss function {self.cfg.loss} not supported.")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict using the fitted model.

        Args:
            features (pd.DataFrame): The input features.

        Returns:
            array-like: The predicted values (in monthly format).
        """
        # # Check if the model is fitted
        # check_is_fitted(self)

        return super().predict(features)

    def evalMetrics(
        self, metadata: np.array, y_pred: np.array, y_target: np.array, period=None
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
            metadata,
            y_target,
            y_pred,
            meta_data_columns=self.cfg.metaData,
            period=period,
        )

        scores = metrics.scores(y_true_mean, y_pred_agg)
        mse = scores["mse"]
        rmse = scores["rmse"]
        mae = scores["mae"]
        pearson_corr = scores["pearson_corr"]  # Pearson correlation
        r2 = scores["r2"]  # R2 regression score
        bias = scores["bias"]  # Model bias

        return mse, rmse, mae, pearson_corr, r2, bias

    def aggrPredict(
        self,
        metadata: np.array,
        features: pd.DataFrame,
        meta_data_columns: list = None,
    ) -> np.ndarray:
        """
        Makes predictions in aggregated format using the fitted model.
        Aggregate to meas ID level (annual or seasonal, etc.)
        Args:
            features (pd.DataFrame): The input features.
            meta_data_columns (list[str]): The metadata columns. If not specified,
                metadata fields of the configuration instance will be used.
            metadata (np.array): The metadata values.

        Returns:
            array-like: The predicted values (in monthly format).
        """
        # # Check if the model is fitted
        # check_is_fitted(self)

        # Predictions in monthly format
        y_pred = super().predict(features)

        # Aggregate to meas ID level (annual or seasonal, etc.)
        meta_data_columns = meta_data_columns or self.cfg.metaData
        df_metadata = pd.DataFrame(metadata, columns=meta_data_columns)

        # Aggregate y_pred and y_true for each group
        grouped_ids = df_metadata.assign(y_pred=y_pred).groupby("ID")
        y_pred_agg = grouped_ids["y_pred"].sum().values

        return y_pred_agg

    def cumulative_pred(self, df, month_pos):
        """Make cumulative monthly predictions for each stake measurement.

        Args:
            df pd.DataFrame: monthly input dataframe
            month_pos: dict that provides the position of each month relative to each other

        Returns:
            df: the same dataframe as input filled with the cumulative prediction
        """
        features, metadata = data_processing.utils.create_features_metadata(
            self.cfg, df
        )

        # Predictions in monthly format
        y_pred = super().predict(features)

        df = df.assign(pred=y_pred)

        # Vectorized operation for month abbreviation
        df["MONTH_NB"] = df["MONTHS"].map(month_pos)

        # Cumulative monthly sums using groupby
        df.sort_values(by=["ID", "MONTH_NB"], inplace=True)
        df["cum_pred"] = df.groupby("ID")["pred"].cumsum()

        return df

    def glacier_wide_pred(
        self, df_grid, months_head_pad, months_tail_pad, type_pred="annual"
    ):
        """
        Generate predictions for an entire glacier grid
        and return them aggregated by measurement point ID.

        Args:
            df_grid (pd.DataFrame): The input features of whole glacier grid including metadata columns.
            type_pred (str): The type of seasonal prediction to perform.
            months_head_pad, months_tail_pad: Unused variables which are here only to have the same interface between XGBoost and the Neural Network
        Returns:
            pd.DataFrame: The aggregated predictions for each measurement point ID.
        """
        if type_pred == "winter":
            # winter months from October to April
            winter_months = ["sep", "oct", "nov", "dec", "jan", "feb", "mar", "apr"]
            df_grid = df_grid[df_grid.MONTHS.isin(winter_months)]

        # Make predictions on whole glacier grid
        features_grid, metadata_grid = data_processing.utils.create_features_metadata(
            self.cfg, df_grid
        )

        # Make predictions aggregated to measurement ID:
        y_pred_grid_agg = self.aggrPredict(metadata_grid, features_grid)

        grouped_ids = df_grid.groupby("ID")[
            ["YEAR", "POINT_LAT", "POINT_LON", "GLWD_ID"]
        ].first()

        grouped_ids["pred"] = y_pred_grid_agg
        grouped_ids.reset_index(inplace=True)
        grouped_ids.sort_values(by="YEAR", inplace=True)

        return grouped_ids, None

    def save_model(self, fname: str) -> None:
        """Save a grid search or randomized search CV instance to a file"""
        with self.model_file(fname, "wb") as f:
            dill.dump(self.param_search, f)

    def load_model(self, fname: str) -> GridSearchCV | RandomizedSearchCV:
        """Load a grid search or randomized search CV instance from a file"""
        with self.model_file(fname, "rb") as f:
            self.param_search = dill.load(f)

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

    def _custom_mse_metadata(
        self,
        y_true: np.array,
        y_pred: np.array,
        metadata: np.array,
        meta_data_columns: list,
        geod_periods=[],
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
            self._create_metadata_scores(
                metadata, y_true, y_pred, meta_data_columns, period=None
            )
        )

        if self.cfg.loss == "MSE":
            # Compute gradients and hessians
            # Source: https://xgboosting.com/xgboost-train-model-with-custom-objective-function/#:~:text=XGBoost%20allows%20users%20to%20define,MSE)%20for%20a%20regression%20task.
            gradients_agg = y_pred_agg - y_true_mean

        else:
            raise ValueError(f"Loss function {self.cfg.loss} not supported.")

        # Create a mapping from ID to gradient
        gradient_map = dict(zip(grouped_ids.groups.keys(), gradients_agg))

        # Assign gradients to corresponding indices
        df_metadata["gradient"] = df_metadata["ID"].map(gradient_map)
        gradients[df_metadata.index] = df_metadata["gradient"].values

        return gradients, hessians

    @staticmethod
    def _create_metadata_scores(
        metadata: np.array,
        y_true: np.array,
        y_pred: np.array,
        meta_data_columns: list,
        period: str = None,
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
        df_metadata = df_metadata.assign(y_true=y_true, y_pred=y_pred)

        # Filter to specific period if necessary
        if period is not None:
            df_metadata = df_metadata[df_metadata.PERIOD == period]

        grouped_ids = df_metadata.groupby("ID")
        y_pred_agg = grouped_ids["y_pred"].sum().values
        y_true_mean = grouped_ids["y_true"].mean().values
        return y_pred_agg, y_true_mean, grouped_ids, df_metadata

    def _create_metadata_scores_geod(
        self,
        metadata: np.array,
        y_pred: np.array,  # predicted geodetic mass balance
        meta_data_columns: list,
        geod_periods: list,
        period: str = None,
    ) -> Tuple[np.array, pd.DataFrame]:
        """
        Create aggregated geodetic scores based on metadata.

        """
        df_metadata = pd.DataFrame(metadata, columns=meta_data_columns)
        df_metadata = df_metadata.assign(y_pred=y_pred)

        # Filter to specific period if necessary
        if period is not None:
            df_metadata = df_metadata[df_metadata.PERIOD == period]

        grouped_ids = df_metadata.groupby("ID")
        y_pred_agg = grouped_ids["y_pred"].sum().values

        grouped_ids = df_metadata.groupby("ID").agg(
            {
                "YEAR": "first",
                "POINT_LAT": "first",
                "POINT_LON": "first",
                "GLWD_ID": "first",
            }
        )

        grouped_ids["pred"] = y_pred_agg
        grouped_ids.reset_index(inplace=True)
        grouped_ids.sort_values(by="YEAR", inplace=True)

        # Calculate mean SMB per year and geod period and store in a DataFrame
        grouped_ids = (
            grouped_ids.groupby("GLWD_ID")
            .agg(
                pred_mean=("pred", "mean"),
                YEAR=("YEAR", "first"),  # Assumes YEAR is unique per GEOD_ID
            )
            .set_index("YEAR")
        )

        # Compute geodetic mass balance predictions
        geodetic_MB_pred = []
        for start_year, end_year in geod_periods:
            geodetic_range = range(start_year, end_year + 1)

            # Ensure years exist in index before selection
            valid_years = [yr for yr in geodetic_range if yr in grouped_ids.index]
            if valid_years:
                geodetic_MB_pred.append(
                    grouped_ids.loc[valid_years, "pred_mean"].mean()
                )
            else:
                geodetic_MB_pred.append(np.nan)  # Handle missing years

        # Convert to NumPy for numerical operations
        y_pred_agg = np.array(geodetic_MB_pred)

        return y_pred_agg, grouped_ids
