from typing import Union, Dict, Tuple
from pathlib import Path
from contextlib import contextmanager
from collections.abc import Mapping

import pickle
import config
import torch

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from skorch import NeuralNetRegressor
from skorch.utils import to_tensor
from skorch.helper import SliceDataset


class CustomNeuralNetRegressor(NeuralNetRegressor):
    """
    A custom Neural Network regressor that extends the NeuralNetRegressor class.

    This class implements a custom objective function and scoring method
    that takes into account metadata for each stake measurement.
    As the dataset has a monthly resolution, multiple records belong to one time
    period and should therefore take be into account when evaluating the score/loss.
    """

    def __init__(self, *args, nbFeatures:int=None, metadataColumns=None, **kwargs):
        """
        Initialize the CustomNeuralNetRegressor.

        Args:
            *args: Arguments to be passed to the parent NeuralNetRegressor class.
            nbFeatures (int): The number of features of the non aggregated data.
            metadataColumns (list): The metadata columns of the dataset.
            **kwargs: Keyword arguments to be passed to the parent NeuralNetRegressor class.
        """
        super().__init__(*args, **kwargs)
        self.param_search = None
        self.metadataColumns = metadataColumns
        self.nbFeatures = nbFeatures
        self.modelDtype = list(self.module.parameters())[0].dtype if len(list(self.module.parameters()))>0 else None

    def gridsearch(
        self,
        parameters: Dict[str, Union[list, np.ndarray]],
        splits: Dict[str, Union[list, np.ndarray]],
        dataset: list[SliceDataset],
    ) -> None:
        """
        Perform a grid search for hyperparameter tuning.

        This method uses GridSearchCV to exhaustively search through a specified parameter grid.

        Args:
            parameters (dict): A dictionary of parameters to search over.
            splits (tuple[list[tuple[ndarray, ndarray]]]): A dictionary containing cross-validation split information.
            dataset (list of skorch.helper.SliceDataset): The datasets that provides both input features and targets for training.

        Sets:
            self.param_search (GridSearchCV): The fitted GridSearchCV object.
        """

        clf = GridSearchCV(
            estimator=self,
            param_grid=parameters,
            cv=splits,
            verbose=1,
            n_jobs=config.NUM_JOBS,
            scoring=None,
            refit=True,
            error_score="raise",
            return_train_score=True,
        )

        clf.fit(dataset[0], y=dataset[1])
        self.param_search = clf

    def randomsearch(
        self,
        parameters: Dict[str, Union[list, np.ndarray]],
        n_iter: int,
        splits: Dict[str, Union[list, np.ndarray]],
        dataset: list[SliceDataset],
    ) -> None:
        """
        Perform a randomized search for hyperparameter tuning.

        This method uses RandomizedSearchCV to search a subset of the specified parameter space.

        Args:
            parameters (dict): A dictionary of parameters and their distributions to sample from.
            n_iter (int): Number of parameter settings that are sampled.
            splits (tuple[list[tuple[ndarray, ndarray]]]): A dictionary containing cross-validation split information.
            dataset (list of skorch.helper.SliceDataset): The datasets that provides both input features and targets for training.

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
            scoring=None,
            refit=True,
            error_score="raise",
            return_train_score=True,
            random_state=config.SEED,
        )

        clf.fit(dataset[0], y=dataset[1])
        self.param_search = clf

    def _unpack(self, x):
        indNonNan = [~xi.isnan() for xi in x]
        v = [x[i][indNonNan[i]].reshape(-1, self.nbFeatures) for i in range(x.shape[0])]
        x = torch.concatenate(v, dim=0)
        return x, indNonNan

    def _pack(self, y, indNonNan):
        out = torch.empty((len(indNonNan), indNonNan[0].shape[0]//self.nbFeatures), dtype=y.dtype, device=y.device)
        out.fill_(torch.nan)
        cnt = 0
        for i in range(len(indNonNan)):
            incr = indNonNan[i][::self.nbFeatures].sum().item()
            out[i][indNonNan[i][::self.nbFeatures]] = y[cnt:cnt+incr,0]
            cnt += incr
        return out

    def infer(self, x, **fit_params):
        """Perform a single inference step on a batch of data.

        Parameters
        ----------
        x: input data
            A batch of the input data.

        **fit_params: dict
            Additional parameters passed to the ``forward`` method of
            the module and to the ``self.train_split`` call.
        """
        x = to_tensor(x, device=self.device)
        if len(x.shape)==1:
            x = x[None]
        x, indNonNan = self._unpack(x)
        if self.modelDtype is not None:
            x = x.type(self.modelDtype)
        if isinstance(x, Mapping):
            raise NotImplementedError("The case when x is a Mapping has not been implemented yet. If you need it, copy the implementation of the infer method in the NeuralNet class and add the _pack and _unpack methods.")
        res = self.module_(x, **fit_params)
        return self._pack(res, indNonNan)

    def get_loss(self, y_pred, y_true, X=None, training=False):
        loss = 0.
        for yi_pred, yi_true in zip(y_pred, y_true):
            loss += (yi_pred[~yi_pred.isnan()].sum() - yi_true[~yi_true.isnan()].mean())**2
        return loss/len(y_true)

    def score(self, X: SliceDataset, y: SliceDataset) -> float:
        """
        Compute the mean squared error of the model on the given test data and labels.

        Args:
            X (skorch.helper.SliceDataset): The dataset that contains input features.
            y (skorch.helper.SliceDataset): The dataset that contains targets.

        Returns:
            float: The negative mean squared error (for compatibility with sklearn's GridSearchCV).
        """

        # Make a prediction based on the features available in the dataset
        y_pred = self.predict(X)
        y_pred = torch.tensor(y_pred)

        y_true = []
        dataset = self.get_dataset(X, y)
        iterator = self.get_iterator(dataset, training=False)
        for batch in iterator:
            y_true.append(batch[1])
        y_true = torch.concatenate(y_true, dim=0)
        mse = self.get_loss(y_pred, y_true).item()

        if config.LOSS == 'MSE':
            return -mse  # Return negative because GridSearchCV maximizes score
        else:
            raise ValueError(f"Loss function {config.LOSS} not supported.")

    def predict(self, features: SliceDataset) -> np.ndarray:
        """
        Predict using the fitted model.

        Args:
            features (skorch.helper.SliceDataset): The dataset that contains input features.

        Returns:
            array-like: The predicted values (in monthly format).
        """
        # Check if the model is fitted
        check_is_fitted(self)

        return super().predict(features)

    def evalMetrics(self,
                    y_pred: np.array,
                    y_target: np.array) -> Tuple[float, float, float]:
        """
        Compute three evaluation metrics of the model on the given test data and labels.

        Args:
            y_target (array-like): The true target values.
            y_pred (array-like): The predicted values.

        Returns:
            Tuple[float, float, float]: The mean squared error, root mean squared error and mean absolute error.
        """

        # Get the aggregated predictions and the mean score based on the true labels, and predicted labels.
        y_pred_agg = self.aggrPred(y_pred)
        y_true_mean = self.meanTrue(y_target)

        mse = mean_squared_error(y_true_mean, y_pred_agg)
        rmse = root_mean_squared_error(y_true_mean, y_pred_agg)
        mae = mean_absolute_error(y_true_mean, y_pred_agg)

        # Pearson correlation
        pearson_corr = np.corrcoef(y_true_mean, y_pred_agg)[0, 1]

        return mse, rmse, mae, pearson_corr

    def aggrPred(self, y_pred):
        if isinstance(y_pred, torch.Tensor):
            y_pred_agg = [yi_pred[~yi_pred.isnan()].sum() for yi_pred in y_pred]
            return torch.tensor(y_pred_agg)
        elif isinstance(y_pred, np.ndarray):
            y_pred_agg = [yi_pred[~np.isnan(yi_pred)].sum() for yi_pred in y_pred]
            return np.array(y_pred_agg)
        else: raise TypeError

    def meanTrue(self, y_true):
        if isinstance(y_true, torch.Tensor):
            y_true_agg = [yi_true[~np.isnan(yi_true)].mean() for yi_true in y_true]
            return torch.tensor(y_true_agg)
        elif isinstance(y_true, np.ndarray):
            y_true_agg = [yi_true[~np.isnan(yi_true)].mean() for yi_true in y_true]
            return np.array(y_true_agg)
        else: raise TypeError

    def aggrPredict(self, features: SliceDataset) -> np.ndarray:
        """
        Makes predictions in aggregated format using the fitted model.
        Args:
            features (skorch.helper.SliceDataset): The dataset that contains input features.

        Returns:
            np.ndarray: The predicted values.
        """
        # Check if the model is fitted
        check_is_fitted(self)

        # Predictions in monthly format
        y_pred = super().predict(features)
        y_pred_agg = self.aggrPred(y_pred)

        return y_pred_agg

    def cumulative_pred(self, features: SliceDataset) -> np.ndarray:
        """Make cumulative monthly predictions for each stake measurement.

        Args:
            features (skorch.helper.SliceDataset): The dataset that contains input features.

        Returns:
            np.ndarray: The cumulative predicted values.
        """

        y_pred = super().predict(features)
        cum_pred = np.zeros_like(y_pred)
        cum_pred.fill(np.nan)
        for i in range(len(y_pred)):
            ind = ~np.isnan(y_pred[i])
            cum_pred[i][ind] = np.cumsum(y_pred[i][ind])
        return cum_pred

    def save_model(self, fname: str) -> None:
        """Save a grid search or randomized search CV instance to a file"""
        with self.model_file(fname, "wb") as f:
            pickle.dump(self.param_search, f)

    def load_model(self, fname: str) -> GridSearchCV | RandomizedSearchCV:
        """Load a grid search or randomized search CV instance from a file"""
        with self.model_file(fname, "rb") as f:
            self.param_search = pickle.load(f)

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

    @staticmethod
    def _create_features_metadata(
            X: pd.DataFrame,
            meta_data_columns: list) -> Tuple[np.array, np.ndarray]:
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

        # remove columns that are not used in metadata or features
        feature_columns = feature_columns.drop(
            config.NOT_METADATA_NOT_FEATURES)
        # Convert feature_columns to a list (if needed)
        feature_columns = list(feature_columns)

        # Extract metadata and features
        metadata = X[meta_data_columns].values
        features = X[feature_columns].values

        return features, metadata
