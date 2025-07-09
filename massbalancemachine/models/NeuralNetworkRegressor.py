from typing import Union, Dict, Tuple
from pathlib import Path
from contextlib import contextmanager
from collections.abc import Mapping
from datetime import datetime
import traceback

import os
import config
import torch

import numpy as np
import pandas as pd
import random as rd

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from skorch import NeuralNetRegressor
from skorch.utils import to_tensor
from skorch.helper import SliceDataset
import data_processing

_models_dir = Path("./models")


class CustomNeuralNetRegressor(NeuralNetRegressor):
    """
    A custom Neural Network regressor that extends the NeuralNetRegressor class.

    This class implements a custom objective function and scoring method
    that takes into account metadata for each stake measurement.
    As the dataset has a monthly resolution, multiple records belong to one time
    period and should therefore take be into account when evaluating the score/loss.
    """

    def __init__(self,
                 cfg: config.Config,
                 *args,
                 nbFeatures: int = None,
                 metadataColumns=None,
                 **kwargs):
        """
        Initialize the CustomNeuralNetRegressor.

        Args:
            cfg (config.Config): Configuration instance.
            *args: Arguments to be passed to the parent NeuralNetRegressor class.
            nbFeatures (int): The number of features of the non aggregated data.
            metadataColumns (list): The metadata columns of the dataset. If not
                specified, metadata fields of the configuration instance are used.
            **kwargs: Keyword arguments to be passed to the parent NeuralNetRegressor
                class.
        """
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.param_search = None
        self.metadataColumns = metadataColumns or self.cfg.metaData
        self.nbFeatures = nbFeatures

        # seed all
        self.seed_all()

    def initialize_module(self):
        super().initialize_module()
        # Now the module instance is available as self.module_
        if hasattr(self.module_, "parameters"):
            self.modelDtype = list(self.module_.parameters())[0].dtype
        else:
            self.modelDtype = None
        return self

    def _unpack_inp(self, x):
        indNonNan = [~xi.isnan() for xi in x]
        v = [
            x[i][indNonNan[i]].reshape(-1, self.nbFeatures)
            for i in range(x.shape[0])
        ]
        x = torch.concatenate(v, dim=0)
        return x, indNonNan

    def _pack_out(self, y, indNonNan):
        out = torch.empty(
            (len(indNonNan), indNonNan[0].shape[0] // self.nbFeatures),
            dtype=y.dtype,
            device=y.device)
        out.fill_(torch.nan)
        cnt = 0
        for i in range(len(indNonNan)):
            incr = indNonNan[i][::self.nbFeatures].sum().item()
            out[i][indNonNan[i][::self.nbFeatures]] = y[cnt:cnt + incr, 0]
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
        if len(x.shape) == 1:
            x = x[None]
        x, indNonNan = self._unpack_inp(x)
        if self.modelDtype is not None:
            x = x.type(self.modelDtype)
        if isinstance(x, Mapping):
            raise NotImplementedError(
                "The case when x is a Mapping has not been implemented yet. If you need it, copy the implementation of the infer method in the NeuralNet class and add the _pack_out and _unpack_inp methods."
            )
        res = self.module_(x, **fit_params)
        return self._pack_out(res, indNonNan)

    def get_loss(self, y_pred, y_true, X=None, training=False):
        loss = 0.
        for yi_pred, yi_true in zip(y_pred, y_true):
            loss += (yi_pred[~yi_pred.isnan()].sum() -
                     yi_true[~yi_true.isnan()].mean())**2
        return loss / len(y_true)

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

        if self.cfg.loss == 'MSE':
            return -mse  # Return negative because GridSearchCV maximizes score
        else:
            raise ValueError(f"Loss function {self.cfg.loss} not supported.")

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

    def evalMetrics(self, y_pred: np.array,
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
            y_pred_agg = [
                yi_pred[~yi_pred.isnan()].sum() for yi_pred in y_pred
            ]
            return torch.tensor(y_pred_agg)
        elif isinstance(y_pred, np.ndarray):
            y_pred_agg = [
                yi_pred[~np.isnan(yi_pred)].sum() for yi_pred in y_pred
            ]
            return np.array(y_pred_agg)
        else:
            raise TypeError

    def meanTrue(self, y_true):
        if isinstance(y_true, torch.Tensor):
            y_true_agg = [
                yi_true[~np.isnan(yi_true)].mean() for yi_true in y_true
            ]
            return torch.tensor(y_true_agg)
        elif isinstance(y_true, np.ndarray):
            y_true_agg = [
                yi_true[~np.isnan(yi_true)].mean() for yi_true in y_true
            ]
            return np.array(y_true_agg)
        else:
            raise TypeError

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
        y_pred = self.predict(features)
        y_pred_agg = self.aggrPred(y_pred)

        return y_pred_agg

    def cumulative_pred(self, features: SliceDataset) -> np.ndarray:
        """Make cumulative monthly predictions for each stake measurement.

        Args:
            features (skorch.helper.SliceDataset): The dataset that contains input features.

        Returns:
            np.ndarray: The cumulative predicted values.
        """

        y_pred = self.predict(features)
        cum_pred = np.zeros_like(y_pred)
        cum_pred.fill(np.nan)
        for i in range(len(y_pred)):
            ind = ~np.isnan(y_pred[i])
            cum_pred[i][ind] = np.cumsum(y_pred[i][ind])
        return cum_pred

    def glacier_wide_pred(self, df_grid_monthly, type_pred='annual'):
        """    
        Generate predictions for an entire glacier grid 
        and return them aggregated by measurement point ID.
        
        Args:
            df_grid_monthly (pd.DataFrame): The input features of whole glacier grid including metadata columns.
            type_pred (str): The type of seasonal prediction to perform.
        Returns:
            pd.DataFrame: The aggregated predictions for each measurement point ID.
        """

        if type_pred == 'winter':
            # winter months from October to April
            winter_months = [
                'sep', 'oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr'
            ]
            df_grid_monthly = df_grid_monthly[df_grid_monthly.MONTHS.isin(
                winter_months)]

        # Create features and metadata
        features_grid, metadata_grid = self._create_features_metadata(
            df_grid_monthly)

        # Ensure all tensors are on CPU if they are torch tensors
        if hasattr(features_grid, 'cpu'):
            features_grid = features_grid.cpu()

        # Ensure targets are also on CPU
        targets_grid = np.empty(len(features_grid))  # No targets in grid data
        if hasattr(targets_grid, 'cpu'):
            targets_grid = targets_grid.cpu()

        # Create the dataset
        dataset_grid = data_processing.AggregatedDataset(
            self.cfg,
            features=features_grid,
            metadata=metadata_grid,
            targets=targets_grid)

        dataset_grid = [
            SliceDataset(dataset_grid, idx=0),
            SliceDataset(dataset_grid, idx=1)
        ]

        # Make predictions aggr to meas ID
        y_pred = self.predict(dataset_grid[0])
        y_pred_agg = self.aggrPredict(dataset_grid[0])

        batchIndex = np.arange(len(y_pred_agg))
        y_true = np.array([e for e in dataset_grid[1][batchIndex]])

        # Aggregate predictions
        id = dataset_grid[0].dataset.indexToId(batchIndex)
        data = {'ID': id, 'pred': y_pred_agg}
        data = pd.DataFrame(data)
        data.set_index('ID', inplace=True)

        # Aggregated over seasonal:
        grouped_ids = df_grid_monthly.groupby('ID')[[
            'YEAR', 'POINT_LAT', 'POINT_LON', 'GLWD_ID'
        ]].first()

        grouped_ids = grouped_ids.merge(data, on='ID', how='left')

        months_per_id = df_grid_monthly.groupby('ID')['MONTHS'].unique()
        grouped_ids = grouped_ids.merge(months_per_id, on='ID')

        grouped_ids.reset_index(inplace=True)
        grouped_ids.sort_values(by='ID', inplace=True)
        grouped_ids['PERIOD'] = type_pred

        # Monthly preds:
        df_pred_months = pd.DataFrame(y_pred)
        df_pred_months['ID'] = id
        df_pred_months['MONTHS'] = grouped_ids['MONTHS']
        df_pred_months['PERIOD'] = grouped_ids['PERIOD']

        months_extended = [
            'sep',
            'oct',
            'nov',
            'dec',
            'jan',
            'feb',
            'mar',
            'apr',
            'may',
            'jun',
            'jul',
            'aug',
        ]

        df_months_nn = pd.DataFrame(columns=months_extended)

        for i, row in df_pred_months.iterrows():
            dic = {}
            for j, month in enumerate(row.MONTHS):
                if month in dic.keys():
                    month = month + '_'
                dic[month] = row[j]

            # add missing months from months extended
            for month in months_extended:
                if month not in dic.keys():
                    dic[month] = np.nan
            df_months_nn = pd.concat(
                [df_months_nn, pd.DataFrame([dic])], ignore_index=True)

        df_months_nn = df_months_nn.dropna(axis=1, how='all')
        df_months_nn['ID'] = df_pred_months['ID']
        df_months_nn['PERIOD'] = type_pred
        # df_months_nn['y_agg'] = y_pred_agg
        # if type_pred == 'winter':
        #     months = winter_months
        # else:
        #     months = months_extended
        # df_months_nn['sum'] = df_months_nn[months].sum(axis=1)

        return grouped_ids, df_months_nn

    def save_model(self, fname: str) -> None:
        """save the model parameters to a file.

        Args:
            fname (str): filename to save the model parameters to (without .pt extension).
        """
        file_path = _models_dir / fname
        _models_dir.mkdir(exist_ok=True)
        self.save_params(f_params=file_path.with_suffix(".pt"))

    def to(self, device):
        """Move model and necessary attributes to the specified device."""
        self.device = device

        # Only move if model is already initialized
        if hasattr(self, 'module_') and self.module_ is not None:
            self.module_.to(device)

        # Optional: move other tensor attributes
        if hasattr(self, 'some_tensor_attribute'):
            self.some_tensor_attribute = self.some_tensor_attribute.to(device)

        return self

    def _create_features_metadata(
            self,
            X: pd.DataFrame,
            meta_data_columns: list = None) -> Tuple[np.array, np.ndarray]:
        """
        Split the input DataFrame into features and metadata.

        Args:
            X (pd.DataFrame): The input DataFrame containing both features and metadata.
            meta_data_columns (list): The metadata columns to be extracted. If not
                specified, metadata fields of the configuration instance will be used.

        Returns:
            tuple: A tuple containing:
                - features (array-like): The feature values.
                - metadata (array-like): The metadata values.
        """
        meta_data_columns = meta_data_columns or self.cfg.metaData

        # # Split features from metadata
        # # Get feature columns by subtracting metadata columns from all columns
        # feature_columns = X.columns.difference(meta_data_columns)

        # # remove columns that are not used in metadata or features
        # feature_columns = feature_columns.drop(self.cfg.notMetaDataNotFeatures)
        # # Convert feature_columns to a list (if needed)
        # feature_columns = list(feature_columns)
        # print(feature_columns, len(feature_columns))

        feature_columns = self.cfg.featureColumns

        # Extract metadata and features
        metadata = X[meta_data_columns].values
        features = X[feature_columns].values

        return features, metadata

    def seed_all(self):
        """Sets the random seed everywhere for reproducibility.
        """
        # Python built-in random
        rd.seed(self.cfg.seed)

        # NumPy random
        np.random.seed(self.cfg.seed)

        # PyTorch seed
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)  # If using multiple GPUs

        # Ensuring deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Setting CUBLAS environment variable (helps in newer versions)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    @staticmethod
    def load_model(cfg: config.Config, fname: str, *args,
                   **kwargs) -> "CustomNeuralNetRegressor":
        """Loads a pre-trained model from a file.

        Args:
            cfg (config.Config): config file.
            fname (str): model filename (with .pt extension).
            *args & **kwargs: Additional arguments for model initialisation.
            
        Returns:
            CustomNeuralNetRegressor: loaded (and trained) model instance.
        """
        model = CustomNeuralNetRegressor(cfg, *args, **kwargs)
        model.initialize()
        model.load_params(f_params=_models_dir / fname)
        return model
