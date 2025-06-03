import torch
import torch.nn as nn
import numpy as np

class CustomTorchNeuralNetRegressor(nn.Module):
    """
    Custom Torch neural network regressor that supports geodetic data aggregation.

    Args:
        module (torch.Module): Neural nework architecture with its associated weights.
    """
    def __init__(self, module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module = module
    def forward(self, x):
        """
        Forward evaluation of the model.
        """
        return self.module(x)
    def predict(self):
        # TODO: implement
        pass
    def evalMetrics(self):
        # TODO: implement
        pass

    def aggrPredict(self, pred, idAggr, reduce='sum'):
        """
        Performs temporal aggregation of the data.

        Args:
            pred (torch.Tensor): Predicted values
            idAggr (np.ndarray): Integer ID of the data used to aggregate them.
            reduce ('sum' or 'mean'): Reduction mode, default is 'sum'.

        Returns a torch.Tensor whose size is the number of unique IDs in idAggr.
        """
        assert isinstance(idAggr, np.ndarray), "Argument idAggr must be a numpy.ndarray."
        assert isinstance(pred, torch.Tensor), "Argument predAggr must be a torch.Tensor."
        idAggrTorch = torch.tensor(idAggr)
        out = torch.zeros((len(np.unique(idAggr)), ), device=pred.device, dtype=pred.dtype)
        predSumAnnual = out.scatter_reduce(0, idAggrTorch, pred, reduce=reduce)
        return predSumAnnual

    def aggrPredictGlwd(self, pred, idAggr):
        """
        Performs spatial aggregation of the data glacier wide.

        Args:
            pred (torch.Tensor): Predicted values
            idAggr (np.ndarray): Integer ID of the data used to aggregate them.

        Returns a torch.Tensor whose size is the number of unique IDs in idAggr.
        """
        assert isinstance(idAggr, np.ndarray), "Argument idAggr must be a numpy.ndarray."
        assert isinstance(pred, torch.Tensor), "Argument pred must be a torch.Tensor."
        idAggrTorch = torch.tensor(idAggr)
        out = torch.zeros((len(np.unique(idAggr)), ), device=pred.device, dtype=pred.dtype)
        predSumAnnualGlwd = out.scatter_reduce(0, idAggrTorch, pred, reduce='mean') # Aggregations of glacier wide values are always averaged
        return predSumAnnualGlwd

    def cumulative_pred(self):
        # TODO: implement
        pass

    def aggrMetadataId(self, metadata, groupByCol):
        """
        Aggregates metadata temporally by taking the first value encountered in each
        aggregated group. These values are supposed to be unique per group.

        Args:
            metadata (pd.DataFrame): Input metadata to aggregate.
            groupByCol (str): The column to use for aggregation.

        Returns an aggregated pd.DataFrame.
        """
        metadataAggrId = metadata.groupby(groupByCol).agg({
            'YEAR': 'first',
            'POINT_LAT': 'first',
            'POINT_LON': 'first',
            'GLWD_ID_int': 'first',
        })
        return metadataAggrId

    def aggrMetadataGlwdId(self, metadata, groupByCol):
        """
        Performs the glacier wide aggregation of the metadata by taking the first
        value encountered in each aggregated group. These values are supposed to be
        unique per group.

        Args:
            metadata (pd.DataFrame): Input metadata to aggregate.
            groupByCol (str): The column to use for aggregation.

        Returns an aggregated pd.DataFrame.
        """
        metadataAggrYear = metadata.groupby(groupByCol).agg(
            YEAR=('YEAR', 'first') # Assumes YEAR is unique per GEOD_ID
        )#.set_index('YEAR')
        return metadataAggrYear
