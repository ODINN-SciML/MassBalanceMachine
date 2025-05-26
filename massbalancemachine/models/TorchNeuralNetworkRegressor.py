import torch
import torch.nn as nn
import numpy as np

class CustomTorchNeuralNetRegressor(nn.Module):
    def __init__(self, module, *args, **kwargs):
        # TODO: should we move module into args?
        super().__init__(*args, **kwargs)
        self.module = module
    def forward(self, x):
        # Should support torch.Tensor and optionally df
        # Takes input features not concatenated together
        # Returns prediction not aggregated
        # The loss is reponsible of aggregating the terms with respect to glacier and time indices
        # Dataloader returns tensors of features + relevant metadata
        return self.module(x)
    def predict(self):
        pass
    def evalMetrics(self):
        pass

    def aggrPredict(self, pred, idAggr, reduce='sum'):
        assert isinstance(idAggr, np.ndarray), "Argument idAggr must be a numpy.ndarray."
        assert isinstance(pred, torch.Tensor), "Argument predAggr must be a torch.Tensor."
        idAggrTorch = torch.tensor(idAggr)
        out = torch.zeros((len(np.unique(idAggr)), ), device=pred.device, dtype=pred.dtype)
        predSumAnnual = out.scatter_reduce(0, idAggrTorch, pred, reduce=reduce)
        return predSumAnnual

    def aggrPredictGlwd(self, predAggr, idAggr):
        assert isinstance(idAggr, np.ndarray), "Argument idAggr must be a numpy.ndarray."
        assert isinstance(predAggr, torch.Tensor), "Argument predAggr must be a torch.Tensor."
        idAggrTorch = torch.tensor(idAggr)
        out = torch.zeros((len(np.unique(idAggr)), ), device=predAggr.device, dtype=predAggr.dtype)
        predSumAnnualGlwd = out.scatter_reduce(0, idAggrTorch, predAggr, reduce='mean')
        return predSumAnnualGlwd

    def cumulative_pred(self):
        pass

    def _create_features_metadata(self):
        pass

    def aggrMetadataId(self, metadata, groupByCol):
        metadataAggrId = metadata.groupby(groupByCol).agg({
            'YEAR': 'first',
            'POINT_LAT': 'first',
            'POINT_LON': 'first',
            'GLWD_ID_int': 'first',
        })
        return metadataAggrId

    def aggrMetadataGlwdId(self, metadata, groupByCol):
        metadataAggrYear = metadata.groupby(groupByCol).agg(
            YEAR=('YEAR', 'first') # Assumes YEAR is unique per GEOD_ID
        )#.set_index('YEAR')
        return metadataAggrYear
