import torch
import torch.nn as nn
import numpy as np
import yaml

def createModel(cfg, modelParams):
    nInp = len(cfg.featureColumns)
    if modelParams['type'] == 'sequential':
        assert len(modelParams['layers']) > 0
        l = [nn.Linear(nInp, modelParams['layers'][0])]
        for i in range(len(modelParams['layers'])-1):
            l.append(nn.ReLU())
            l.append(nn.Linear(modelParams['layers'][i], modelParams['layers'][i+1]))
        l.append(nn.ReLU())
        l.append(nn.Linear(modelParams['layers'][-1], 1))
        network = nn.Sequential(*l)
        return network
    else: raise ValueError(f"Model {modelParams['type']} is not supported.")

def selectModel(cfg, version):
    if version == 'minimalistic':
        paramsFile = 'minimalistic.yml'
    with open('scripts/netcfg/'+paramsFile) as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return createModel(cfg, params['model'])

def buildModel(cfg, version=None, params=None):
    assert (version is None) ^ (params is None), "Either version or params must be provided."
    if version is not None:
        model = selectModel(cfg, version)
    else:
        if 'model' in params:
            params = params['model']
        model = createModel(cfg, params)
    return model

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
