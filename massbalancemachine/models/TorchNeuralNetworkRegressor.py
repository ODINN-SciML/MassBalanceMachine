import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml


def createModel(cfg, modelParams):
    nInp = len(cfg.featureColumns)
    if modelParams["type"] == "sequential":
        assert len(modelParams["layers"]) > 0
        l = [nn.Linear(nInp, modelParams["layers"][0])]
        for i in range(len(modelParams["layers"]) - 1):
            l.append(nn.ReLU())
            l.append(nn.Linear(modelParams["layers"][i], modelParams["layers"][i + 1]))
        l.append(nn.ReLU())
        l.append(nn.Linear(modelParams["layers"][-1], 1))
        network = nn.Sequential(*l)
        return network
    else:
        raise ValueError(f"Model {modelParams['type']} is not supported.")


def selectModel(cfg, version):
    if version == "minimalistic":
        paramsFile = "minimalistic.yml"
    with open("scripts/netcfg/" + paramsFile) as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return createModel(cfg, params["model"])


def buildModel(cfg, version=None, params=None):
    assert (version is None) ^ (
        params is None
    ), "Either version or params must be provided."
    if version is not None:
        model = selectModel(cfg, version)
    else:
        if "model" in params:
            params = params["model"]
        model = createModel(cfg, params)
    return model


def aggrMetadataId(metadata, groupByCol):
    """
    Aggregates metadata temporally by taking the first value encountered in each
    aggregated group. These values are supposed to be unique per group.

    Args:
        metadata (pd.DataFrame): Input metadata to aggregate.
        groupByCol (str): The column to use for aggregation.

    Returns an aggregated pd.DataFrame.
    """
    metadataKeys = metadata.keys()
    aggMap = {"YEAR": "first", "ID": "first", "RGIId": "first"}
    if "GLWD_ID" in metadataKeys:
        aggMap["GLWD_ID"] = "first"
    if "POINT_LAT" in metadataKeys:
        aggMap["POINT_LAT"] = "first"
    if "POINT_LON" in metadataKeys:
        aggMap["POINT_LON"] = "first"
    if "PERIOD" in metadataKeys:
        aggMap["PERIOD"] = "first"
    if "POINT_ELEVATION" in metadataKeys:
        aggMap["POINT_ELEVATION"] = "first"
    metadataAggrId = metadata.groupby(groupByCol).agg(aggMap)
    return metadataAggrId


def aggrPredict(pred, idAggr, reduce="sum", out=None):
    """
    Performs temporal aggregation of the data.

    Args:
        pred (torch.Tensor): Predicted values
        idAggr (np.ndarray): Integer ID of the data used to aggregate them.
        reduce ('sum' or 'mean'): Reduction mode, default is 'sum'.

    Returns a torch.Tensor whose size is the number of unique IDs in idAggr.
    """
    assert isinstance(
        idAggr, (np.ndarray, torch.Tensor)
    ), "Argument idAggr must be either a numpy.ndarray or a torch.Tensor."
    assert isinstance(pred, torch.Tensor), "Argument predAggr must be a torch.Tensor."
    idAggrTorch = (
        torch.tensor(idAggr).to(pred.device)
        if isinstance(idAggr, np.ndarray)
        else idAggr
    )
    if out is None:
        out = torch.zeros(
            (len(np.unique(idAggr)),), device=pred.device, dtype=pred.dtype
        )
    # predSumAnnual = out.scatter_reduce(0, idAggrTorch, pred, reduce=reduce)
    predSumAnnual = out.scatter_reduce_(0, idAggrTorch, pred, reduce=reduce)
    return predSumAnnual  # This shares memory with out


def aggrMetadataGlwdId(metadata, groupByCol):
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
        YEAR=("YEAR", "first")  # Assumes YEAR is unique per GEOD_ID
    )  # .set_index('YEAR')
    return metadataAggrYear


def aggrPredictGlwd(pred, idAggr, out=None):
    """
    Performs spatial aggregation of the data glacier wide.

    Args:
        pred (torch.Tensor): Predicted values
        idAggr (np.ndarray): Integer ID of the data used to aggregate them.

    Returns a torch.Tensor whose size is the number of unique IDs in idAggr.
    """
    assert isinstance(
        idAggr, (np.ndarray, torch.Tensor)
    ), "Argument idAggr must be either a numpy.ndarray or a torch.Tensor."
    assert isinstance(pred, torch.Tensor), "Argument pred must be a torch.Tensor."
    idAggrTorch = (
        torch.tensor(idAggr).to(pred.device)
        if isinstance(idAggr, np.ndarray)
        else idAggr
    )
    if out is None:
        out = torch.zeros(
            (len(np.unique(idAggr)),), device=pred.device, dtype=pred.dtype
        )
    predSumAnnualGlwd = out.scatter_reduce_(
        0, idAggrTorch, pred, reduce="mean"
    )  # Aggregations of glacier wide values are always averaged
    return predSumAnnualGlwd  # This shares memory with out


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

    def cumulative_pred(self):
        # TODO: implement
        pass

    def evaluate_group_pred(self, geodataloader):
        grouped_ids = pd.DataFrame()
        with torch.no_grad():
            for g in geodataloader.glaciers():
                # Get input features, metadata and ground truth
                stakes, metadata, point_balance = geodataloader.stakes(g)
                idAggr = metadata["ID"].values

                # Make prediction
                stakesTorch = torch.tensor(stakes.astype(np.float32)).to(
                    geodataloader.device
                )
                pred = self.forward(stakesTorch)[:, 0]

                # Aggregate per stake and periods
                groundTruthTorch = torch.tensor(point_balance.astype(np.float32)).to(
                    geodataloader.device
                )
                int_id, unique_id = pd.factorize(idAggr)
                trueMean = aggrPredict(groundTruthTorch, int_id, reduce="mean")
                predSum = aggrPredict(pred, int_id)
                metadata = metadata.assign(ID_int=int_id)
                grouped_ids_glacier = aggrMetadataId(metadata, "ID_int")

                # Create grouped prediction DataFrame
                assert grouped_ids_glacier.index.name == "ID_int"
                grouped_ids_glacier = pd.DataFrame(
                    {
                        "target": trueMean.cpu(),
                        "ID_int": grouped_ids_glacier.index,
                        "pred": predSum.cpu(),
                        "PERIOD": grouped_ids_glacier.PERIOD,
                        "YEAR": grouped_ids_glacier.YEAR,
                        "RGIId": grouped_ids_glacier.RGIId,
                    }
                )

                grouped_ids = pd.concat(
                    [grouped_ids, grouped_ids_glacier], ignore_index=True
                )

        if grouped_ids.shape[0] > 0:
            # ID_int does not make sense since it is used only to perform the aggregation with PyTorch, the variable to use is ID instead
            grouped_ids.drop(columns=["ID_int"], inplace=True)

        return grouped_ids
