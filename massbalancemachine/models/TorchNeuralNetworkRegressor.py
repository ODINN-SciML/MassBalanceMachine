import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yaml

from data_processing.Dataset import Normalizer


class TILikeModel(nn.Module):
    def __init__(self, modelParams, normalizing_bounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_labels = modelParams["inputs"]
        self.normalizing_bounds = normalizing_bounds
        self.ind_precip = self.input_labels.index("tp")
        self.ind_temp = self.input_labels.index("t2m")
        # self.inputs = [
        #     "ELEVATION_DIFFERENCE",
        #     "aspect",
        #     "fal", # forcast albedo
        #     "slhf", # surface latent heat flux
        #     "slope",
        #     "sshf", # surface sensible heat flux
        #     "ssrd", # surface solar radiation downwards
        #     "str", # surface net thermal radiation
        #     "t2m", # 2m temperature
        #     "tp", # total precipitation
        #     "svf", # sky view factor
        # ]
        self.inp_cor_T = ["ELEVATION_DIFFERENCE", "t2m", "svf"]
        self.inp_cor_fac = [
            "aspect",
            "fal",
            "slhf",
            "slope",
            "sshf",
            "ssrd",
            "str",
            "t2m",
            "svf",
        ]
        self.ind_inp_cor_T = sorted(
            [self.input_labels.index(inp) for inp in self.inp_cor_T]
        )
        self.ind_inp_cor_fac = sorted(
            [self.input_labels.index(inp) for inp in self.inp_cor_fac]
        )
        self.beta1 = 1.4
        self.beta2 = 0.0049
        self.tau_P_s = 1.5
        self.tau_P_c = 1.0

        cor_T_params = modelParams["cor_T"]
        cor_T = [nn.Linear(len(self.inp_cor_T), cor_T_params["layers"][0])]
        for i in range(len(cor_T_params["layers"]) - 1):
            cor_T.append(nn.ReLU())
            cor_T.append(
                nn.Linear(cor_T_params["layers"][i], cor_T_params["layers"][i + 1])
            )
        cor_T.append(nn.ReLU())
        cor_T.append(nn.Linear(cor_T_params["layers"][-1], 2))
        self.cor_T = nn.Sequential(*cor_T)

        cor_fac_params = modelParams["cor_fac"]
        cor_fac = [nn.Linear(len(self.inp_cor_fac), cor_fac_params["layers"][0])]
        for i in range(len(cor_fac_params["layers"]) - 1):
            cor_fac.append(nn.ReLU())
            cor_fac.append(
                nn.Linear(cor_fac_params["layers"][i], cor_fac_params["layers"][i + 1])
            )
        cor_fac.append(nn.ReLU())
        cor_fac.append(nn.Linear(cor_fac_params["layers"][-1], 2))
        self.cor_fac = nn.Sequential(*cor_fac)

    def forward(self, inputs):
        P = Normalizer._unorm(
            inputs[:, self.ind_precip],
            self.normalizing_bounds["tp"][0],
            self.normalizing_bounds["tp"][1],
        )
        T = (
            Normalizer._unorm(
                inputs[:, self.ind_temp],
                self.normalizing_bounds["t2m"][0],
                self.normalizing_bounds["t2m"][1],
            )
            + 273.15
        )
        inp_cor_T = inputs[:, self.ind_inp_cor_T]
        inp_cor_fac = inputs[:, self.ind_inp_cor_fac]

        # Temperature correction for precipitation and PDD
        # Acts as a shift of the temperature
        # Accounts for: downscaling, local effects (?)
        cor_T = self.cor_T(inp_cor_T)
        cor_T_P = cor_T[:, 0]
        cor_T_PDD = cor_T[:, 1]

        P_solid = P * F.sigmoid(self.tau_P_s * (self.tau_P_c + cor_T_P - T))
        PDD = F.softplus(T + cor_T_PDD)

        cor_fac = self.cor_fac(inp_cor_fac)

        # Accumulation factor correction
        # Accounts for:
        #   - local effects: topography, exposure, radiative effects, refreeze
        cor_acc = cor_fac[:, 0]

        # Ablation factor correction
        # Accounts for:
        #   - local effects: topography, exposure, radiative effects, refreeze
        cor_abl = cor_fac[:, 1]

        MB = self.beta1 * cor_acc * P_solid - self.beta2 * cor_abl * PDD
        return MB.view(-1, 1)


def createModel(cfg, modelParams):
    nInp = len(cfg.featureColumns)
    dropout = modelParams.get("dropout", 0.0)
    if modelParams["type"] == "sequential":
        assert len(modelParams["layers"]) > 0
        l = [nn.Linear(nInp, modelParams["layers"][0])]
        for i in range(len(modelParams["layers"]) - 1):
            l.append(nn.ReLU())
            if dropout > 0:
                l.append(nn.Dropout(dropout))
            l.append(nn.Linear(modelParams["layers"][i], modelParams["layers"][i + 1]))
        l.append(nn.ReLU())
        if dropout > 0:
            l.append(nn.Dropout(dropout))
        l.append(nn.Linear(modelParams["layers"][-1], 1))
        network = nn.Sequential(*l)
        return network
    elif modelParams["type"] == "TIlike":
        return TILikeModel(modelParams, cfg.bnds)
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

    def evaluate_group_pred(self, geodataloader, val=False):
        grouped_ids = pd.DataFrame()
        with torch.no_grad():
            iterator = geodataloader.glaciersVal if val else geodataloader.glaciers
            stakeMethod = geodataloader.stakesVal if val else geodataloader.stakes
            for g in iterator():
                # Get input features, metadata and ground truth
                stakes, metadata, point_balance = stakeMethod(g)
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
                d = {
                    "target": trueMean.cpu(),
                    "ID_int": grouped_ids_glacier.index,
                    "pred": predSum.cpu(),
                    "PERIOD": grouped_ids_glacier.PERIOD,
                    "YEAR": grouped_ids_glacier.YEAR,
                    "RGIId": grouped_ids_glacier.RGIId,
                }
                if "POINT_ELEVATION" in grouped_ids_glacier.columns:
                    d["POINT_ELEVATION"] = grouped_ids_glacier.POINT_ELEVATION
                if "PERIOD" in grouped_ids_glacier.columns:
                    d["PERIOD"] = grouped_ids_glacier.PERIOD
                grouped_ids_glacier = pd.DataFrame(d)

                grouped_ids = pd.concat(
                    [grouped_ids, grouped_ids_glacier], ignore_index=True
                )

                if geodataloader.allStakesPerIter:
                    break

        if grouped_ids.shape[0] > 0:
            # ID_int does not make sense since it is used only to perform the aggregation with PyTorch, the variable to use is ID instead
            grouped_ids.drop(columns=["ID_int"], inplace=True)

        return grouped_ids
