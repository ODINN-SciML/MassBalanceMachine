import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yaml
import copy

from data_processing.Dataset import Normalizer


class TILikeModel(nn.Module):
    def __init__(self, modelParams, normalizing_bounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_labels = modelParams["inputs"]
        self.normalizing_bounds = normalizing_bounds
        self.ind_precip = self.input_labels.index("tp")
        self.ind_temp = self.input_labels.index("t2m")
        self.ind_elev_diff = self.input_labels.index("ELEVATION_DIFFERENCE")
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
        # self.inp_cor_T = ["t2m", "slhf", "sshf", "ssrd", "str"]
        self.inp_cor_T = modelParams["cor_T"]["inputs"]
        # self.inp_cor_fac = [
        #     "aspect",
        #     "fal",
        #     "slhf",
        #     "slope",
        #     "sshf",
        #     "ssrd",
        #     "str",
        #     "svf",
        # ]
        # self.inp_cor_fac = [
        #     "fal",
        #     "slhf",
        #     "sshf",
        #     "ssrd",
        #     "str",
        # ]
        self.inp_cor_acc = modelParams["cor_acc"]["inputs"]
        self.inp_cor_abl = modelParams["cor_abl"]["inputs"]
        self.ind_inp_cor_T = sorted(
            [self.input_labels.index(inp) for inp in self.inp_cor_T]
        )
        # self.ind_inp_cor_fac = sorted(
        #     [self.input_labels.index(inp) for inp in self.inp_cor_fac]
        # )
        self.ind_inp_cor_acc = sorted(
            [self.input_labels.index(inp) for inp in self.inp_cor_acc]
        )
        self.ind_inp_cor_abl = sorted(
            [self.input_labels.index(inp) for inp in self.inp_cor_abl]
        )
        self.beta1 = 1.4
        self.beta2 = 0.0049
        init_tau_P_s = 1.5
        init_tau_P_c = 1.0
        init_beta_pdd = 1.0
        self.tau_P_s = torch.nn.Parameter(
            torch.arctanh((torch.ones(1) * init_tau_P_s / 2) - 1)
        )
        self.tau_P_c = torch.nn.Parameter(torch.ones(1) * init_tau_P_c)
        self.beta_pdd = torch.nn.Parameter(
            torch.arctanh((torch.ones(1) * init_beta_pdd / 2) - 1)
        )
        self.tau_P_s.requires_grad = True
        self.tau_P_c.requires_grad = True
        self.beta_pdd.requires_grad = True

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

        cor_acc_params = modelParams["cor_acc"]
        cor_acc = [nn.Linear(len(self.inp_cor_acc), cor_acc_params["layers"][0])]
        for i in range(len(cor_acc_params["layers"]) - 1):
            cor_acc.append(nn.ReLU())
            cor_acc.append(
                nn.Linear(cor_acc_params["layers"][i], cor_acc_params["layers"][i + 1])
            )
        cor_acc.append(nn.ReLU())
        cor_acc.append(nn.Linear(cor_acc_params["layers"][-1], 1))
        cor_acc.append(nn.Softplus())
        self.cor_acc = nn.Sequential(*cor_acc)

        cor_abl_params = modelParams["cor_abl"]
        cor_abl = [nn.Linear(len(self.inp_cor_abl), cor_abl_params["layers"][0])]
        for i in range(len(cor_abl_params["layers"]) - 1):
            cor_abl.append(nn.ReLU())
            cor_abl.append(
                nn.Linear(cor_abl_params["layers"][i], cor_abl_params["layers"][i + 1])
            )
        cor_abl.append(nn.ReLU())
        cor_abl.append(nn.Linear(cor_abl_params["layers"][-1], 1))
        cor_abl.append(nn.Softplus())
        self.cor_abl = nn.Sequential(*cor_abl)

    def get_cor_T(self, inputs):
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
            # + 273.15
        )  # in Celsius degrees
        elev_diff_unorm = Normalizer._unorm(
            inputs[:, self.ind_elev_diff],
            self.normalizing_bounds["ELEVATION_DIFFERENCE"][0],
            self.normalizing_bounds["ELEVATION_DIFFERENCE"][1],
        )
        inp_cor_T = inputs[:, self.ind_inp_cor_T]
        inp_cor_acc = inputs[:, self.ind_inp_cor_acc]
        inp_cor_abl = inputs[:, self.ind_inp_cor_abl]

        # Temperature correction for precipitation and PDD
        # Acts as a shift of the temperature
        cor_T_val = self.cor_T(inp_cor_T)

        elev_diff = inputs[:, self.ind_elev_diff]
        cor_T = elev_diff * cor_T_val[:, 0] + cor_T_val[:, 1]

        P_solid = P * F.sigmoid(
            (torch.tanh(self.tau_P_s) + 1) * 2 * (self.tau_P_c - cor_T - T)
        )
        curv_pdd = (torch.tanh(self.beta_pdd) + 1) * 2
        PDD = F.softplus((T + cor_T) * curv_pdd) / curv_pdd

        cor_acc = self.cor_acc(inp_cor_acc).view(-1)
        cor_abl = self.cor_abl(inp_cor_abl).view(-1)

        return cor_T, P, T, P_solid, PDD, elev_diff_unorm, cor_acc, cor_abl

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
            # + 273.15
        )  # in Celsius degrees
        inp_cor_T = inputs[:, self.ind_inp_cor_T]
        inp_cor_acc = inputs[:, self.ind_inp_cor_acc]
        inp_cor_abl = inputs[:, self.ind_inp_cor_abl]

        # Temperature correction for precipitation and PDD
        # Acts as a shift of the temperature
        cor_T_val = self.cor_T(inp_cor_T)

        elev_diff = inputs[:, self.ind_elev_diff]
        cor_T = elev_diff * cor_T_val[:, 0] + cor_T_val[:, 1]

        P_solid = P * F.sigmoid(
            (torch.tanh(self.tau_P_s) + 1) * 2 * (self.tau_P_c - cor_T - T)
        )
        curv_pdd = (torch.tanh(self.beta_pdd) + 1) * 2
        PDD = F.softplus((T + cor_T) * curv_pdd) / curv_pdd

        # Accumulation factor correction
        cor_acc = self.cor_acc(inp_cor_acc).view(-1)

        # Ablation factor correction
        cor_abl = self.cor_abl(inp_cor_abl).view(-1)

        MB = self.beta1 * cor_acc * P_solid - self.beta2 * cor_abl * PDD
        return MB.view(-1, 1)


class GeodeticCorrectionModel(nn.Module):
    def __init__(
        self,
        modelParams,
        *args,
        glacioModule=None,
        geoModule=None,
        activateGlacio=False,
        train="geo",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.glacioModule = glacioModule
        self.geoModule = geoModule
        assert train in ["geo", "glacio", "joint"]
        self.moduleToTrain = train
        self.activateGlacio = activateGlacio
        self.input_labels = modelParams["inputs"]
        # self.normalizing_bounds = normalizing_bounds
        # self.inp_geo = ["ELEVATION_DIFFERENCE", "t2m", "svf"]
        self.inp_geo = modelParams["geo"]["inputs"]

        self.ind_inp_geo = sorted(
            [self.input_labels.index(inp) for inp in self.inp_geo]
        )

        # if self.geoModule is None:
        #     cor = [nn.Linear(len(self.inp_geo), modelParams["layers_geodetic"][0])]
        #     for i in range(len(modelParams["layers_geodetic"]) - 1):
        #         cor.append(nn.ReLU())
        #         cor.append(
        #             nn.Linear(modelParams["layers_geodetic"][i], modelParams["layers_geodetic"][i + 1])
        #         )
        #     cor.append(nn.ReLU())
        #     cor.append(nn.Linear(modelParams["layers_geodetic"][-1], 2))
        #     self.geoModule = nn.Sequential(*cor)

        # self.freeze_glacio = True

        # if self.freeze_glacio:
        if self.moduleToTrain == "geo":
            self.glacioModule.eval()
            for p in self.glacioModule.parameters():
                p.requires_grad = False
        elif self.moduleToTrain == "glacio":
            self.geoModule.eval()
            for p in self.geoModule.parameters():
                p.requires_grad = False

    def forward(self, inputs):
        inp_geo = inputs[:, self.ind_inp_geo]
        geoContrib = self.geoModule(inp_geo)
        if self.activateGlacio:
            return self.glacioModule(inputs) + geoContrib
        else:
            return geoContrib

    def glacioModuleOnly(self, inputs):
        return self.glacioModule(inputs)

    def train(self, mode: bool = True):
        if self.moduleToTrain == "geo":
            self.geoModule.train(mode)
            self.glacioModule.eval()
        elif self.moduleToTrain == "glacio":
            self.geoModule.eval()
            self.glacioModule.train(mode)
        elif self.moduleToTrain == "joint":
            self.geoModule.train(mode)
            self.glacioModule.train(mode)

    def eval(self):
        self.glacioModule.eval()
        self.geoModule.eval()


def createModel(cfg, modelParams, nInp=None, multi=None):
    nInp = nInp or len(cfg.featureColumns)
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
    if modelParams["type"] == "sequential_downscaled":
        assert len(modelParams["layers"]) > 0
        return SequentialDownscaledModel(modelParams, cfg.bnds)
    elif modelParams["type"] == "TIlike":
        return TILikeModel(modelParams, cfg.bnds)
    elif modelParams["type"] == "multi":
        tmp_params_glacio = copy.deepcopy(modelParams["glacio"])
        tmp_params_glacio["inputs"] = modelParams[
            "inputs"
        ]  # Copy inputs which are not defined for glacio
        glacioModule = createModel(cfg, tmp_params_glacio)
        geoModule = createModel(
            cfg, modelParams["geo"], nInp=len(modelParams["geo"]["inputs"])
        )
        return GeodeticCorrectionModel(
            modelParams, glacioModule=glacioModule, geoModule=geoModule, train=multi
        )
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


def buildModel(cfg, version=None, params=None, multi=None):
    assert (version is None) ^ (
        params is None
    ), "Either version or params must be provided."
    if version is not None:
        model = selectModel(cfg, version)
    else:
        if "model" in params:
            params = params["model"]
        model = createModel(cfg, params, multi=multi)
    return model


def aggrMetadata(metadata, groupByCol):
    """
    Aggregates metadata by taking the first value encountered in each
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
    if "GLWD_M_ID" in metadataKeys:
        aggMap["GLWD_M_ID"] = "first"
    if "POINT_LAT" in metadataKeys:
        aggMap["POINT_LAT"] = "first"
    if "POINT_LON" in metadataKeys:
        aggMap["POINT_LON"] = "first"
    if "PERIOD" in metadataKeys:
        aggMap["PERIOD"] = "first"
    if "POINT_ELEVATION" in metadataKeys:
        aggMap["POINT_ELEVATION"] = "first"
    if "ELEVATION_DIFFERENCE" in metadataKeys:
        aggMap["ELEVATION_DIFFERENCE"] = "first"
    metadataAggr = metadata.groupby(groupByCol).agg(aggMap)
    return metadataAggr


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
                grouped_ids_glacier = aggrMetadata(metadata, "ID_int")

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
