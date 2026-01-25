import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

import torch
import json
import argparse
import massbalancemachine as mbm

from scripts.common import (
    # seed_all,
    loadParams,
)
from scripts.nongeo.utils import getMetaData, setFeatures, trainValData, testData


parser = argparse.ArgumentParser()
parser.add_argument("modelType", type=str, help="Type of model to train")
parser.add_argument("--load", type=str, default="", help="Model to load")
args = parser.parse_args()

params = loadParams(args.modelType)
modelToLoad = args.load
featuresInpModel = params["model"]["inputs"]
sourceData = params["training"]["source_data"]

metaData = getMetaData(featuresInpModel, sourceData)

if sourceData == "switzerland":
    cfg = mbm.SwitzerlandConfig(
        metaData=metaData,
        notMetaDataNotFeatures=["POINT_BALANCE"],
    )
elif sourceData == "iceland":
    cfg = mbm.Config(
        metaData=["RGIId", "POINT_ID", "ID", "N_MONTHS", "MONTHS", "PERIOD"]
    )
elif sourceData == "norway":
    cfg = mbm.Config(
        metaData=["RGIId", "ID", "N_MONTHS", "MONTHS", "PERIOD", "YEAR"],
        notMetaDataNotFeatures=["POINT_BALANCE", "svf"],
    )
else:
    raise ValueError(f"source_data={sourceData} is unknown")
# seed_all(cfg.seed)


keyGlacier = "GLACIER" if sourceData == "switzerland" else "RGIId"
if sourceData == "switzerland":
    datasetManager = mbm.dataloader.VeryPoorlyNamedClassSwitzerland(
        cfg, params, test_split_on=keyGlacier
    )
elif sourceData == "iceland":
    datasetManager = mbm.dataloader.VeryPoorlyNamedClassIceland(
        cfg, params, test_split_on=keyGlacier
    )
elif sourceData == "norway":
    datasetManager = mbm.dataloader.VeryPoorlyNamedClassNorway(
        cfg, params, test_split_on=keyGlacier
    )
train_set, test_set, months_head_pad, months_tail_pad = datasetManager.train_test_sets()


# Validation and train split:
data_train = train_set["df_X"]
data_train["y"] = train_set["y"]

feature_columns = setFeatures(cfg, data_train, featuresInpModel)
df_X_train, y_train, df_X_val, y_val = trainValData(cfg, train_set, feature_columns)


print(
    "Shape of testing dataset:",
    test_set["df_X"][cfg.featureColumns + cfg.fieldsNotFeatures].shape,
)


if sourceData == "switzerland":
    glaciers = list(data_train.GLACIER.unique())
elif sourceData in ["iceland", "norway"]:
    glaciers = list(data_train.RGIId.unique())
gdl = mbm.dataloader.GeoDataLoader(
    cfg,
    glaciers,
    trainStakesDf=df_X_train,
    months_head_pad=months_head_pad,
    months_tail_pad=months_tail_pad,
    valStakesDf=df_X_val,
    keyGlacierSel="GLACIER" if sourceData == "switzerland" else "RGIId",
    preloadGeodetic=True,
)


network = mbm.models.buildModel(cfg, params=params)

model = mbm.models.CustomTorchNeuralNetRegressor(network)

if modelToLoad != "":
    bestModelPath = mbm.training.loadBestModel(os.path.join("logs", modelToLoad), model)
    print(f"Loaded model {bestModelPath}")

optimType = params["training"]["optim"]
schedulerType = params["training"]["scheduler"]
lr = params["training"]["lr"]
momentum = params["training"]["momentum"]
beta1 = params["training"]["beta1"]
beta2 = params["training"]["beta2"]
Nepochs = params["training"]["Nepochs"]
weight_decay = params["training"]["weight_decay"]
if optimType == "ADAM":
    optim = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
    )
elif optimType == "SGD":
    optim = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
else:
    raise ValueError(f"Optimizer {optimType} is not supported.")

if schedulerType is None:
    scheduler = None
elif schedulerType == "StepLR":
    gamma = params["training"]["scheduler_gamma"]
    step_size = params["training"]["scheduler_step_size"]
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=gamma)
else:
    raise ValueError(f"Scheduler {schedulerType} is not supported.")


data_test = testData(cfg, test_set, feature_columns)

if sourceData == "switzerland":
    test_glaciers = list(data_test.GLACIER.unique())
elif sourceData in ["iceland", "norway"]:
    test_glaciers = list(data_test.RGIId.unique())
gdl_test = mbm.dataloader.GeoDataLoader(
    cfg,
    test_glaciers,
    trainStakesDf=data_test,
    months_head_pad=months_head_pad,
    months_tail_pad=months_tail_pad,
    keyGlacierSel="GLACIER" if sourceData == "switzerland" else "RGIId",
    preloadGeodetic=True,
)

trainCfg = {
    "Nepochs": Nepochs,
    # "wGeo": 0,
    # "log_suffix": "wgeo=0_scaling",
    "wGeo": 10,
    "log_suffix": "wgeo=10_scaling_clamp_debug",
    "scalingStakes": params["training"]["scalingStakes"],
}
ret = mbm.training.train_geo(
    model,
    gdl,
    optim,
    trainCfg,
    params,
    scheduler=scheduler,
    geodataloader_test=gdl_test,
)

print()
bestModelPath = mbm.training.loadBestModel(ret["misc"]["log_dir"], model)
print(f"Best model is {bestModelPath}")

model.eval()
with torch.no_grad():
    print("Computing performance on test set")
    resTest = mbm.training.assessOnTest(ret["misc"]["log_dir"], model, gdl_test)
print("Performance:")
print(
    json.dumps(
        json.loads(json.dumps(resTest), parse_float=lambda x: round(float(x), 3)),
        indent=2,
    )
)
