import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

import numpy as np
import torch
import json
import argparse
import massbalancemachine as mbm

from scripts.common import (
    loadParams,
)
from scripts.nongeo.utils import getMetaData, setFeatures, trainValData, testData

parser = argparse.ArgumentParser()
parser.add_argument("modelType", type=str, help="Type of model to train")
parser.add_argument("--load", type=str, default="", help="Model to load")
parser.add_argument(
    "--cpu",
    dest="cpu",
    default=False,
    action="store_true",
    help="Force model to run on CPU, even if a GPU is available.",
)
parser.add_argument(
    "--noTest",
    dest="noTest",
    default=False,
    action="store_true",
    help="Do not evaluate on the test set during training.",
)
parser.add_argument(
    "--time",
    dest="time",
    default=False,
    action="store_true",
    help="Evaluate loading and inference time.",
)
parser.add_argument(
    "--prof",
    dest="prof",
    default=False,
    action="store_true",
    help="Profile the code.",
)
parser.add_argument(
    "--wGeo",
    type=float,
    default=None,
    help="Weight of the geodetic term.",
)
args = parser.parse_args()

params = loadParams(args.modelType)
modelToLoad = args.load
cpu = args.cpu
noTest = args.noTest
timeExec = args.time
prof = args.prof
wGeo = args.wGeo
if wGeo is not None:  # Overwrite geodetic weight
    params["training"]["wGeo"] = wGeo
wGeo = params["training"]["wGeo"]
if params["training"]["log_suffix"] == "":
    params["training"]["log_suffix"] = f"wgeo={wGeo}" if wGeo > 0 else ""
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
        metaData=[
            "RGIId",
            "ID",
            "N_MONTHS",
            "MONTHS",
            "PERIOD",
            "YEAR",
            "POINT_ELEVATION",
        ],
        notMetaDataNotFeatures=["POINT_BALANCE", "svf"],
    )
elif "wgms" in sourceData:
    cfg = mbm.Config(
        metaData=[
            "RGIId",
            "ID",
            "N_MONTHS",
            "MONTHS",
            "PERIOD",
            "YEAR",
            "POINT_ELEVATION",
        ],
        notMetaDataNotFeatures=["POINT_BALANCE", "svf"],
    )
else:
    raise ValueError(f"source_data={sourceData} is unknown")


keyGlacier = "GLACIER" if sourceData == "switzerland" else "RGIId"
if sourceData == "switzerland":
    datasetManager = mbm.dataloader.SourceManagerSwitzerland(
        cfg, params, test_split_on=keyGlacier
    )
elif sourceData == "iceland":
    datasetManager = mbm.dataloader.SourceManagerIceland(
        cfg, params, test_split_on=keyGlacier
    )
elif sourceData == "norway":
    datasetManager = mbm.dataloader.SourceManagerNorway(
        cfg, params, test_split_on=keyGlacier
    )
elif "wgms" in sourceData:
    _split = sourceData.split(":")
    if len(_split) > 1:
        rgi_region = int(_split[1])
    else:
        rgi_region = None
    datasetManager = mbm.dataloader.SourceManagerWGMS(
        cfg, params, test_split_on="RGIId", rgi_region=rgi_region
    )
train_set, test_set, months_head_pad, months_tail_pad = datasetManager.train_test_sets()


# Validation and train split:
data_train = train_set["df_X"]
data_train["y"] = train_set["y"]

setFeatures(cfg, data_train, featuresInpModel)
df_X_train, y_train, df_X_val, y_val = trainValData(cfg, train_set, featuresInpModel)


print(
    "Shape of testing dataset:",
    test_set["df_X"][cfg.featureColumns + cfg.fieldsNotFeatures].shape,
)

device = torch.device("cuda:0" if torch.cuda.is_available() and not cpu else "cpu")

if sourceData == "switzerland":
    glaciers = list(data_train.GLACIER.unique())
    glaciersVal = list(df_X_val.GLACIER.unique())
elif sourceData in ["iceland", "norway"]:
    glaciers = list(data_train.RGIId.unique())
    glaciersVal = list(df_X_val.RGIId.unique())
elif "wgms" in sourceData:
    glaciers = list(data_train.RGIId.unique())
    glaciersVal = list(df_X_val.RGIId.unique())
gdl = mbm.dataloader.GeoDataLoader(
    cfg,
    glaciers,
    device=device,
    trainStakesDf=df_X_train,
    glacierListVal=glaciersVal,
    months_head_pad=months_head_pad,
    months_tail_pad=months_tail_pad,
    valStakesDf=df_X_val,
    keyGlacierSel="GLACIER" if sourceData == "switzerland" else "RGIId",
    preloadGeodetic=True,
)


network = mbm.models.buildModel(cfg, params=params)

model = mbm.models.CustomTorchNeuralNetRegressor(network)
model = model.to(device)

if modelToLoad != "":
    bestModelPath = mbm.training.loadBestModel(os.path.join("logs", modelToLoad), model)
    print(f"Loaded model {bestModelPath}")

optimType = params["training"]["optim"]
schedulerType = params["training"]["scheduler"]
lr = params["training"]["lr"]
momentum = params["training"]["momentum"]
beta1 = params["training"]["beta1"]
beta2 = params["training"]["beta2"]
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


data_test = testData(cfg, test_set, featuresInpModel)

if sourceData == "switzerland":
    test_glaciers = list(data_test.GLACIER.unique())
elif sourceData in ["iceland", "norway"]:
    test_glaciers = list(data_test.RGIId.unique())
elif "wgms" in sourceData:
    test_glaciers = list(data_test.RGIId.unique())
if not noTest:
    gdl_test = mbm.dataloader.GeoDataLoader(
        cfg,
        test_glaciers,
        device=device,
        trainStakesDf=data_test,
        months_head_pad=months_head_pad,
        months_tail_pad=months_tail_pad,
        keyGlacierSel="GLACIER" if sourceData == "switzerland" else "RGIId",
        preloadGeodetic=True,
    )
else:
    gdl_test = None

ret = mbm.training.train_geo(
    model,
    gdl,
    optim,
    params,
    scheduler=scheduler,
    geodataloader_test=gdl_test,
    timeExec=timeExec,
    useProfiler=prof,
)

print()
bestModelPath = mbm.training.loadBestModel(ret["misc"]["log_dir"], model)
print(f"Best model is {bestModelPath}")


# Save the best model in .json format along with the normalization values
model.eval()
st = model.module.state_dict()


class EncodeTensor(json.JSONEncoder, torch.utils.data.Dataset):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(EncodeTensor, self).default(obj)


norm = mbm.data_processing.Normalizer({k: cfg.bnds[k] for k in cfg.featureColumns})
norm_values = norm.export_bounds()
with open(os.path.join(ret["misc"]["log_dir"], "best_model.json"), "w") as f:
    info = {"norm": norm_values, "model": st, "inputs": cfg.featureColumns}
    json.dump(info, f, cls=EncodeTensor, sort_keys=True)
with open(os.path.join(ret["misc"]["log_dir"], "sample_inputs.json"), "w") as f:
    features, metadata, y = gdl.stakes(glaciers[0])
    with torch.no_grad():
        features_torch = torch.tensor(features.astype(np.float32)).to(gdl.device)
        pred = model.forward(features_torch)[:, 0]
    info = {
        "features": features.tolist(),
        "y": y.tolist(),
        "pred": pred.cpu().detach().numpy().tolist(),
    }
    json.dump(info, f, sort_keys=True)
X = gdl.trainStakesDf[
    gdl.trainStakesDf["GLACIER" if sourceData == "switzerland" else "RGIId"]
    == glaciers[0]
]
X.to_csv(os.path.join(ret["misc"]["log_dir"], "sample_inputs_before_norm.csv"))


if noTest:
    gdl_test = mbm.dataloader.GeoDataLoader(
        cfg,
        test_glaciers,
        device=device,
        trainStakesDf=data_test,
        months_head_pad=months_head_pad,
        months_tail_pad=months_tail_pad,
        keyGlacierSel="GLACIER" if sourceData == "switzerland" else "RGIId",
        preloadGeodetic=True,
    )

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
