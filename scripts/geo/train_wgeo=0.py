import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

import torch
import json
import argparse
import massbalancemachine as mbm
from massbalancemachine.dataloader.GeoDataLoader import GeoDataLoader

from scripts.common import (
    trainTestGlaciers,
    getTrainTestSetsSwitzerland,
    _default_input,
    seed_all,
    loadParams,
)

parser = argparse.ArgumentParser()
parser.add_argument("modelType", type=str, help="Type of model to train")
args = parser.parse_args()

params = loadParams(args.modelType)
featuresInpModel = params["model"]["inputs"]

featuresToRemove = list(set(_default_input) - set(featuresInpModel))
metaData = list(
    set(
        [
            "RGIId",
            "POINT_ID",
            "ID",
            "GLWD_ID",
            "N_MONTHS",
            "MONTHS",
            "PERIOD",
            "GLACIER",
            "YEAR",
            "POINT_LAT",
            "POINT_LON",
        ]
    ).union(set(featuresToRemove))
)

cfg = mbm.SwitzerlandConfig(
    metaData=metaData,
    notMetaDataNotFeatures=["POINT_BALANCE"],
)
seed_all(cfg.seed)


train_glaciers, test_glaciers = trainTestGlaciers(params)

train_set, test_set, data_glamos, months_head_pad, months_tail_pad = (
    getTrainTestSetsSwitzerland(
        train_glaciers,
        test_glaciers,
        params,
        cfg,
        "CH_wgms_dataset_monthly_NN_geo.csv",
        process=False,
    )
)


# Validation and train split:
data_train = train_set["df_X"]
data_train["y"] = train_set["y"]
dataloader = mbm.dataloader.DataLoader(cfg, data=data_train)

feature_columns = list(
    data_train.columns.difference(cfg.metaData)
    .drop(cfg.notMetaDataNotFeatures)
    .drop("y")
)
assert set(feature_columns) == set(
    featuresInpModel
), f"Asked features are {featuresInpModel} but the one obtained from the dataframe are {feature_columns}"
cfg.setFeatures(feature_columns)


train_itr, val_itr = dataloader.set_train_test_split(test_size=0.2)

# Get all indices of the training and valing dataset at once from the iterators. Once called, the iterators are empty.
train_indices, val_indices = list(train_itr), list(val_itr)

df_X_train = data_train.iloc[train_indices]
y_train = df_X_train["POINT_BALANCE"].values

# Get val set
df_X_val = data_train.iloc[val_indices]
y_val = df_X_val["POINT_BALANCE"].values

assert all(data_train.POINT_BALANCE == train_set["y"])


all_columns = feature_columns + cfg.fieldsNotFeatures
print("Shape of training dataset:", df_X_train[all_columns].shape)
print("Shape of validation dataset:", df_X_val[all_columns].shape)
print("Shape of testing dataset:", test_set["df_X"][all_columns].shape)
print("Running with features:", feature_columns)


glaciers = list(data_train.GLACIER.unique())
gdl = GeoDataLoader(
    cfg,
    glaciers,
    trainStakesDf=df_X_train,
    months_head_pad=months_head_pad,
    months_tail_pad=months_tail_pad,
    valStakesDf=df_X_val,
)


network = mbm.models.buildModel(cfg, params=params)

model = mbm.models.CustomTorchNeuralNetRegressor(network)
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


data_test = test_set["df_X"]
data_test["y"] = test_set["y"]

gdl_test = GeoDataLoader(
    cfg,
    test_glaciers,
    trainStakesDf=data_test,
    months_head_pad=months_head_pad,
    months_tail_pad=months_tail_pad,
)

trainCfg = {
    "Nepochs": Nepochs,
    "wGeo": 0,
    "log_suffix": "wgeo=0_scaling",
    "scalingStakes": params["training"]["scalingStakes"],
}
ret = mbm.training.train_geo(
    model,
    gdl,
    optim,
    trainCfg,
    scheduler=scheduler,
    geodataloader_test=gdl_test,
)

print()
bestModelPath = mbm.training.loadBestModel(ret["misc"]["log_dir"], model)
print(f"Best model is {bestModelPath}")

resTest = mbm.training.assessOnTest(ret["misc"]["log_dir"], model, gdl_test)
print("Performance:")
print(
    json.dumps(
        json.loads(json.dumps(resTest), parse_float=lambda x: round(float(x), 3)),
        indent=2,
    )
)
