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
    already_completed_trial,
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
    "-s",
    "--suffix",
    type=str,
    default=None,
    help="Suffix to add to the folder that contains the model once trained.",
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
parser.add_argument(
    "--gridsearch",
    dest="gridsearch",
    default=[],
    nargs="+",
    help="Grid search configuration file (name only) and grid search name.",
)
args = parser.parse_args()

params = loadParams(args.modelType)
modelToLoad = args.load
cpu = args.cpu
suffix = args.suffix
noTest = args.noTest
timeExec = args.time
prof = args.prof
wGeo = args.wGeo

gridsearch = args.gridsearch
do_gridsearch = len(gridsearch) > 0
if do_gridsearch:
    assert len(gridsearch) == 2
    gridsearch_name = gridsearch[1]
    gridsearch_config = gridsearch[0]
    import yaml
    import optuna

    with open("scripts/netcfg/" + gridsearch_config + ".yml") as stream:
        try:
            gridsearch_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    def flatten_dict(d, parent_key="", sep="."):
        result = {}
        for key, value in d.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                result.update(flatten_dict(value, new_key, sep))
            else:
                if isinstance(value, list):
                    if isinstance(value[0], (tuple, list)):
                        result[new_key] = tuple(tuple(v) for v in value)
                    else:
                        result[new_key] = tuple(value)
                else:
                    result[new_key] = value
        return result

    search_space = flatten_dict(gridsearch_params)
    for k, v in search_space.items():
        if isinstance(v, (list, tuple)) and isinstance(v[0], (list, tuple)):
            search_space[k] = tuple([",".join([str(e) for e in t]) for t in v])
    print(f"{search_space=}")
    study = optuna.create_study(
        study_name=gridsearch_name,
        storage=optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(
                file_path="./journal_gridsearch.log"
            )
        ),
        sampler=optuna.samplers.GridSampler(search_space),
        direction="minimize",
        load_if_exists=True,
    )
    trial = study.ask()
    print(f"{trial.number=}")
    params["training"]["log_prefix"] = gridsearch_name + f"_{trial.number}"
    params["gridsearch"] = {
        "study_name": trial.study.study_name,
        "trial_number": trial.number,
        "search_space": search_space,
    }

    def recursive_update_from_flat(target, source, sep="."):
        for flat_key, value in source.items():
            parts = flat_key.split(sep)
            _set_recursive(target, parts, value)

    def _set_recursive(current, parts, value):
        key = parts[0]
        if len(parts) == 1:
            current[key] = value
            return
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        _set_recursive(current[key], parts[1:], value)

    candidate_params = {
        k: trial.suggest_categorical(k, v) for k, v in search_space.items()
    }
    for k, v in candidate_params.items():
        if isinstance(v, str):
            candidate_params[k] = [int(e) for e in v.split(",")]
    print(f"{candidate_params=}")
    exists, old_trial = already_completed_trial(study, candidate_params)
    if exists:
        print("This combination has already been tested.")
        print("metric:", old_trial.value)
        sys.exit(0)
    recursive_update_from_flat(params, candidate_params)
else:
    trial = None

if wGeo is not None:  # Overwrite geodetic weight
    params["training"]["wGeo"] = wGeo
wGeo = params["training"]["wGeo"]
if params["training"]["log_suffix"] == "":
    params["training"]["log_suffix"] = f"wgeo={wGeo}" if wGeo > 0 else ""
if suffix is not None:
    params["training"]["log_suffix"] += f"_{suffix}"
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
df_X_train, y_train, df_X_val, y_val = trainValData(
    cfg,
    train_set,
    featuresInpModel,
    split_key=params["training"].get("splitVal", "group-meas-id"),
)


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
    preloadGeodetic=wGeo > 0,
)


network = mbm.models.buildModel(cfg, params=params)

model = mbm.models.CustomTorchNeuralNetRegressor(network)
model = model.to(device)

if modelToLoad != "":
    bestModelPath, _ = mbm.training.loadBestModel(
        os.path.join("logs", modelToLoad), model
    )
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
        preloadGeodetic=wGeo > 0,
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
bestModelPath, bestVal = mbm.training.loadBestModel(ret["misc"]["log_dir"], model)
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
        preloadGeodetic=wGeo > 0,
    )

model.eval()
with torch.no_grad():
    print("Computing performance on test set")
    resTest = mbm.training.assessOnTest(ret["misc"]["log_dir"], model, gdl_test)
    with open(os.path.join(ret["misc"]["log_dir"], "perf.json"), "w") as f:
        json.dump({"test": resTest, "val": resVal}, f, indent=4)
    resVal = mbm.training.assessOnVal(model, gdl, params)
print("Performance:")
print(
    json.dumps(
        json.loads(
            json.dumps({"test": resTest, "val": resVal}),
            parse_float=lambda x: round(float(x), 3),
        ),
        indent=2,
    )
)

if trial is not None:
    trial.set_user_attr("log_dir", ret["misc"]["log_dir"])
    trial.set_user_attr("r2", resVal["r2"])
    trial.set_user_attr("bias", resVal["bias"])
    trial.set_user_attr("lossValStake", resVal["lossValStake"])
    trial.set_user_attr("lossValGeo", resVal["lossValGeo"])
    trial.set_user_attr("lossVal", resVal["lossVal"])
    trial.set_user_attr("rmse", resVal["rmse"])
    study.tell(trial, float(bestVal))
