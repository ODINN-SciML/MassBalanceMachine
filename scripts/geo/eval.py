import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

import warnings
import matplotlib
import massbalancemachine as mbm
import logging
import torch
import json
import argparse
import pandas as pd
import numpy as np
import tqdm

from scripts.nongeo.utils import (
    getMetaData,
    buildArgs,
    trainValData,
    testData,
    setFeatures,
)

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser("Evaluate a model and save the figures.")
parser.add_argument("modelFolder", type=str, help="Folder of the model to load.")
parser.add_argument(
    "--cpu",
    dest="cpu",
    default=False,
    action="store_true",
    help="Force model to run on CPU, even if a GPU is available.",
)
parser.add_argument(
    "--plot",
    dest="plot",
    default=False,
    action="store_true",
    help="Display figures in addition to saving.",
)
parser.add_argument(
    "--noTest",
    dest="noTest",
    default=False,
    action="store_true",
    help="Do not evaluate on test data.",
)
parser.add_argument(
    "--onRegion",
    dest="onRegion",
    default=False,
    action="store_true",
    help="Evaluate prediction on the whole region in addition to classical plots.",
)
parser.add_argument(
    "--savePred",
    dest="savePred",
    default=False,
    action="store_true",
    help="Save predictions as CSV for further analysis or comparison.",
)
args = parser.parse_args()

modelFolder = args.modelFolder
cpu = args.cpu
plot = args.plot
noTest = args.noTest
onRegion = args.onRegion
savePred = args.savePred
pathFolder = os.path.join("logs", modelFolder)

if not plot:
    # To avoid GC issues because of the threads, we run the script without a GUI
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open(f"{pathFolder}/params.json", "r") as f:
    params = json.load(f)

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


if torch.cuda.is_available():
    print("CUDA is available")
    # free_up_cuda()
else:
    print("CUDA is NOT available")


# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


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


data_train = train_set["df_X"]
data_train["y"] = train_set["y"]
data_test = test_set["df_X"]
data_test["y"] = test_set["y"]

setFeatures(cfg, data_train, featuresInpModel)
df_X_train, y_train, df_X_val, y_val = trainValData(cfg, train_set, featuresInpModel)
df_X_test_subset = testData(cfg, test_set, featuresInpModel)


# dataset = dataset_val = None  # Initialized hereafter


# param_init = {"device": "cpu"}  # Use CPU for evaluation


network = mbm.models.buildModel(cfg, params=params)
model = mbm.models.CustomTorchNeuralNetRegressor(network)
device = torch.device("cuda:0" if torch.cuda.is_available() and not cpu else "cpu")
model = model.to(device)


# Load model and set to CPU
bestModelPath = mbm.training.loadBestModel(pathFolder, model)
print(f"Loaded model {bestModelPath}")
# loaded_model = mbm.models.CustomNeuralNetRegressor.load_model(
#     cfg,
#     pathFolder,
#     **{**args, **param_init},
# )
# model = model.set_params(device="cpu")
# model = model.to("cpu")


if len(df_X_test_subset) > 0 and not noTest:
    if sourceData == "switzerland":
        test_glaciers = list(data_test.GLACIER.unique())
    elif sourceData in ["iceland", "norway"]:
        test_glaciers = list(data_test.RGIId.unique())
    elif "wgms" in sourceData:
        test_glaciers = list(data_test.RGIId.unique())

    assert set(df_X_test_subset.RGIId.unique()) == set(test_glaciers)

    # Create dataloader
    test_gdl = mbm.dataloader.GeoDataLoader(
        cfg,
        test_glaciers,
        device=device,
        trainStakesDf=df_X_test_subset,
        months_head_pad=months_head_pad,
        months_tail_pad=months_tail_pad,
        keyGlacierSel="GLACIER" if sourceData == "switzerland" else "RGIId",
    )

    grouped_ids = model.evaluate_group_pred(test_gdl)
    scores = mbm.metrics.seasonal_scores(
        grouped_ids, target_col="target", pred_col="pred"
    )
    if "annual" in scores:
        scores_annual = {
            "rmse": scores["annual"]["rmse"],
            "r2": scores["annual"]["r2"],
            "bias": scores["annual"]["bias"],
        }
    else:
        scores_annual = None
    scores_winter = {
        "rmse": scores["winter"]["rmse"],
        "r2": scores["winter"]["r2"],
        "bias": scores["winter"]["bias"],
    }
    if "summer" in scores:
        scores_summer = {
            "rmse": scores["summer"]["rmse"],
            "r2": scores["summer"]["r2"],
            "bias": scores["summer"]["bias"],
        }
    else:
        scores_summer = None

    fig = mbm.plots.predVSTruthTimeSeries(
        grouped_ids=grouped_ids,
        scores_annual=scores_annual,
        scores_winter=scores_winter,
        scores_summer=scores_summer,
        ax_xlim=(-8, 6),
        ax_ylim=(-8, 6),
    )
    fig.savefig(f"{pathFolder}/prediction_test_PMB.pdf")
    if plot:
        plt.show()
    plt.close(fig)

    # submission_df = grouped_ids[["ID", "pred"]].sort_values(by="ID")
    # submission_df.rename(columns={"pred": "POINT_BALANCE"}, inplace=True)
    # # change 'ID' to string
    # submission_df["ID"] = submission_df["ID"].astype(str)
    # # save solution
    # submission_df.to_csv(f"{pathFolder}/submission.csv", index=False)

    # solution_df = grouped_ids[["ID", "target"]].sort_values(by="ID")
    # solution_df.rename(columns={"target": "POINT_BALANCE"}, inplace=True)
    # # change 'ID' to string
    # solution_df["ID"] = solution_df["ID"].astype(str)

    # # save solution
    # solution_df.to_csv(f"{pathFolder}/solution.csv", index=False)

    test_gl_per_el = {
        k: datasetManager.mean_stakes_elevation[k] for k in datasetManager.test_glaciers
    }
    test_gl_per_el = list(
        dict(sorted(test_gl_per_el.items(), key=lambda item: item[1])).keys()
    )

    grouped_ids["gl_elv"] = grouped_ids[keyGlacier].map(
        datasetManager.mean_stakes_elevation
    )

    fig = mbm.plots.predVSTruthPerGlacier(
        grouped_ids,
        custom_order=test_gl_per_el,
    )
    fig.savefig(f"{pathFolder}/individual_glaciers_test_PMB.pdf")
    if plot:
        plt.show()
    plt.close(fig)

    # Geodetic performance
    with torch.no_grad():
        resTest = mbm.training.assessOnTest(pathFolder, model, test_gdl)

    geoPred, geoTarget, geoErr, dict_df_gridded = mbm.training.eval_geodetic(
        model, test_gdl, return_grid_pred=["annual", "monthly"]
    )
    df_gridded_annual = dict_df_gridded["annual"]
    df_gridded_monthly = dict_df_gridded["monthly"]
    del dict_df_gridded
    if savePred:
        print("Saving gridded prediction for further analysis...")
        kk = geoTarget.keys()
        df_geo = pd.DataFrame(
            {
                "RGIId": kk,
                "target": [geoTarget[k] for k in kk],
                "err": [geoErr[k] for k in kk],
                "pred": [geoPred[k] for k in kk],
            }
        )
        df_geo.to_csv(f"{pathFolder}/gridded_geodetic_test.csv")
        df_gridded_annual.to_csv(f"{pathFolder}/gridded_annual_test.csv")
        df_gridded_monthly.to_csv(f"{pathFolder}/gridded_monthly_test.csv")
        del df_gridded_annual, df_gridded_monthly


if sourceData == "switzerland":
    train_glaciers = list(data_train.GLACIER.unique())
elif sourceData in ["iceland", "norway"]:
    train_glaciers = list(data_train.RGIId.unique())
elif "wgms" in sourceData:
    train_glaciers = list(data_train.RGIId.unique())

# Create dataloader
train_gdl = mbm.dataloader.GeoDataLoader(
    cfg,
    train_glaciers,
    device=device,
    trainStakesDf=data_train,
    months_head_pad=months_head_pad,
    months_tail_pad=months_tail_pad,
    keyGlacierSel="GLACIER" if sourceData == "switzerland" else "RGIId",
)

grouped_ids_train = model.evaluate_group_pred(train_gdl)
scores_train = mbm.metrics.seasonal_scores(
    grouped_ids_train, target_col="target", pred_col="pred"
)
scores_annual = {
    "rmse": scores_train["annual"]["rmse"],
    "r2": scores_train["annual"]["r2"],
    "bias": scores_train["annual"]["bias"],
}
scores_winter = {
    "rmse": scores_train["winter"]["rmse"],
    "r2": scores_train["winter"]["r2"],
    "bias": scores_train["winter"]["bias"],
}
if "summer" in scores_train:
    scores_summer = {
        "rmse": scores_train["summer"]["rmse"],
        "r2": scores_train["summer"]["r2"],
        "bias": scores_train["summer"]["bias"],
    }
else:
    scores_summer = None

fig = mbm.plots.predVSTruthTimeSeries(
    grouped_ids=grouped_ids_train,
    scores_annual=scores_annual,
    scores_winter=scores_winter,
    scores_summer=scores_summer,
    ax_xlim=(-14, 8),
    ax_ylim=(-14, 8),
)
fig.savefig(f"{pathFolder}/prediction_train_PMB.pdf")
if plot:
    plt.show()
plt.close(fig)

train_gl_per_el = {
    k: datasetManager.mean_stakes_elevation[k] for k in datasetManager.train_glaciers
}
train_gl_per_el = list(
    dict(sorted(train_gl_per_el.items(), key=lambda item: item[1])).keys()
)

grouped_ids_train["gl_elv"] = grouped_ids_train[keyGlacier].map(
    datasetManager.mean_stakes_elevation
)


scores = {}
for train_gl in datasetManager.train_glaciers:
    scores_glacier = mbm.metrics.seasonal_scores(
        grouped_ids_train[grouped_ids_train[keyGlacier] == train_gl],
        target_col="target",
        pred_col="pred",
    )
    scores[train_gl] = {"rmse": {}, "r2": {}, "bias": {}}
    if "annual" in scores_glacier:
        scores[train_gl]["rmse"]["a"] = scores_glacier["annual"]["rmse"]
        scores[train_gl]["r2"]["a"] = scores_glacier["annual"]["r2"]
        scores[train_gl]["bias"]["a"] = scores_glacier["annual"]["bias"]
    if "winter" in scores_glacier:
        scores[train_gl]["rmse"]["w"] = scores_glacier["winter"]["rmse"]
        scores[train_gl]["r2"]["w"] = scores_glacier["winter"]["r2"]
        scores[train_gl]["bias"]["w"] = scores_glacier["winter"]["bias"]
    if "summer" in scores_glacier:
        scores[train_gl]["rmse"]["s"] = scores_glacier["summer"]["rmse"]
        scores[train_gl]["r2"]["s"] = scores_glacier["summer"]["r2"]
        scores[train_gl]["bias"]["s"] = scores_glacier["summer"]["bias"]

fig = mbm.plots.predVSTruthPerGlacier(
    grouped_ids_train,
    scores=scores,
    custom_order=train_gl_per_el,
    hue="PERIOD",
)
fig.savefig(f"{pathFolder}/individual_glaciers_train_PMB.pdf")
if plot:
    plt.show()
plt.close(fig)


geoPred, geoTarget, geoErr, dict_df_gridded = mbm.training.eval_geodetic(
    model, train_gdl, return_grid_pred=["annual", "monthly"]
)
df_gridded_annual = dict_df_gridded["annual"]
df_gridded_monthly = dict_df_gridded["monthly"]
del dict_df_gridded
if savePred:
    print("Saving gridded prediction for further analysis...")
    kk = geoTarget.keys()
    df_geo = pd.DataFrame(
        {
            "RGIId": kk,
            "target": [geoTarget[k] for k in kk],
            "err": [geoErr[k] for k in kk],
            "pred": [geoPred[k] for k in kk],
        }
    )
    df_geo.to_csv(f"{pathFolder}/gridded_geodetic_train.csv")
    df_gridded_annual.to_csv(f"{pathFolder}/gridded_annual_train.csv")
    df_gridded_monthly.to_csv(f"{pathFolder}/gridded_monthly_train.csv")


# Geodetic performance
fig = mbm.plots.predVSTruthGlacierWide(
    geoTarget, geoPred, geoErr, title="Glacier wide MB on train"
)
plt.savefig(os.path.join(pathFolder, "geodetic_train.png"))
if plot:
    plt.show()
plt.close(fig)


# Plot MB profile
fig = mbm.plots.profilePerGlacier(
    df_gridded_annual, custom_order=train_gl_per_el
)  # , df_stakes=data_train)
fig.savefig(f"{pathFolder}/PMB_profile_individual_glaciers_train.pdf")
if plot:
    plt.show()
plt.close(fig)


# Plot cumulated mass change
fig = mbm.plots.cumulatedMassChange(
    df_gridded_monthly,
    geo={
        rgi_id: {"mean": geoTarget[rgi_id], "err": geoErr[rgi_id]}
        for rgi_id in geoTarget
    },
)
fig.savefig(f"{pathFolder}/cumulated_mass_change_glaciers_train.pdf")
if plot:
    plt.show()
plt.close(fig)

# TODO: since we changed the iterator, is the evaluation consistent on train/test ?

if onRegion:
    regionId = int(data_train.RGIId.unique()[0].split(".")[0].split("-")[1])
    thresArea = 1e6  # 1km²

    # Create dataloader
    region_gdl = mbm.dataloader.GeoDataLoader(
        cfg,
        train_glaciers,
        device=device,
        trainStakesDf=data_train,
        months_head_pad=months_head_pad,
        months_tail_pad=months_tail_pad,
        keyGlacierSel="GLACIER" if sourceData == "switzerland" else "RGIId",
        geoGlaciers=f"region-{regionId}-{thresArea}",
        ignoreGlaciers=["RGI60-08.00333", "RGI60-08.02308", "RGI60-08.02550"],
    )

    geoPred, geoTarget, geoErr, _ = mbm.training.eval_geodetic(model, region_gdl)

    # Geodetic performance
    fig = mbm.plots.predVSTruthGlacierWide(
        geoTarget, geoPred, geoErr, title="Glacier wide MB on the whole region"
    )
    plt.savefig(os.path.join(pathFolder, "geodetic_region.png"))
    if plot:
        plt.show()
    plt.close(fig)
