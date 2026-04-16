import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

import warnings
import matplotlib.pyplot as plt
import massbalancemachine as mbm
import logging
import torch
import json
import argparse

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
    "--plot",
    dest="plot",
    default=False,
    action="store_true",
    help="Display figures in addition to saving.",
)
args = parser.parse_args()

modelFolder = args.modelFolder
plot = args.plot
pathFolder = os.path.join("logs", modelFolder)

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
        metaData=["RGIId", "ID", "N_MONTHS", "MONTHS", "PERIOD"],
        notMetaDataNotFeatures=["POINT_BALANCE", "YEAR"],
    )
elif "wgms" in sourceData:
    cfg = mbm.Config(
        metaData=["RGIId", "ID", "N_MONTHS", "MONTHS", "PERIOD"],
        notMetaDataNotFeatures=["POINT_BALANCE", "YEAR"],
    )
else:
    raise ValueError(f"source_data={sourceData} is unknown")


if torch.cuda.is_available():
    print("CUDA is available")
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

setFeatures(cfg, data_train, featuresInpModel)
df_X_train, y_train, df_X_val, y_val = trainValData(cfg, train_set, featuresInpModel)
df_X_test_subset = testData(cfg, test_set, featuresInpModel)


dataset = dataset_val = None  # Initialized hereafter


def my_train_split(ds, y=None, **fit_params):
    return dataset, dataset_val


param_init = {"device": "cpu"}  # Use CPU for evaluation


model = mbm.models.buildModel(cfg, params=params)


args = buildArgs(cfg, params, model, my_train_split)

# Load model and set to CPU
loaded_model = mbm.models.CustomNeuralNetRegressor.load_model(
    cfg,
    pathFolder,
    **{**args, **param_init},
)
loaded_model = loaded_model.set_params(device="cpu")
loaded_model = loaded_model.to("cpu")


if len(df_X_test_subset) > 0:
    grouped_ids, _, _, _ = loaded_model.evaluate_group_pred(
        df_X_test_subset,
        test_set["y"],
        months_head_pad,
        months_tail_pad,
        group_by=["YEAR", "PERIOD", keyGlacier],
    )
    scores = mbm.metrics.seasonal_scores(
        grouped_ids, target_col="target", pred_col="pred"
    )
    scores_annual = {
        "rmse": scores["annual"]["rmse"],
        "r2": scores["annual"]["r2"],
        "bias": scores["annual"]["bias"],
    }
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

    submission_df = grouped_ids[["ID", "pred"]].sort_values(by="ID")
    submission_df.rename(columns={"pred": "POINT_BALANCE"}, inplace=True)
    # change 'ID' to string
    submission_df["ID"] = submission_df["ID"].astype(str)
    # save solution
    submission_df.to_csv(f"{pathFolder}/submission.csv", index=False)

    solution_df = grouped_ids[["ID", "target"]].sort_values(by="ID")
    solution_df.rename(columns={"target": "POINT_BALANCE"}, inplace=True)
    # change 'ID' to string
    solution_df["ID"] = solution_df["ID"].astype(str)

    # save solution
    solution_df.to_csv(f"{pathFolder}/solution.csv", index=False)

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


grouped_ids_train, _, _, _ = loaded_model.evaluate_group_pred(
    data_train,
    data_train["POINT_BALANCE"].values,
    months_head_pad,
    months_tail_pad,
    group_by=["YEAR", "PERIOD", keyGlacier],
)
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
