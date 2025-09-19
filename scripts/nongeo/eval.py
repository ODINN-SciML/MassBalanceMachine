import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

import warnings
import matplotlib.pyplot as plt
from cmcrameri import cm
import massbalancemachine as mbm
import logging
import torch
import json
import argparse

from scripts.common import (
    trainTestGlaciers,
    getTrainTestSetsSwitzerland,
    seed_all,
)
from scripts.nongeo.utils import (
    getMetaData,
    buildArgs,
    trainValData,
    testData,
    setFeatures,
)

from regions.Switzerland.scripts.helpers import get_cmap_hex
from regions.Switzerland.scripts.plots import (
    compute_seasonal_scores,
    plot_predictions_summary,
    predVSTruth,
    plotMeanPred,
    PlotIndividualGlacierPredVsTruth,
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

metaData = getMetaData(featuresInpModel)


cfg = mbm.SwitzerlandConfig(
    metaData=metaData,
    notMetaDataNotFeatures=["POINT_BALANCE"],
)
seed_all(cfg.seed)


# Plot styles:
path_style_sheet = "regions/Switzerland/scripts/example.mplstyle"
plt.style.use(path_style_sheet)
colors = get_cmap_hex(cm.batlow, 10)
color_dark_blue = colors[0]
color_pink = "#c51b7d"


if torch.cuda.is_available():
    print("CUDA is available")
    # free_up_cuda()
else:
    print("CUDA is NOT available")


# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

train_glaciers, test_glaciers = trainTestGlaciers(params)

train_set, test_set, data_glamos, months_head_pad, months_tail_pad = (
    getTrainTestSetsSwitzerland(
        train_glaciers,
        test_glaciers,
        params,
        cfg,
        "CH_wgms_dataset_monthly_NN_nongeo.csv",
        process=False,
    )
)

data_train = train_set["df_X"]
data_train["y"] = train_set["y"]

feature_columns = setFeatures(cfg, data_train, featuresInpModel)
df_X_train, y_train, df_X_val, y_val = trainValData(cfg, train_set, feature_columns)
df_X_test_subset = testData(cfg, test_set, feature_columns)


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


grouped_ids, scores_NN, ids_NN, y_pred_NN = loaded_model.evaluate_group_pred(
    df_X_test_subset,
    test_set["y"],
    months_head_pad,
    months_tail_pad,
)
scores_annual, scores_winter = compute_seasonal_scores(
    grouped_ids, target_col="target", pred_col="pred"
)
fig = plot_predictions_summary(
    grouped_ids=grouped_ids,
    scores_annual=scores_annual,
    scores_winter=scores_winter,
    predVSTruth=predVSTruth,
    plotMeanPred=plotMeanPred,
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


gl_per_el = (
    data_glamos[data_glamos.PERIOD == "annual"]
    .groupby(["GLACIER"])["POINT_ELEVATION"]
    .mean()
)
gl_per_el = gl_per_el.sort_values(ascending=False)

test_gl_per_el = gl_per_el[test_glaciers].sort_values().index

fig, axs = plt.subplots(3, 3, figsize=(20, 15), sharex=True)

PlotIndividualGlacierPredVsTruth(
    grouped_ids,
    axs=axs,
    color_annual=color_dark_blue,
    color_winter=color_pink,
    custom_order=test_gl_per_el,
)
fig.savefig(f"{pathFolder}/individual_glaciers_test_PMB.pdf")
if plot:
    plt.show()
plt.close(fig)


grouped_ids_NN_train, scores_NN_train, ids_train, y_pred_train = (
    loaded_model.evaluate_group_pred(
        data_train,
        data_train["POINT_BALANCE"].values,
        months_head_pad,
        months_tail_pad,
    )
)
scores_annual_NN, scores_winter_NN = compute_seasonal_scores(
    grouped_ids_NN_train, target_col="target", pred_col="pred"
)
fig = plot_predictions_summary(
    grouped_ids=grouped_ids_NN_train,
    scores_annual=scores_annual_NN,
    scores_winter=scores_winter_NN,
    predVSTruth=predVSTruth,
    plotMeanPred=plotMeanPred,
    ax_xlim=(-14, 8),
    ax_ylim=(-14, 8),
)
fig.savefig(f"{pathFolder}/prediction_train_PMB.pdf")
if plot:
    plt.show()
plt.close(fig)


train_gl_per_el = gl_per_el[train_glaciers].sort_values().index

fig, axs = plt.subplots(8, 3, figsize=(20, 30), sharex=False)

PlotIndividualGlacierPredVsTruth(
    grouped_ids_NN_train,
    axs=axs,
    color_annual=color_dark_blue,
    color_winter=color_pink,
    custom_order=train_gl_per_el,
    ax_xlim=None,
)
fig.savefig(f"{pathFolder}/individual_glaciers_train_PMB.pdf")
if plot:
    plt.show()
plt.close(fig)


# Copied up to the "Extrapolate in space" section
