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
from scripts.common import default_glacier_name

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
parser.add_argument(
    "-m",
    "--multi",
    type=str,
    default=None,
    help="Component of the multistage network to train.",
)
parser.add_argument(
    "--pgo",
    dest="pgo",
    default=False,
    action="store_true",
    help="Evaluate on PGO grid.",
)
parser.add_argument(
    "-c",
    "--color",
    type=str,
    default="blue",
    help="Color to use for most of the plots.",
)
parser.add_argument(
    "--maps",
    dest="maps",
    default=[],
    nargs="+",
    help="Generate annual MB maps for specific glaciers.",
)
parser.add_argument(
    "--years",
    dest="years",
    default=[],
    nargs="+",
    help="Years for which to compute the distributed MB. This is also used to generate the annual MB maps.",
)
args = parser.parse_args()

modelFolder = args.modelFolder
cpu = args.cpu
plot = args.plot
noTest = args.noTest
onRegion = args.onRegion
savePred = args.savePred
multi = args.multi
pgo = args.pgo
color = args.color
maps = args.maps
yearsMaps = [int(y) for y in args.years]
pathFolder = os.path.join("logs", modelFolder)

if len(maps) > 0:
    assert (
        len(yearsMaps) > 0
    ), "If distributed maps are generated, the option years must be provided."

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


# Dataset manager
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
split_key = params["training"].get("splitVal", "group-meas-id")
val_glaciers = params["training"].get("val_glaciers", None)
df_X_train, y_train, df_X_val, y_val = trainValData(
    cfg,
    train_set,
    featuresInpModel,
    split_key=split_key,
    val_glaciers=val_glaciers,
)
df_X_test_subset = testData(cfg, test_set, featuresInpModel)


geodeticYears = list(range(2000, 2020))
additionalYears = list(set(yearsMaps).difference(geodeticYears))
# TODO: change lines above


# dataset = dataset_val = None  # Initialized hereafter


# param_init = {"device": "cpu"}  # Use CPU for evaluation


# Create model
network = mbm.models.buildModel(cfg, params=params, multi=multi)

if multi is not None:
    network.moduleToTrain = multi
    if multi == "geo":
        network.activateGlacio = False
    elif multi == "glacio":
        network.activateGlacio = True
    elif multi == "joint":
        network.activateGlacio = True
    else:
        raise ValueError("Option multi should be set either to 'glacio' or 'geo'.")

model = mbm.models.CustomTorchNeuralNetRegressor(network)
device = torch.device("cuda:0" if torch.cuda.is_available() and not cpu else "cpu")
model = model.to(device)


# Load model and set to CPU
bestModelPath, _ = mbm.training.loadBestModel(pathFolder, model)
print(f"Loaded model {bestModelPath}")
# loaded_model = mbm.models.CustomNeuralNetRegressor.load_model(
#     cfg,
#     pathFolder,
#     **{**args, **param_init},
# )
# model = model.set_params(device="cpu")
# model = model.to("cpu")

if pgo:
    pathFolderPGO = os.path.join(pathFolder, "PGO")
    os.makedirs(pathFolderPGO, exist_ok=True)
    # pgo_glaciers = ["RGI2000-v7.0-G-11-03575", "RGI2000-v7.0-G-11-03576"]
    pgo_glaciers = None
    from massbalancemachine.data_processing.pgo import pgo_target_file

    df = mbm.data_processing.geodetic_target_PGO(pgo_target_file())
    area = df.a_m2 / 1e6
    df = df[area > 1]
    # pgo_glaciers = df.RGIId.iloc[:10].values
    pgo_glaciers = df.RGIId.values
    # pgo_glaciers = ["RGI2000-v7.0-G-11-04025", "RGI2000-v7.0-G-11-03872"]
    # pgo_glaciers = ["RGI2000-v7.0-G-11-04020", "RGI2000-v7.0-G-11-03872"]
    del df

    # Create dataloader
    pgo_gdl = mbm.dataloader.GeoDataLoader(
        cfg,
        pgo_glaciers,
        device=device,
        trainStakesDf=None,
        months_head_pad=months_head_pad,
        months_tail_pad=months_tail_pad,
        geodeticSource="PGO",
        keyGlacierSel="RGIId",
        allStakesPerIter=(params["training"]["scalingStakes"] == "full"),
        # additionalYears=additionalYears,
        # geodeticSource=params["training"]["geodetic_source"],
    )
    pathFolderPred = f"{pathFolderPGO}/pred"
    if savePred:
        os.makedirs(pathFolderPred, exist_ok=True)

    def callback_save_geodetic_annual(g, df):
        df.to_parquet(
            f"{pathFolderPred}/annual_{g}.parquet",
            engine="pyarrow",
            compression="snappy",
        )

    def callback_save_geodetic_monthly(g, df):
        df.to_parquet(
            f"{pathFolderPred}/monthly_{g}.parquet",
            engine="pyarrow",
            compression="snappy",
        )

    geoPred, geoTarget, geoErr, dict_df_gridded = mbm.training.eval_geodetic(
        model,
        pgo_gdl,
        return_grid_pred=["annual", "monthly"],
        callback_annual=(callback_save_geodetic_annual if savePred else None),
        callback_monthly=(callback_save_geodetic_monthly if savePred else None),
    )
    df_gridded_annual = dict_df_gridded["annual"]
    df_gridded_monthly = dict_df_gridded["monthly"]
    del dict_df_gridded
    if savePred:
        print("Saving gridded prediction...")
        kk = geoTarget.keys()
        df_geo = pd.DataFrame(
            {
                "RGIId": kk,
                "target": [geoTarget[k] for k in kk],
                "err": [geoErr[k] for k in kk],
                "pred": [geoPred[k] for k in kk],
            }
        )
        df_geo.to_parquet(
            f"{pathFolderPGO}/gridded_geodetic_pgo.parquet",
            engine="pyarrow",
            compression="snappy",
        )
        df_gridded_annual.to_parquet(
            f"{pathFolderPGO}/gridded_annual_pgo.parquet",
            engine="pyarrow",
            compression="snappy",
        )
        df_gridded_monthly.to_parquet(
            f"{pathFolderPGO}/gridded_monthly_pgo.parquet",
            engine="pyarrow",
            compression="snappy",
        )

        with open(os.path.join(pathFolderPGO, "periodsPerGlacier.json"), "w") as f:
            json.dump(
                {
                    rgi_id: [
                        (
                            np.datetime_as_string(
                                pgo_gdl.periods_per_glacier[rgi_id][0][0]
                            ),
                            np.datetime_as_string(
                                pgo_gdl.periods_per_glacier[rgi_id][0][1]
                            ),
                        )
                    ]
                    for rgi_id in pgo_gdl.periods_per_glacier.keys()
                },
                f,
                indent=4,
                sort_keys=True,
            )

    # Plot cumulated mass change
    fig, _ = mbm.plots.cumulatedMassChange(
        df_gridded_monthly,
        geo={
            rgi_id: {
                "mean": geoTarget[rgi_id],
                "err": geoErr[rgi_id],
                "start": pgo_gdl.periods_per_glacier[rgi_id][0][0],
                "end": pgo_gdl.periods_per_glacier[rgi_id][0][1],
            }
            for rgi_id in geoTarget
        },
    )
    fig.savefig(f"{pathFolderPGO}/cumulated_mass_change_glaciers_test.pdf")
    if plot:
        plt.show()
    plt.close(fig)

    # Geodetic performance
    fig = mbm.plots.predVSTruthGlacierWide(
        geoTarget,
        geoPred,
        geoErr,
        title="Glacier wide MB on PGO",
        ax_xlim=(-2.5, 1.0),
        ax_ylim=(-2.5, 1.0),
        legend=False,
        color=color,
    )
    plt.savefig(os.path.join(pathFolderPGO, "geodetic_pgo.png"))
    if plot:
        plt.show()
    plt.close(fig)

    # if any([m in test_glaciers for m in maps]):
    #     mapsFolder = f"{pathFolderPGO}/maps"
    #     os.makedirs(mapsFolder, exist_ok=True)
    #     # Initialize OGGM once for all to avoid repeated and useless computations
    #     mbm.data_processing.oggm_utils._initialize_oggm_config("")
    #     rgi_ids = list(set(test_glaciers).intersection(set(maps)))
    #     gdirs = mbm.data_processing.oggm_utils._initialize_glacier_directories(
    #         rgi_ids, cfg
    #     )
    #     for rgi_id, gdir in zip(rgi_ids, gdirs):
    #         years = df_gridded_annual[df_gridded_annual.RGIId == rgi_id].YEAR.unique()
    #         for year in yearsMaps:
    #             # TODO: allow to generate maps outside of that range
    #             assert year in years
    #             fig = mbm.plots.mapGlacier(
    #                 df_gridded_annual, rgi_id, year, cfg, gdir=gdir
    #             )
    #             fig.savefig(f"{mapsFolder}/{rgi_id}_{year}.pdf")
    #             plt.close(fig)
    del df_gridded_annual, df_gridded_monthly
    assert False


test_glacierNames = {}
if len(df_X_test_subset) > 0 and not noTest:
    if sourceData == "switzerland":
        test_glaciers = params["training"].get("test_glaciers_geo") or list(
            data_test.GLACIER.unique()
        )
    elif sourceData in ["iceland", "norway"]:
        test_glaciers = params["training"].get("test_glaciers_geo") or list(
            data_test.RGIId.unique()
        )
    elif "wgms" in sourceData:
        test_glaciers = params["training"].get("test_glaciers_geo") or list(
            data_test.RGIId.unique()
        )
    test_glacierNames = mbm.data_processing.oggm_utils._glacier_name(
        list(data_test.RGIId.unique()), cfg
    )

    # assert set(df_X_test_subset.RGIId.unique()) == set(test_glaciers)

    # Create dataloader
    test_gdl = mbm.dataloader.GeoDataLoader(
        cfg,
        test_glaciers,
        device=device,
        trainStakesDf=df_X_test_subset,
        months_head_pad=months_head_pad,
        months_tail_pad=months_tail_pad,
        keyGlacierSel="GLACIER" if sourceData == "switzerland" else "RGIId",
        allStakesPerIter=(params["training"]["scalingStakes"] == "full"),
        additionalYears=additionalYears,
        geodeticSource=params["training"].get(
            "geodetic_source", "Hugonnet21"
        ),  # TODO: change
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
        precLegend=2,
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

    test_glaciers_with_stakes = set(datasetManager.test_glaciers).intersection(
        set(datasetManager.mean_stakes_elevation.keys())
    )
    test_gl_per_el = {
        k: datasetManager.mean_stakes_elevation[k] for k in test_glaciers_with_stakes
    }
    test_gl_per_el = list(
        dict(sorted(test_gl_per_el.items(), key=lambda item: item[1])).keys()
    )

    grouped_ids["gl_elv"] = grouped_ids[keyGlacier].map(
        datasetManager.mean_stakes_elevation
    )
    if savePred:
        print("Saving stakes prediction...")
        grouped_ids.to_parquet(
            f"{pathFolder}/stakes_test.parquet",
            engine="pyarrow",
            compression="snappy",
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
        resTest = mbm.training.assessOnTest(
            pathFolder, model, test_gdl, params, color=color
        )

    pathFolderPred = f"{pathFolder}/pred"
    if savePred:
        os.makedirs(pathFolderPred, exist_ok=True)

    def callback_save_geodetic_annual(g, df):
        df.to_parquet(
            f"{pathFolderPred}/annual_{g}.parquet",
            engine="pyarrow",
            compression="snappy",
        )

    def callback_save_geodetic_monthly(g, df):
        df.to_parquet(
            f"{pathFolderPred}/monthly_{g}.parquet",
            engine="pyarrow",
            compression="snappy",
        )

    geoPred, geoTarget, geoErr, dict_df_gridded = mbm.training.eval_geodetic(
        model,
        test_gdl,
        return_grid_pred=["annual", "monthly"],
        callback_annual=(callback_save_geodetic_annual if savePred else None),
        callback_monthly=(callback_save_geodetic_monthly if savePred else None),
    )
    df_gridded_annual = dict_df_gridded["annual"]
    df_gridded_monthly = dict_df_gridded["monthly"]
    del dict_df_gridded
    if savePred:
        print("Saving gridded prediction...")
        kk = geoTarget.keys()
        df_geo = pd.DataFrame(
            {
                "RGIId": kk,
                "target": [geoTarget[k] for k in kk],
                "err": [geoErr[k] for k in kk],
                "pred": [geoPred[k] for k in kk],
            }
        )
        df_geo.to_parquet(
            f"{pathFolder}/gridded_geodetic_test.parquet",
            engine="pyarrow",
            compression="snappy",
        )
        df_gridded_annual.to_parquet(
            f"{pathFolder}/gridded_annual_test.parquet",
            engine="pyarrow",
            compression="snappy",
        )
        df_gridded_monthly.to_parquet(
            f"{pathFolder}/gridded_monthly_test.parquet",
            engine="pyarrow",
            compression="snappy",
        )

    # Plot MB profile
    # TODO: ignore years outside of the geodetic time window
    fig = mbm.plots.profilePerGlacier(
        df_gridded_annual,
        custom_order=test_gl_per_el,
        # titles={
        #     k: (f"{k} ({glacierNames[k]})" if glacierNames[k] is not None else None)
        #     for k in glacierNames
        # },
        df_stakes=grouped_ids,
        average_stakes=False,
    )
    fig.savefig(f"{pathFolder}/PMB_profile_individual_glaciers_test.pdf")
    if plot:
        plt.show()
    plt.close(fig)

    # # Plot MB profile per month
    # # TODO: ignore years outside of the geodetic time window
    # fig = mbm.plots.profilePerGlacierPerMonth(
    #     TO_LOAD,
    #     custom_order=test_gl_per_el,
    #     # titles={
    #     #     k: (f"{k} ({glacierNames[k]})" if glacierNames[k] is not None else None)
    #     #     for k in glacierNames
    #     # },
    # )
    # fig.savefig(f"{pathFolder}/PMB_profile_monthly_individual_glaciers_test.pdf")
    # if plot:
    #     plt.show()
    # plt.close(fig)
    # assert False

    # Plot cumulated mass change
    fig, _ = mbm.plots.cumulatedMassChange(
        df_gridded_monthly,
        geo={
            rgi_id: {
                "mean": geoTarget[rgi_id],
                "err": geoErr[rgi_id],
                "start": test_gdl.geodetic_periods(rgi_id)[0][0],
                "end": test_gdl.geodetic_periods(rgi_id)[0][1],
            }
            for rgi_id in geoTarget
        },
    )
    fig.savefig(f"{pathFolder}/cumulated_mass_change_glaciers_test.pdf")
    if plot:
        plt.show()
    plt.close(fig)

    if any([m in test_glaciers for m in maps]):
        mapsFolder = f"{pathFolder}/maps"
        os.makedirs(mapsFolder, exist_ok=True)
        # Initialize OGGM once for all to avoid repeated and useless computations
        mbm.data_processing.oggm_utils._initialize_oggm_config("")
        rgi_ids = list(set(test_glaciers).intersection(set(maps)))
        gdirs = mbm.data_processing.oggm_utils._initialize_glacier_directories(
            rgi_ids, cfg
        )
        for rgi_id, gdir in zip(rgi_ids, gdirs):
            years = df_gridded_annual[df_gridded_annual.RGIId == rgi_id].YEAR.unique()
            for year in yearsMaps:
                # TODO: allow to generate maps outside of that range
                assert year in years
                fig = mbm.plots.mapGlacier(
                    df_gridded_annual, rgi_id, year, cfg, gdir=gdir
                )
                fig.savefig(f"{mapsFolder}/{rgi_id}_{year}.pdf")
                plt.close(fig)
    del df_gridded_annual, df_gridded_monthly

else:
    resTest = None


if sourceData == "switzerland":
    train_glaciers = params["training"].get("train_glaciers_geo") or list(
        df_X_train.GLACIER.unique()
    )
    valid_glaciers = params["training"].get("val_glaciers_geo") or list(
        df_X_val.GLACIER.unique()
    )
elif sourceData in ["iceland", "norway"]:
    train_glaciers = params["training"].get("train_glaciers_geo") or list(
        df_X_train.RGIId.unique()
    )
    valid_glaciers = params["training"].get("val_glaciers_geo") or list(
        df_X_val.RGIId.unique()
    )
elif "wgms" in sourceData:
    train_glaciers = params["training"].get("train_glaciers_geo") or list(
        df_X_train.RGIId.unique()
    )
    valid_glaciers = params["training"].get("val_glaciers_geo") or list(
        df_X_val.RGIId.unique()
    )
train_glacierNames = mbm.data_processing.oggm_utils._glacier_name(
    list(data_train.RGIId.unique()), cfg
)
glacierNames = train_glacierNames | test_glacierNames
if len(train_glacierNames) > 0 and len(test_glacierNames) > 0:
    for k in glacierNames:
        if glacierNames[k] == "":
            glacierNames[k] = default_glacier_name(k)
    with open(os.path.join(pathFolder, "glacierNames.json"), "w") as f:
        json.dump(glacierNames, f, indent=4, sort_keys=True)

# Create dataloader
train_gdl = mbm.dataloader.GeoDataLoader(
    cfg,
    train_glaciers,
    device=device,
    trainStakesDf=df_X_train,
    glacierListVal=valid_glaciers,
    months_head_pad=months_head_pad,
    months_tail_pad=months_tail_pad,
    valStakesDf=df_X_val,
    keyGlacierSel="GLACIER" if sourceData == "switzerland" else "RGIId",
    allStakesPerIter=(params["training"]["scalingStakes"] == "full"),
    additionalYears=additionalYears,
    geodeticSource=params["training"].get(
        "geodetic_source", "Hugonnet21"
    ),  # TODO: change
)

with torch.no_grad():
    resVal = mbm.training.assessOnVal(model, train_gdl, params)
    with open(os.path.join(pathFolder, "perf.json"), "w") as f:
        json.dump({"test": resTest, "val": resVal}, f, indent=4)

# PMB predictions
grouped_ids_train = model.evaluate_group_pred(train_gdl)
grouped_ids_valid = model.evaluate_group_pred(train_gdl, val=True)

# PMB train
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
    precLegend=2,
)
fig.savefig(f"{pathFolder}/prediction_train_PMB.pdf")
if plot:
    plt.show()
plt.close(fig)

train_gl_per_el = {
    k: datasetManager.mean_stakes_elevation.get(k, 0.0)
    for k in datasetManager.train_glaciers
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

grouped_ids_train_valid = pd.concat(
    [grouped_ids_train, grouped_ids_valid], ignore_index=True
)
if savePred:
    print("Saving stakes prediction...")
    grouped_ids_train_valid.to_parquet(
        f"{pathFolder}/stakes_train.parquet",
        engine="pyarrow",
        compression="snappy",
    )
fig = mbm.plots.predVSTruthPerGlacier(
    grouped_ids_train_valid,
    scores=scores,
    custom_order=train_gl_per_el,
    hue="PERIOD",
)
fig.savefig(f"{pathFolder}/individual_glaciers_train_PMB.pdf")
if plot:
    plt.show()
plt.close(fig)


# PMB validation
scores_valid = mbm.metrics.seasonal_scores(
    grouped_ids_valid, target_col="target", pred_col="pred"
)
scores_annual = {
    "rmse": scores_valid["annual"]["rmse"],
    "r2": scores_valid["annual"]["r2"],
    "bias": scores_valid["annual"]["bias"],
}
scores_winter = {
    "rmse": scores_valid["winter"]["rmse"],
    "r2": scores_valid["winter"]["r2"],
    "bias": scores_valid["winter"]["bias"],
}
if "summer" in scores_valid:
    scores_summer = {
        "rmse": scores_valid["summer"]["rmse"],
        "r2": scores_valid["summer"]["r2"],
        "bias": scores_valid["summer"]["bias"],
    }
else:
    scores_summer = None

fig = mbm.plots.predVSTruthTimeSeries(
    grouped_ids=grouped_ids_valid,
    scores_annual=scores_annual,
    scores_winter=scores_winter,
    scores_summer=scores_summer,
    ax_xlim=(-14, 8),
    ax_ylim=(-14, 8),
    precLegend=2,
)
fig.savefig(f"{pathFolder}/prediction_validation_PMB.pdf")
if plot:
    plt.show()
plt.close(fig)


pathFolderPred = f"{pathFolder}/pred"
if savePred:
    os.makedirs(pathFolderPred, exist_ok=True)


def callback_save_geodetic_annual(g, df):
    df.to_parquet(
        f"{pathFolderPred}/annual_{g}.parquet",
        engine="pyarrow",
        compression="snappy",
    )


def callback_save_geodetic_monthly(g, df):
    df.to_parquet(
        f"{pathFolderPred}/monthly_{g}.parquet",
        engine="pyarrow",
        compression="snappy",
    )


geoPred, geoTarget, geoErr, dict_df_gridded = mbm.training.eval_geodetic(
    model,
    train_gdl,
    return_grid_pred=["annual", "monthly"],
    callback_annual=(callback_save_geodetic_annual if savePred else None),
    callback_monthly=(callback_save_geodetic_monthly if savePred else None),
)
df_gridded_annual = dict_df_gridded["annual"]
df_gridded_monthly = dict_df_gridded["monthly"]
del dict_df_gridded
if savePred:
    print("Saving gridded prediction...")
    kk = geoTarget.keys()
    df_geo = pd.DataFrame(
        {
            "RGIId": kk,
            "target": [geoTarget[k] for k in kk],
            "err": [geoErr[k] for k in kk],
            "pred": [geoPred[k] for k in kk],
        }
    )
    df_geo.to_parquet(
        f"{pathFolder}/gridded_geodetic_train.parquet",
        engine="pyarrow",
        compression="snappy",
    )
    df_gridded_annual.to_parquet(
        f"{pathFolder}/gridded_annual_train.parquet",
        engine="pyarrow",
        compression="snappy",
    )
    df_gridded_monthly.to_parquet(
        f"{pathFolder}/gridded_monthly_train.parquet",
        engine="pyarrow",
        compression="snappy",
    )


# Geodetic performance
fig = mbm.plots.predVSTruthGlacierWide(
    geoTarget,
    geoPred,
    geoErr,
    title="Glacier wide MB on train",
    ax_xlim=(-2.5, 1.0),
    ax_ylim=(-2.5, 1.0),
    color=color,
    legend=False,
)
plt.savefig(os.path.join(pathFolder, "geodetic_train.png"))
if plot:
    plt.show()
plt.close(fig)


# Plot MB profile
# TODO: ignore years outside of the geodetic time window
fig = mbm.plots.profilePerGlacier(
    df_gridded_annual,
    custom_order=train_gl_per_el,
    titles={
        k: (f"{k} ({glacierNames[k]})" if glacierNames[k] is not None else None)
        for k in glacierNames
    },
    df_stakes=grouped_ids_train,
    average_stakes=False,
)
fig.savefig(f"{pathFolder}/PMB_profile_individual_glaciers_train.pdf")
if plot:
    plt.show()
plt.close(fig)


# Plot cumulated mass change
fig, _ = mbm.plots.cumulatedMassChange(
    df_gridded_monthly,
    geo={
        rgi_id: {
            "mean": geoTarget[rgi_id],
            "err": geoErr[rgi_id],
            "start": train_gdl.geodetic_periods(rgi_id)[0][0],
            "end": train_gdl.geodetic_periods(rgi_id)[0][1],
        }
        for rgi_id in geoTarget
    },
)
fig.savefig(f"{pathFolder}/cumulated_mass_change_glaciers_train.pdf")
if plot:
    plt.show()
plt.close(fig)

if any([m in train_glaciers for m in maps]):
    mapsFolder = f"{pathFolder}/maps"
    os.makedirs(mapsFolder, exist_ok=True)
    # Initialize OGGM once for all to avoid repeated and useless computations
    mbm.data_processing.oggm_utils._initialize_oggm_config("")
    rgi_ids = list(set(train_glaciers).intersection(set(maps)))
    gdirs = mbm.data_processing.oggm_utils._initialize_glacier_directories(rgi_ids, cfg)
    for rgi_id, gdir in zip(rgi_ids, gdirs):
        years = df_gridded_annual[df_gridded_annual.RGIId == rgi_id].YEAR.unique()
        for year in yearsMaps:
            # TODO: allow to generate maps outside of that range
            assert year in years
            fig = mbm.plots.mapGlacier(df_gridded_annual, rgi_id, year, cfg, gdir=gdir)
            fig.savefig(f"{mapsFolder}/{rgi_id}_{year}.pdf")
            plt.close(fig)
del df_gridded_annual, df_gridded_monthly

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
        allStakesPerIter=(params["training"]["scalingStakes"] == "full"),
        geodeticSource=params["training"]["geodetic_source"],
    )

    geoPred, geoTarget, geoErr, _ = mbm.training.eval_geodetic(model, region_gdl)

    # Geodetic performance
    fig = mbm.plots.predVSTruthGlacierWide(
        geoTarget,
        geoPred,
        geoErr,
        title="Glacier wide MB on the whole region",
        legend=False,
    )
    plt.savefig(os.path.join(pathFolder, "geodetic_region.png"))
    if plot:
        plt.show()
    plt.close(fig)
