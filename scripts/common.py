import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

import massbalancemachine as mbm
from regions.Switzerland.scripts.geodetic.geodetic_processing import get_geodetic_MB
from regions.Switzerland.scripts.config_CH import (
    path_PMB_GLAMOS_csv,
    path_ERA5_raw,
    path_pcsr,
)

from regions.Switzerland.scripts.dataset.data_loader import (
    process_or_load_data,
    get_CV_splits,
    get_stakes_data,
)
from regions.Switzerland.scripts.utils import seed_all

_default_test_glaciers = [
    "tortin",
    "plattalva",
    "sanktanna",
    "schwarzberg",
    "hohlaub",
    "pizol",
    "corvatsch",
    "tsanfleuron",
    "forno",
]
_default_train_glaciers = [
    "clariden",
    "oberaar",
    "otemma",
    "gietro",
    "rhone",
    "silvretta",
    "gries",
    "sexrouge",
    "allalin",
    "corbassiere",
    "aletsch",
    "joeri",
    "basodino",
    "morteratsch",
    "findelen",
    "albigna",
    "gorner",
    "murtel",
    "plainemorte",
    "adler",
    "limmern",
    "schwarzbach",
]

_default_additional_var = [
    "ALTITUDE_CLIMATE",
    "ELEVATION_DIFFERENCE",
    "POINT_ELEVATION",
    "pcsr",
]
_default_vois_climate = [
    "t2m",
    "tp",
    "slhf",
    "sshf",
    "ssrd",
    "fal",
    "str",
    "u10",
    "v10",
]
_default_vois_topographical = [
    "aspect_sgi",
    "slope_sgi",
    "hugonnet_dhdt",
    "consensus_ice_thickness",
    "millan_v",
]
_default_input = (
    _default_additional_var + _default_vois_climate + _default_vois_topographical
)


def parseParams(params):
    lr = float(params["training"].get("lr", 1e-3))
    optim = params["training"].get("optim", "ADAM")
    momentum = float(params["training"].get("momentum", 0.0))
    beta1 = float(params["training"].get("beta1", 0.9))
    beta2 = float(params["training"].get("beta2", 0.999))
    scheduler = params["training"].get("scheduler", None)
    scheduler_gamma = float(params["training"].get("scheduler_gamma", 0.5))
    scheduler_step_size = int(params["training"].get("scheduler_step_size", 200))
    Nepochs = int(params["training"].get("Nepochs", 1000))
    inputs = params["model"].get("inputs") or _default_input
    batch_size = int(params["training"].get("batch_size", 128))
    weight_decay = float(params["training"].get("weight_decay", 0.0))
    downscale = params["model"].get("downscale", None)
    scalingStakes = params["training"].get("scalingStakes", "glacier")
    return {
        "model": {
            "type": params["model"]["type"],
            "layers": params["model"]["layers"],
            "inputs": inputs,
            "downscale": downscale,
        },
        "training": {
            "lr": lr,
            "momentum": momentum,
            "beta1": beta1,
            "beta2": beta2,
            "optim": optim,
            "scheduler": scheduler,
            "scheduler_gamma": scheduler_gamma,
            "scheduler_step_size": scheduler_step_size,
            "Nepochs": Nepochs,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "scalingStakes": scalingStakes,
        },
    }


def loadParams(modelType):
    with open("scripts/netcfg/" + modelType + ".yml") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    parsedParams = parseParams(params)
    return parsedParams


def trainTestGlaciers(params):
    train_glaciers = params["training"].get("train_glaciers") or _default_train_glaciers
    test_glaciers = params["training"].get("test_glaciers") or _default_test_glaciers
    return train_glaciers, test_glaciers


def get_CV_splits_iceland(
    dataloader, test_split_on="YEAR", test_splits=None, random_state=0, test_size=0.2
):
    # Split into training and test splits with train_test_split
    if test_splits is None:
        train_splits, test_splits = train_test_split(
            dataloader.data[test_split_on].unique(),
            test_size=test_size,
            random_state=random_state,
        )
    else:
        split_data = dataloader.data[test_split_on].unique()
        train_splits = [x for x in split_data if x not in test_splits]

    train_indices = dataloader.data[
        dataloader.data[test_split_on].isin(train_splits)
    ].index
    test_indices = dataloader.data[
        dataloader.data[test_split_on].isin(test_splits)
    ].index

    dataloader.set_custom_train_test_indices(train_indices, test_indices)

    # Get the features and targets of the training data for the indices as defined above, that will be used during the cross validation.
    df_X_train = dataloader.data.iloc[train_indices]
    y_train = df_X_train["POINT_BALANCE"].values
    train_meas_id = df_X_train["ID"].unique()

    # Get test set
    df_X_test = dataloader.data.iloc[test_indices]
    y_test = df_X_test["POINT_BALANCE"].values
    test_meas_id = df_X_test["ID"].unique()

    # Values split in training and test set
    train_splits = df_X_train[test_split_on].unique()
    test_splits = df_X_test[test_split_on].unique()

    test_set = {
        "df_X": df_X_test,
        "y": y_test,
        "meas_id": test_meas_id,
        "splits_vals": test_splits,
    }
    train_set = {
        "df_X": df_X_train,
        "y": y_train,
        "splits_vals": train_splits,
        "meas_id": train_meas_id,
    }
    return test_set, train_set


def getTrainTestSetsIceland(test_glaciers, params, cfg):
    # TODO: for the moment the arguments are the RGIId but we should manage this properly in the future
    data = pd.read_csv(
        os.path.join(
            mbm_path, "notebooks/example_data/iceland/files/iceland_monthly_dataset.csv"
        )
    )
    existing_glaciers = set(data.RGIId.unique())
    missing_glaciers = [g for g in test_glaciers if g not in existing_glaciers]
    if missing_glaciers:
        print(
            f"Warning: The following test glaciers are not in the dataset: {missing_glaciers}"
        )

    train_glaciers = [i for i in existing_glaciers if i not in test_glaciers]

    dataloader = mbm.dataloader.DataLoader(
        cfg, data=data, random_seed=cfg.seed, meta_data_columns=cfg.metaData
    )
    data_monthly = dataloader.data
    months_head_pad, months_tail_pad = (
        mbm.data_processing.utils.build_head_tail_pads_from_monthly_df(data_monthly)
    )

    data_test = dataloader.data[dataloader.data.RGIId.isin(test_glaciers)]
    print("Size of monthly test data:", len(data_test))

    data_train = dataloader.data[dataloader.data.RGIId.isin(train_glaciers)]
    print("Size of monthly train data:", len(data_train))

    if len(data_train) == 0:
        print("Warning: No training data available!")
    else:
        test_perc = (len(data_test) / len(data_train)) * 100
        print("Percentage of test size: {:.2f}%".format(test_perc))

    # Split on measurements (IDs)
    test_set, train_set = get_CV_splits_iceland(
        dataloader,
        test_split_on="RGIId",
        test_splits=test_glaciers,
        random_state=cfg.seed,
    )
    return train_set, test_set, months_head_pad, months_tail_pad


def getTrainTestSetsSwitzerland(
    target_train_glaciers, test_glaciers, params, cfg, csvFileName, process=False
):

    data_glamos = get_stakes_data(cfg)
    data_glamos.drop(
        data_glamos[data_glamos.GLACIER == "taelliboden"].index, inplace=True
    )
    downscale = params["model"]["downscale"]
    if downscale is not None:
        assert False, "The downscale option is not supported yet."
        # assert downscale == "linear"

    vois_climate = params["model"].get("vois_climate", _default_vois_climate)
    vois_topographical = params["model"].get(
        "vois_topographical", _default_vois_topographical
    )
    paths = {
        "csv_path": cfg.dataPath + path_PMB_GLAMOS_csv,
        "era5_climate_data": cfg.dataPath
        + path_ERA5_raw
        + "era5_monthly_averaged_data.nc",
        "geopotential_data": cfg.dataPath
        + path_ERA5_raw
        + "era5_geopotential_pressure.nc",
        "radiation_save_path": cfg.dataPath + path_pcsr + "zarr/",
    }

    # Transform data to monthly format (run or load data)
    data_monthly = process_or_load_data(
        run_flag=process,
        data_glamos=data_glamos,
        paths=paths,
        cfg=cfg,
        vois_climate=vois_climate,
        vois_topographical=vois_topographical,
        output_file=csvFileName,
    )

    months_head_pad, months_tail_pad = (
        mbm.data_processing.utils.build_head_tail_pads_from_monthly_df(data_monthly)
    )

    data_monthly["GLWD_ID"] = data_monthly.apply(
        lambda x: mbm.data_processing.utils.get_hash(f"{x.GLACIER}_{x.YEAR}"), axis=1
    )
    data_monthly["GLWD_ID"] = data_monthly["GLWD_ID"].astype(str)

    # data_seas = transform_df_to_seasonal(data_monthly)
    # print('Number of seasonal rows', len(data_seas))

    dataloader_gl = mbm.dataloader.DataLoader(
        cfg, data=data_monthly, random_seed=cfg.seed, meta_data_columns=cfg.metaData
    )

    print(dataloader_gl.data.keys())

    print("len(target_train_glaciers) =", len(target_train_glaciers))

    # Ensure all test glaciers exist in the dataset
    existing_glaciers = set(dataloader_gl.data.GLACIER.unique())
    missing_glaciers = [g for g in test_glaciers if g not in existing_glaciers]

    if missing_glaciers:
        print(
            f"Warning: The following test glaciers are not in the dataset: {missing_glaciers}"
        )

    # Define training glaciers correctly
    train_glaciers = [i for i in existing_glaciers if i not in test_glaciers]

    data_test = dataloader_gl.data[dataloader_gl.data.GLACIER.isin(test_glaciers)]
    print("Size of monthly test data:", len(data_test))

    data_train = dataloader_gl.data[dataloader_gl.data.GLACIER.isin(train_glaciers)]
    print("Size of monthly train data:", len(data_train))

    if len(data_train) == 0:
        print("Warning: No training data available!")
    else:
        test_perc = (len(data_test) / len(data_train)) * 100
        print("Percentage of test size: {:.2f}%".format(test_perc))

    # Number of annual versus winter measurements:
    print("-------------\nTrain:")
    print("Number of monthly winter and annual samples:", len(data_train))
    print(
        "Number of monthly annual samples:",
        len(data_train[data_train.PERIOD == "annual"]),
    )
    print(
        "Number of monthly winter samples:",
        len(data_train[data_train.PERIOD == "winter"]),
    )

    # Same for test
    data_test_annual = data_test[data_test.PERIOD == "annual"]
    data_test_winter = data_test[data_test.PERIOD == "winter"]

    print("Test:")
    print("Number of monthly winter and annual samples:", len(data_test))
    print("Number of monthly annual samples:", len(data_test_annual))
    print("Number of monthly winter samples:", len(data_test_winter))

    print("Total:")
    print("Number of monthly rows:", len(dataloader_gl.data))
    print(
        "Number of annual rows:",
        len(dataloader_gl.data[dataloader_gl.data.PERIOD == "annual"]),
    )
    print(
        "Number of winter rows:",
        len(dataloader_gl.data[dataloader_gl.data.PERIOD == "winter"]),
    )

    # same for original data:
    print("-------------\nIn annual format:")
    print(
        "Number of annual train rows:",
        len(data_glamos[data_glamos.GLACIER.isin(train_glaciers)]),
    )
    print(
        "Number of annual test rows:",
        len(data_glamos[data_glamos.GLACIER.isin(test_glaciers)]),
    )

    print("len(train_glaciers) =", len(train_glaciers))
    assert set(train_glaciers) == set(target_train_glaciers)

    # Split on measurements (IDs)
    splits, test_set, train_set = get_CV_splits(
        dataloader_gl,
        test_split_on="GLACIER",
        test_splits=test_glaciers,
        random_state=cfg.seed,
    )

    print(
        "Test glaciers: ({}) {}".format(
            len(test_set["splits_vals"]), test_set["splits_vals"]
        )
    )
    test_perc = (len(test_set["df_X"]) / len(train_set["df_X"])) * 100
    print("Percentage of test size: {:.2f}%".format(test_perc))
    print("Size of test set:", len(test_set["df_X"]))
    print(
        "Train glaciers: ({}) {}".format(
            len(train_set["splits_vals"]), train_set["splits_vals"]
        )
    )
    print("Size of train set:", len(train_set["df_X"]))

    return train_set, test_set, data_glamos, months_head_pad, months_tail_pad
