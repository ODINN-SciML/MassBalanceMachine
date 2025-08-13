import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

import massbalancemachine as mbm
from regions.Switzerland.scripts.glamos_preprocess import getStakesData, get_geodetic_MB
from regions.Switzerland.scripts.geodata import build_periods_per_glacier
from regions.Switzerland.scripts.config_CH import (
    path_PMB_GLAMOS_csv,
    path_ERA5_raw,
    path_pcsr,
)
from regions.Switzerland.scripts.xgb_helpers import (
    process_or_load_data,
    transform_df_to_seasonal,
    get_CV_splits,
)
from regions.Switzerland.scripts.helpers import seed_all


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


def getTrainTestSets(target_train_glaciers, test_glaciers, params, cfg, csvFileName, process=False):

    data_glamos = getStakesData(cfg)
    data_glamos.drop(
        data_glamos[data_glamos.GLACIER == "taelliboden"].index, inplace=True
    )

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
    dataloader_gl = process_or_load_data(
        run_flag=process,
        data_glamos=data_glamos,
        paths=paths,
        cfg=cfg,
        vois_climate=vois_climate,
        vois_topographical=vois_topographical,
        output_file=csvFileName,
    )

    data_monthly = dataloader_gl.data

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

    return train_set, test_set, data_glamos
