import os
import pytest
import numpy as np
import re
import torch
from torch import nn
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

if "CI" in os.environ:
    pathDataDownload = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../../dataDownload/data/"
        )
    )
    dataPath = pathDataDownload
else:
    dataPath = None


@pytest.mark.order3
def test_swiss_train_geo():
    cfg = mbm.SwitzerlandConfig(
        metaData=[
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
        ],
        notMetaDataNotFeatures=["POINT_BALANCE"],
        dataPath=dataPath,
        seed=30,
    )

    vois_climate = ["t2m", "tp", "slhf", "sshf", "ssrd", "fal", "str", "u10", "v10"]

    vois_topographical = [
        # "aspect", # OGGM
        # "slope", # OGGM
        "aspect_sgi",  # SGI
        "slope_sgi",  # SGI
        "hugonnet_dhdt",  # OGGM
        "consensus_ice_thickness",  # OGGM
        "millan_v",  # OGGM
    ]

    data_glamos = get_stakes_data(cfg)

    # Transform data to monthly format (run or load data):
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
    data_monthly = process_or_load_data(
        run_flag=True,
        data_glamos=data_glamos,
        paths=paths,
        cfg=cfg,
        vois_climate=vois_climate,
        vois_topographical=vois_topographical,
        output_file="CH_wgms_dataset_monthly_all.csv",
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

    # Split on measurements (IDs)
    splits, test_set, train_set = get_CV_splits(
        dataloader_gl, test_split_on="ID", random_state=cfg.seed, test_size=0.1
    )

    data_train = train_set["df_X"]

    feature_columns = list(
        data_train.columns.difference(cfg.metaData).drop(cfg.notMetaDataNotFeatures)
    )
    cfg.setFeatures(feature_columns)

    all_columns = feature_columns + cfg.fieldsNotFeatures
    df_X_train_subset = train_set["df_X"][all_columns]
    print("Shape of training dataset:", df_X_train_subset.shape)
    print("Running with features:", feature_columns)

    assert all(train_set["df_X"].POINT_BALANCE == train_set["y"])

    geodetic_mb = get_geodetic_MB(cfg)

    gdl = mbm.dataloader.GeoDataLoader(
        cfg,
        ["silvretta"],
        train_set["df_X"],
        months_head_pad=months_head_pad,
        months_tail_pad=months_tail_pad,
    )

    nInp = len(feature_columns)
    network = nn.Sequential(
        nn.Linear(nInp, 12),
        nn.ReLU(),
        nn.Linear(12, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
    )
    model = mbm.models.CustomTorchNeuralNetRegressor(network)
    optim = torch.optim.SGD(model.parameters(), lr=1e-4)

    trainCfg = {"Nepochs": 1}
    mbm.training.train_geo(model, gdl, optim, trainCfg)


if __name__ == "__main__":
    test_swiss_train_geo()
