import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

import matplotlib.pyplot as plt
from cmcrameri import cm
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

parser = argparse.ArgumentParser("Compare two different models.")
parser.add_argument("modelFolder1", type=str, help="Folder of the 1st model to load.")
parser.add_argument("modelFolder2", type=str, help="Folder of the 2nd model to load.")
parser.add_argument(
    "--name1",
    dest="name1",
    default=None,
    help="Optional name for the 1st model.",
)
parser.add_argument(
    "--name2",
    dest="name2",
    default=None,
    help="Optional name for the 2nd model.",
)
parser.add_argument(
    "--plot",
    dest="plot",
    default=False,
    action="store_true",
    help="Display figures in addition to saving.",
)
parser.add_argument(
    "--noTrain",
    dest="noTrain",
    default=False,
    action="store_true",
    help="Do not compare on train data.",
)
parser.add_argument(
    "--maps",
    dest="maps",
    default=False,
    action="store_true",
    help="Generate annual MB maps for test glaciers.",
)
parser.add_argument(
    "--mapsTrain",
    dest="mapsTrain",
    default=[],
    nargs="+",
    help="Generate annual MB maps for specific train glaciers.",
)
args = parser.parse_args()

modelFolder1 = args.modelFolder1
modelFolder2 = args.modelFolder2
name1 = args.name1
name2 = args.name2
plot = args.plot
noTrain = args.noTrain
maps = args.maps
mapsTrain = args.mapsTrain
pathFolder1 = os.path.join("logs", modelFolder1)
pathFolder2 = os.path.join("logs", modelFolder2)
name1 = name1 if name1 is not None else modelFolder1
name2 = name2 if name2 is not None else modelFolder2

pathFolder = os.path.join("results/comp/", f"{modelFolder1}_{modelFolder2}")
os.makedirs(pathFolder, exist_ok=True)

with open(f"{pathFolder1}/params.json", "r") as f:
    params1 = json.load(f)
with open(f"{pathFolder2}/params.json", "r") as f:
    params2 = json.load(f)

with open(f"{pathFolder1}/glacierNames.json", "r") as f:
    glacierNames = json.load(f)


if not noTrain:
    # Cumulated mass change on train data
    df_gridded_monthly1 = pd.read_csv(f"{pathFolder1}/gridded_monthly_train.csv")
    df_geo1 = pd.read_csv(f"{pathFolder1}/gridded_geodetic_train.csv")
    geoTarget = df_geo1.set_index("RGIId").target.to_dict()
    geoErr = df_geo1.set_index("RGIId").err.to_dict()

    # Plot cumulated mass change
    fig, l1 = mbm.plots.cumulatedMassChange(
        df_gridded_monthly1,
        geo={
            rgi_id: {"mean": geoTarget[rgi_id], "err": geoErr[rgi_id]}
            for rgi_id in geoTarget
        },
    )
    del df_gridded_monthly1
    df_gridded_monthly2 = pd.read_csv(f"{pathFolder2}/gridded_monthly_train.csv")
    _, l2 = mbm.plots.cumulatedMassChange(
        df_gridded_monthly2,
        geo=None,
        axs=fig.axes,
        color_pred="orange",
        titles={
            k: (f"{k} ({glacierNames[k]})" if glacierNames[k] is not None else None)
            for k in glacierNames
        },
    )
    del df_gridded_monthly2
    fig.legend([l1, l2], [name1, name2], loc="lower center", ncol=2)

    fig.savefig(f"{pathFolder}/cumulated_mass_change_glaciers_train.pdf")
    fig.savefig(f"{pathFolder}/cumulated_mass_change_glaciers_train.png", dpi=300)
    if plot:
        plt.show()
    plt.close(fig)

    if len(mapsTrain) > 0:
        df_gridded_annual1 = pd.read_csv(f"{pathFolder1}/gridded_annual_train.csv")
        df_gridded_annual2 = pd.read_csv(f"{pathFolder2}/gridded_annual_train.csv")

        mapsFolder = f"{pathFolder}/maps"
        os.makedirs(mapsFolder, exist_ok=True)
        cfg = mbm.Config("11")  # Fake cfg which is needed just for OGGM
        assert set(mapsTrain).issubset(df_gridded_annual1.RGIId.unique())
        assert set(mapsTrain).issubset(df_gridded_annual2.RGIId.unique())
        for rgi_id in mapsTrain:
            years = df_gridded_annual1[df_gridded_annual1.RGIId == rgi_id].YEAR.unique()
            max1 = (
                df_gridded_annual1[df_gridded_annual1.RGIId == rgi_id].pred.abs().max()
            )
            max2 = (
                df_gridded_annual2[df_gridded_annual2.RGIId == rgi_id].pred.abs().max()
            )
            max_abs = max(max1, max2)
            for year in years:
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                mbm.plots.mapGlacier(
                    df_gridded_annual1,
                    rgi_id,
                    year,
                    cfg,
                    ax=axs[0],
                    max_abs=max_abs,
                    title=name1,
                )
                mbm.plots.mapGlacier(
                    df_gridded_annual2,
                    rgi_id,
                    year,
                    cfg,
                    ax=axs[1],
                    max_abs=max_abs,
                    title=name2,
                )
                fig.suptitle(f"{rgi_id} year {year}")
                plt.tight_layout()
                fig.savefig(f"{mapsFolder}/{rgi_id}_{year}.pdf")
                plt.close(fig)
        del df_gridded_annual1, df_gridded_annual2


# Cumulated mass change on test data
df_gridded_monthly1 = pd.read_csv(f"{pathFolder1}/gridded_monthly_test.csv")
df_geo1 = pd.read_csv(f"{pathFolder1}/gridded_geodetic_test.csv")
geoTarget = df_geo1.set_index("RGIId").target.to_dict()
geoErr = df_geo1.set_index("RGIId").err.to_dict()

# Plot cumulated mass change
fig, l1 = mbm.plots.cumulatedMassChange(
    df_gridded_monthly1,
    geo={
        rgi_id: {"mean": geoTarget[rgi_id], "err": geoErr[rgi_id]}
        for rgi_id in geoTarget
    },
)
del df_gridded_monthly1
df_gridded_monthly2 = pd.read_csv(f"{pathFolder2}/gridded_monthly_test.csv")
_, l2 = mbm.plots.cumulatedMassChange(
    df_gridded_monthly2,
    geo=None,
    axs=fig.axes,
    color_pred="orange",
    titles={
        k: (f"{k} ({glacierNames[k]})" if glacierNames[k] is not None else None)
        for k in glacierNames
    },
)
del df_gridded_monthly2
fig.legend(
    [l1, l2],
    [name1, name2],
    loc="lower center",
    ncol=2,
    fontsize=18,
    bbox_to_anchor=(0.5, 0.02),
)
plt.tight_layout(rect=[0, 0.1, 1, 1])


fig.savefig(f"{pathFolder}/cumulated_mass_change_glaciers_test.pdf")
fig.savefig(f"{pathFolder}/cumulated_mass_change_glaciers_test.png", dpi=300)
if plot:
    plt.show()
plt.close(fig)

if maps:
    df_gridded_annual1 = pd.read_csv(f"{pathFolder1}/gridded_annual_test.csv")
    df_gridded_annual2 = pd.read_csv(f"{pathFolder2}/gridded_annual_test.csv")

    mapsFolder = f"{pathFolder}/maps"
    os.makedirs(mapsFolder, exist_ok=True)
    rgi_ids = df_gridded_annual1.RGIId.unique()
    cfg = mbm.Config("11")  # Fake cfg which is needed just for OGGM
    assert set(rgi_ids) == set(df_gridded_annual2.RGIId.unique())
    for rgi_id in rgi_ids:
        years = df_gridded_annual1[df_gridded_annual1.RGIId == rgi_id].YEAR.unique()
        max1 = df_gridded_annual1[df_gridded_annual1.RGIId == rgi_id].pred.abs().max()
        max2 = df_gridded_annual2[df_gridded_annual2.RGIId == rgi_id].pred.abs().max()
        max_abs = max(max1, max2)
        for year in years:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            mbm.plots.mapGlacier(
                df_gridded_annual1,
                rgi_id,
                year,
                cfg,
                ax=axs[0],
                max_abs=max_abs,
                title=name1,
            )
            mbm.plots.mapGlacier(
                df_gridded_annual2,
                rgi_id,
                year,
                cfg,
                ax=axs[1],
                max_abs=max_abs,
                title=name2,
            )
            fig.suptitle(f"{rgi_id} year {year}")
            plt.tight_layout()
            fig.savefig(f"{mapsFolder}/{rgi_id}_{year}.pdf")
            plt.close(fig)
    del df_gridded_annual1, df_gridded_annual2
