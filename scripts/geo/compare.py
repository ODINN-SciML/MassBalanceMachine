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
    "--pgo",
    dest="pgo",
    default=False,
    action="store_true",
    help="Evaluate on PGO grid.",
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
    help="Years for which to generate the annual MB maps.",
)
args = parser.parse_args()

modelFolder1 = args.modelFolder1
modelFolder2 = args.modelFolder2
name1 = args.name1
name2 = args.name2
plot = args.plot
noTrain = args.noTrain
pgo = args.pgo
maps = args.maps
yearsMaps = [int(y) for y in args.years]
pathFolder1 = os.path.join("logs", modelFolder1)
pathFolder2 = os.path.join("logs", modelFolder2)
name1 = name1 if name1 is not None else modelFolder1
name2 = name2 if name2 is not None else modelFolder2

if len(maps) > 0:
    assert (
        len(yearsMaps) > 0
    ), "If distributed maps are generated, the option years must be provided."

pathFolder = os.path.join("results/comp/", f"{modelFolder1}_{modelFolder2}")
os.makedirs(pathFolder, exist_ok=True)

with open(f"{pathFolder1}/params.json", "r") as f:
    params1 = json.load(f)
with open(f"{pathFolder2}/params.json", "r") as f:
    params2 = json.load(f)

if os.path.isfile(f"{pathFolder1}/glacierNames.json"):
    with open(f"{pathFolder1}/glacierNames.json", "r") as f:
        glacierNames = json.load(f)
else:
    glacierNames = {}


def load_gridded(file_without_ext):
    if os.path.isfile(file_without_ext + ".parquet"):
        return pd.read_parquet(file_without_ext + ".parquet")
    elif os.path.isfile(file_without_ext + ".csv"):
        return pd.read_csv(file_without_ext + ".csv")
    else:
        raise Exception(f"No file with matching extension found for {file_without_ext}")


linear_fit_breaks = [2015 + 9 / 12]
start_geod_period = 2000
end_geod_period = 2020


if pgo:
    pathFolderPGO = os.path.join(pathFolder, "PGO")
    os.makedirs(pathFolderPGO, exist_ok=True)
    # Cumulated mass change on train data
    df_gridded_monthly1 = load_gridded(f"{pathFolder1}/PGO/gridded_monthly_pgo")
    df_geo1 = load_gridded(f"{pathFolder1}/PGO/gridded_geodetic_pgo")
    geoTarget = df_geo1.set_index("RGIId").target.to_dict()
    geoErr = df_geo1.set_index("RGIId").err.to_dict()

    with open(f"{pathFolder1}/PGO/periodsPerGlacier.json", "r") as f:
        periods_per_glacier = json.load(f)
        for rgi_id in periods_per_glacier.keys():
            periods_per_glacier[rgi_id] = [
                (
                    np.datetime64(periods_per_glacier[rgi_id][0][0]),
                    np.datetime64(periods_per_glacier[rgi_id][0][1]),
                )
            ]

    # Plot cumulated mass change
    fig, l1 = mbm.plots.cumulatedMassChange(
        df_gridded_monthly1,
        geo={
            rgi_id: {
                "mean": geoTarget[rgi_id],
                "err": geoErr[rgi_id],
                "start": periods_per_glacier[rgi_id][0][0],
                "end": periods_per_glacier[rgi_id][0][1],
            }
            for rgi_id in geoTarget
        },
    )
    del df_gridded_monthly1
    df_gridded_monthly2 = load_gridded(f"{pathFolder2}/PGO/gridded_monthly_pgo")
    _, l2 = mbm.plots.cumulatedMassChange(
        df_gridded_monthly2,
        geo={
            rgi_id: {
                "start": periods_per_glacier[rgi_id][0][0],
                "end": periods_per_glacier[rgi_id][0][1],
            }  # Provide the bounds to plot only the cumulated MB of the geodetic time window
            for rgi_id in geoTarget
        },
        axs=fig.axes,
        color_pred="red",
        titles={
            k: (f"{k} ({glacierNames[k]})" if glacierNames[k] is not None else None)
            for k in glacierNames
        },
    )
    del df_gridded_monthly2
    fig.legend([l1, l2], [name1, name2], loc="lower center", ncol=2)

    fig.savefig(f"{pathFolderPGO}/cumulated_mass_change_glaciers_pgo.pdf")
    fig.savefig(f"{pathFolderPGO}/cumulated_mass_change_glaciers_pgo.png", dpi=300)
    if plot:
        plt.show()
    plt.close(fig)


if not noTrain:
    # Cumulated mass change on train data
    df_gridded_monthly1 = load_gridded(f"{pathFolder1}/gridded_monthly_train")
    df_geo1 = load_gridded(f"{pathFolder1}/gridded_geodetic_train")
    geoTarget = df_geo1.set_index("RGIId").target.to_dict()
    geoErr = df_geo1.set_index("RGIId").err.to_dict()

    # Plot cumulated mass change
    fig, l1 = mbm.plots.cumulatedMassChange(
        df_gridded_monthly1,
        geo={
            rgi_id: {
                "mean": geoTarget[rgi_id],
                "err": geoErr[rgi_id],
                "start": start_geod_period,
                "end": end_geod_period,
            }
            for rgi_id in geoTarget
        },
        linear_fit_breaks=linear_fit_breaks,
    )
    del df_gridded_monthly1
    df_gridded_monthly2 = load_gridded(f"{pathFolder2}/gridded_monthly_train")
    _, l2 = mbm.plots.cumulatedMassChange(
        df_gridded_monthly2,
        geo={
            rgi_id: {
                "start": start_geod_period,
                "end": end_geod_period,
            }  # Provide the bounds to plot only the cumulated MB of the geodetic time window
            for rgi_id in geoTarget
        },
        axs=fig.axes,
        color_pred="red",
        titles={
            k: (f"{k} ({glacierNames[k]})" if glacierNames[k] is not None else None)
            for k in glacierNames
        },
        linear_fit_breaks=linear_fit_breaks,
    )
    del df_gridded_monthly2
    fig.legend([l1, l2], [name1, name2], loc="lower center", ncol=2)

    fig.savefig(f"{pathFolder}/cumulated_mass_change_glaciers_train.pdf")
    fig.savefig(f"{pathFolder}/cumulated_mass_change_glaciers_train.png", dpi=300)
    if plot:
        plt.show()
    plt.close(fig)

    # Load annual data
    df_gridded_annual1 = load_gridded(f"{pathFolder1}/gridded_annual_train")
    df_gridded_annual2 = load_gridded(f"{pathFolder2}/gridded_annual_train")

    # Load stakes data
    df_groupeds_train1 = load_gridded(f"{pathFolder1}/stakes_train")

    # Plot MB profile
    fig = mbm.plots.profilePerGlacier(
        df_gridded_annual1[
            (df_gridded_annual1.YEAR >= start_geod_period)
            & (df_gridded_annual1.YEAR < end_geod_period)
        ],
        color="blue",
        titles={
            k: (f"{k} ({glacierNames[k]})" if glacierNames[k] is not None else None)
            for k in glacierNames
        },
        df_stakes=df_groupeds_train1,
        average_stakes=False,
    )
    _ = mbm.plots.profilePerGlacier(
        df_gridded_annual2[
            (df_gridded_annual2.YEAR >= start_geod_period)
            & (df_gridded_annual2.YEAR < end_geod_period)
        ],
        color="red",
        axs=fig.axes,
        titles={
            k: (f"{k} ({glacierNames[k]})" if glacierNames[k] is not None else None)
            for k in glacierNames
        },
    )
    fig.savefig(f"{pathFolder}/MB_profile_individual_glaciers_train.pdf")
    if plot:
        plt.show()
    plt.close(fig)

    if len(maps) > 0:
        train_glaciers = df_gridded_annual1.RGIId.unique()

        mapsFolder = f"{pathFolder}/maps"
        os.makedirs(mapsFolder, exist_ok=True)
        cfg = mbm.Config("11")  # Fake cfg which is needed just for OGGM
        mapsTrain = list(set(train_glaciers).intersection(set(maps)))
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
            for year in yearsMaps:
                # TODO: allow to generate maps outside of that range
                assert year in years
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
df_gridded_monthly1 = load_gridded(f"{pathFolder1}/gridded_monthly_test")
df_geo1 = load_gridded(f"{pathFolder1}/gridded_geodetic_test")
geoTarget = df_geo1.set_index("RGIId").target.to_dict()
geoErr = df_geo1.set_index("RGIId").err.to_dict()

# Plot cumulated mass change
fig, l1 = mbm.plots.cumulatedMassChange(
    df_gridded_monthly1,
    geo={
        rgi_id: {
            "mean": geoTarget[rgi_id],
            "err": geoErr[rgi_id],
            "start": start_geod_period,
            "end": end_geod_period,
        }
        for rgi_id in geoTarget
    },
    linear_fit_breaks=linear_fit_breaks,
)
del df_gridded_monthly1
df_gridded_monthly2 = load_gridded(f"{pathFolder2}/gridded_monthly_test")
_, l2 = mbm.plots.cumulatedMassChange(
    df_gridded_monthly2,
    geo={
        rgi_id: {
            "start": start_geod_period,
            "end": end_geod_period,
        }  # Provide the bounds to plot only the cumulated MB of the geodetic time window
        for rgi_id in geoTarget
    },
    axs=fig.axes,
    color_pred="red",
    titles={
        k: (f"{k} ({glacierNames[k]})" if glacierNames[k] is not None else None)
        for k in glacierNames
    },
    linear_fit_breaks=linear_fit_breaks,
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


# Load annual data
df_gridded_annual1 = load_gridded(f"{pathFolder1}/gridded_annual_test")
df_gridded_annual2 = load_gridded(f"{pathFolder2}/gridded_annual_test")

# Load stakes data
df_groupeds_test1 = load_gridded(f"{pathFolder1}/stakes_test")


# Plot MB profile
fig = mbm.plots.profilePerGlacier(
    df_gridded_annual1[
        (df_gridded_annual1.YEAR >= start_geod_period)
        & (df_gridded_annual1.YEAR < end_geod_period)
    ],
    color="blue",
    titles={
        k: (f"{k} ({glacierNames[k]})" if glacierNames[k] is not None else None)
        for k in glacierNames
    },
    df_stakes=df_groupeds_test1,
    average_stakes=False,
)
_ = mbm.plots.profilePerGlacier(
    df_gridded_annual2[
        (df_gridded_annual2.YEAR >= start_geod_period)
        & (df_gridded_annual2.YEAR < end_geod_period)
    ],
    color="red",
    axs=fig.axes,
    titles={
        k: (f"{k} ({glacierNames[k]})" if glacierNames[k] is not None else None)
        for k in glacierNames
    },
)
fig.savefig(f"{pathFolder}/MB_profile_individual_glaciers_test.pdf")
if plot:
    plt.show()
plt.close(fig)


if len(maps) > 0:
    mapsFolder = f"{pathFolder}/maps"
    os.makedirs(mapsFolder, exist_ok=True)
    test_glaciers = df_gridded_annual1.RGIId.unique()
    cfg = mbm.Config("11")  # Fake cfg which is needed just for OGGM
    mapsTest = list(set(test_glaciers).intersection(set(maps)))
    assert set(mapsTest).issubset(set(df_gridded_annual2.RGIId.unique()))
    for rgi_id in mapsTest:
        years = df_gridded_annual1[df_gridded_annual1.RGIId == rgi_id].YEAR.unique()
        max1 = df_gridded_annual1[df_gridded_annual1.RGIId == rgi_id].pred.abs().max()
        max2 = df_gridded_annual2[df_gridded_annual2.RGIId == rgi_id].pred.abs().max()
        max_abs = max(max1, max2)
        for year in yearsMaps:
            # TODO: allow to generate maps outside of that range
            assert year in years
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
