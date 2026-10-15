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
from scripts.common import geodetic_windows

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
    "--profiles",
    dest="profiles",
    default=[],
    nargs="+",
    help="Generate MB profile plots for specific glaciers.",
)
parser.add_argument(
    "--years",
    dest="years",
    default=[],
    nargs="+",
    help="Years for which to generate the annual MB maps.",
)
parser.add_argument(
    "--skip",
    dest="skip",
    default=False,
    action="store_true",
    help="Skip plots that show all the glaciers. This option allows generating glacier specific comparison quickly.",
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
profiles = args.profiles
yearsMaps = [int(y) for y in args.years]
skip = args.skip
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

    if "start" not in df_geo1.columns:
        # Table saved when a glacier could only have one geodetic window, whose bounds
        # were only written to periodsPerGlacier.json
        with open(f"{pathFolder1}/PGO/periodsPerGlacier.json", "r") as f:
            periods_per_glacier = json.load(f)
        df_geo1["start"] = df_geo1.RGIId.map(
            lambda rgi_id: np.datetime64(periods_per_glacier[rgi_id][0][0])
        )
        df_geo1["end"] = df_geo1.RGIId.map(
            lambda rgi_id: np.datetime64(periods_per_glacier[rgi_id][0][1])
        )

    # Plot cumulated mass change
    fig, l1 = mbm.plots.cumulatedMassChange(
        df_gridded_monthly1,
        geo=geodetic_windows(df_geo1),
    )
    del df_gridded_monthly1
    df_gridded_monthly2 = load_gridded(f"{pathFolder2}/PGO/gridded_monthly_pgo")
    _, l2 = mbm.plots.cumulatedMassChange(
        df_gridded_monthly2,
        # Provide the bounds to plot only the cumulated MB of the geodetic time windows
        geo=geodetic_windows(df_geo1, with_target=False),
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
    if not skip:
        # Cumulated mass change on train data
        df_gridded_monthly1 = load_gridded(f"{pathFolder1}/gridded_monthly_train")
        df_geo1 = load_gridded(f"{pathFolder1}/gridded_geodetic_train")
        default_period = (start_geod_period, end_geod_period)

        # Plot cumulated mass change
        fig, l1 = mbm.plots.cumulatedMassChange(
            df_gridded_monthly1,
            geo=geodetic_windows(df_geo1, default_period=default_period),
            linear_fit_breaks=linear_fit_breaks,
        )
        del df_gridded_monthly1
        df_gridded_monthly2 = load_gridded(f"{pathFolder2}/gridded_monthly_train")
        _, l2 = mbm.plots.cumulatedMassChange(
            df_gridded_monthly2,
            # Provide the bounds to plot only the cumulated MB of the geodetic time windows
            geo=geodetic_windows(
                df_geo1, with_target=False, default_period=default_period
            ),
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

    if not skip:
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

    if len(profiles) > 0:
        profilesFolder = f"{pathFolder}/profiles"
        os.makedirs(profilesFolder, exist_ok=True)
        train_glaciers = df_gridded_annual1.RGIId.unique()
        profilesTrain = [rgi_id for rgi_id in profiles if rgi_id in train_glaciers]
        assert set(profilesTrain).issubset(set(df_gridded_annual2.RGIId.unique()))
        for rgi_id in profilesTrain:
            profile_fig = mbm.plots.profilePerGlacier(
                df_gridded_annual1[
                    (df_gridded_annual1.YEAR >= start_geod_period)
                    & (df_gridded_annual1.YEAR < end_geod_period)
                ],
                color="blue",
                custom_order=[rgi_id],
                titles={
                    k: (
                        f"{k} ({glacierNames[k]})"
                        if glacierNames[k] is not None
                        else None
                    )
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
                custom_order=[rgi_id],
                axs=profile_fig.axes,
                titles={
                    k: (
                        f"{k} ({glacierNames[k]})"
                        if glacierNames[k] is not None
                        else None
                    )
                    for k in glacierNames
                },
            )
            profile_fig.savefig(f"{profilesFolder}/{rgi_id}.pdf")
            plt.close(profile_fig)

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
                    cfg,
                    year=year,
                    ax=axs[0],
                    max_abs=max_abs,
                    title=name1,
                )
                mbm.plots.mapGlacier(
                    df_gridded_annual2,
                    rgi_id,
                    cfg,
                    year=year,
                    ax=axs[1],
                    max_abs=max_abs,
                    title=name2,
                )
                fig.suptitle(f"{rgi_id} year {year}")
                plt.tight_layout()
                fig.savefig(f"{mapsFolder}/{rgi_id}_{year}.pdf")
                plt.close(fig)
    del df_gridded_annual1, df_gridded_annual2


if not skip:
    # Cumulated mass change on test data
    df_gridded_monthly1 = load_gridded(f"{pathFolder1}/gridded_monthly_test")
    df_geo1 = load_gridded(f"{pathFolder1}/gridded_geodetic_test")
    default_period = (start_geod_period, end_geod_period)

    # Plot cumulated mass change
    fig, l1 = mbm.plots.cumulatedMassChange(
        df_gridded_monthly1,
        geo=geodetic_windows(df_geo1, default_period=default_period),
        linear_fit_breaks=linear_fit_breaks,
    )
    del df_gridded_monthly1
    df_gridded_monthly2 = load_gridded(f"{pathFolder2}/gridded_monthly_test")
    _, l2 = mbm.plots.cumulatedMassChange(
        df_gridded_monthly2,
        # Provide the bounds to plot only the cumulated MB of the geodetic time windows
        geo=geodetic_windows(df_geo1, with_target=False, default_period=default_period),
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


if not skip:
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


if len(profiles) > 0:
    profilesFolder = f"{pathFolder}/profiles"
    os.makedirs(profilesFolder, exist_ok=True)
    test_glaciers = df_gridded_annual1.RGIId.unique()
    profilesTest = [rgi_id for rgi_id in profiles if rgi_id in test_glaciers]
    assert set(profilesTest).issubset(set(df_gridded_annual2.RGIId.unique()))
    for rgi_id in profilesTest:
        profile_fig = mbm.plots.profilePerGlacier(
            df_gridded_annual1[
                (df_gridded_annual1.YEAR >= start_geod_period)
                & (df_gridded_annual1.YEAR < end_geod_period)
            ],
            color="blue",
            custom_order=[rgi_id],
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
            custom_order=[rgi_id],
            axs=profile_fig.axes,
            titles={
                k: (f"{k} ({glacierNames[k]})" if glacierNames[k] is not None else None)
                for k in glacierNames
            },
        )
        profile_fig.savefig(f"{profilesFolder}/{rgi_id}.pdf")
        plt.close(profile_fig)


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
                cfg,
                year=year,
                ax=axs[0],
                max_abs=max_abs,
                title=name1,
            )
            mbm.plots.mapGlacier(
                df_gridded_annual2,
                rgi_id,
                cfg,
                year=year,
                ax=axs[1],
                max_abs=max_abs,
                title=name2,
            )
            fig.suptitle(f"{rgi_id} year {year}")
            plt.tight_layout()
            fig.savefig(f"{mapsFolder}/{rgi_id}_{year}.pdf")
            plt.close(fig)
del df_gridded_annual1, df_gridded_annual2
