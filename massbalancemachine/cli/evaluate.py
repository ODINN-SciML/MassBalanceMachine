"""Evaluate a trained model and save the figures.

Command-line entry point `mbm-eval`, also run as `python -m massbalancemachine.cli.evaluate`
or through `scripts/geo/eval.py` from the root of the repository. `modelFolder` is a run
folder under `logs/`, or the path of any folder holding a `params.json` and a checkpoint,
for example one written by `scripts/geo/export.py`.
"""

import os
import warnings
import logging
import json
import argparse

import matplotlib
import torch
import pandas as pd
import numpy as np
import geopandas as gpd

import massbalancemachine as mbm
from massbalancemachine.cli.common import (
    getMetaData,
    setFeatures,
    trainValData,
    testData,
    testSplits,
    rgiRegionOf,
    geodetic_table,
    geodetic_windows,
    default_glacier_name,
    recursive_update,
)


def main(argv=None):
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser("Evaluate a model and save the figures.")
    parser.add_argument(
        "modelFolder",
        type=str,
        help="Folder of the model to load: the name of a run under logs/, or the path of a folder holding params.json and a checkpoint.",
    )
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
    parser.add_argument(
        "--inspect",
        dest="inspect",
        default=[],
        nargs="+",
        help="Glaciers for which to plot the intermediate variables of a TIlike model on the geodetic grid. Figures are saved in <log_dir>/inspect/<RGIId> and the variables in <log_dir>/inspect/<RGIId>_<first year>-<last year>.parquet.",
    )
    parser.add_argument(
        "--inspectOnly",
        dest="inspectOnly",
        default=False,
        action="store_true",
        help="Only run the inspection of the glaciers given with --inspect, without evaluating the model.",
    )
    parser.add_argument(
        "--geodeticTestOnly",
        dest="geodeticTestOnly",
        default=False,
        action="store_true",
        help="Only compute the glacier-wide predictions on the geodetic test set and save them in <log_dir>/gridded_geodetic_test.parquet, without evaluating anything else.",
    )
    parser.add_argument(
        "--dataPath",
        dest="dataPath",
        type=str,
        default=None,
        help="Root of the data tree. Defaults to the .data/ folder of the repository.",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        type=str,
        default=None,
        help="File to overwrite the hyper-parameters that define the model and the training.",
    )
    args = parser.parse_args(argv)

    modelFolder = args.modelFolder
    cpu = args.cpu
    plot = args.plot
    noTest = args.noTest
    onRegion = args.onRegion
    savePred = args.savePred
    pgo = args.pgo
    color = args.color
    maps = args.maps
    yearsMaps = [int(y) for y in args.years]
    inspectGlaciers = args.inspect
    overwrite = args.overwrite
    inspectOnly = args.inspectOnly
    geodeticTestOnly = args.geodeticTestOnly
    if os.path.isdir(modelFolder):
        pathFolder = modelFolder
    else:
        pathFolder = os.path.join("logs", modelFolder)
    if args.dataPath is not None:
        mbm.set_data_path(args.dataPath)

    assert (
        not inspectOnly or len(inspectGlaciers) > 0
    ), "Option inspectOnly requires the glaciers to inspect, given with option inspect."
    assert not (
        inspectOnly and geodeticTestOnly
    ), "Options inspectOnly and geodeticTestOnly cannot be combined."
    assert not (
        geodeticTestOnly and noTest
    ), "Options geodeticTestOnly and noTest cannot be combined."

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

    if overwrite is not None:
        import yaml

        overwriting_params = None
        with open("scripts/netcfg/" + overwrite + ".yml") as stream:
            try:
                overwriting_params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        recursive_update(params, overwriting_params)

    featuresInpModel = params["model"]["inputs"]
    sourceData = params["training"]["source_data"]
    assert (
        len(inspectGlaciers) == 0 or params["model"]["type"] == "TIlike"
    ), "Option inspect is only available for TIlike models."

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
        rgi_region = mbm.dataloader.wgms_rgi_regions(sourceData)
        datasetManager = mbm.dataloader.SourceManagerWGMS(
            cfg,
            params,
            test_split_on=params["training"].get("splitTest", "RGIId"),
            rgi_region=rgi_region,
        )
    train_set, test_set, months_head_pad, months_tail_pad = (
        datasetManager.train_test_sets()
    )

    data_train = train_set["df_X"]
    data_train["y"] = train_set["y"]
    data_test = test_set["df_X"]
    data_test["y"] = test_set["y"]

    setFeatures(cfg, data_train, featuresInpModel)
    split_key = params["training"].get("splitVal", "group-meas-id")
    val_glaciers = params["training"].get("val_glaciers", None)
    val_years = params["training"].get("val_years", None)
    df_X_train, y_train, df_X_val, y_val = trainValData(
        cfg,
        train_set,
        featuresInpModel,
        split_key=split_key,
        val_glaciers=val_glaciers,
        val_years=val_years,
        regions=params["training"].get("regions"),
    )
    df_X_test_subset = testData(cfg, test_set, featuresInpModel)

    geodeticYears = list(range(2000, 2020))
    additionalYears = list(set(yearsMaps).difference(geodeticYears))
    # TODO: change lines above

    # dataset = dataset_val = None  # Initialized hereafter

    # param_init = {"device": "cpu"}  # Use CPU for evaluation

    # Create model
    network = mbm.models.buildModel(cfg, params=params)
    model = mbm.models.CustomTorchNeuralNetRegressor(network)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not cpu else "cpu")
    model = model.to(device)

    # Load model and set to CPU
    bestModelPath, _ = mbm.training.loadBestModel(pathFolder, model)
    print(f"Loaded model {bestModelPath}")

    def inspect_glaciers():
        """Plot and save the intermediate variables of the glaciers of option inspect."""
        # The inspected glaciers get their own dataloaders so that they do not have to be
        # part of the test set on which the geodetic scores are computed. They are
        # inspected on the geodetic test source, which with `regions` is the one of the
        # region holding the glacier, so one dataloader is built per region.
        training = params["training"]
        regions = training.get("regions")
        if regions:
            glacierRegion = dict(zip(inspectGlaciers, rgiRegionOf(inspectGlaciers)))
            sourceBlocks = {}
            for rgi_id in inspectGlaciers:
                names = [
                    name
                    for name, region in regions.items()
                    if glacierRegion[rgi_id] in region["rgi_regions"]
                ]
                assert (
                    len(names) == 1
                ), f"Glacier {rgi_id} provided in option inspect belongs to no region."
                sourceBlocks.setdefault(names[0], []).append(rgi_id)
        else:
            sourceBlocks = {None: list(inspectGlaciers)}
        inspect_gdls = {}
        for name, glaciers in sourceBlocks.items():
            block = regions[name] if regions else training
            gdl = mbm.dataloader.GeoDataLoader(
                cfg,
                glaciers,
                device=device,
                trainStakesDf=None,
                months_head_pad=months_head_pad,
                months_tail_pad=months_tail_pad,
                keyGlacierSel="GLACIER" if sourceData == "switzerland" else "RGIId",
                geodeticSource=block.get("geodetic_source_test")
                or block["geodetic_source"],
                geodeticSourceOptions=block.get("geodetic_source_test_options")
                or block.get("geodetic_source_options"),
            )
            for rgi_id in glaciers:
                assert gdl.hasGeo(
                    rgi_id
                ), f"Glacier {rgi_id} provided in option inspect has no geodetic data."
                inspect_gdls[rgi_id] = gdl

        # Initialize OGGM once for all to avoid repeated and useless computations
        mbm.data_processing.oggm_utils._initialize_oggm_config("")
        gdirs = mbm.data_processing.oggm_utils._initialize_glacier_directories(
            inspectGlaciers, cfg
        )

        def save_inspect_fig(fig, rgi_id, name):
            inspectFolder = os.path.join(pathFolder, "inspect", rgi_id)
            os.makedirs(inspectFolder, exist_ok=True)
            fig.savefig(os.path.join(inspectFolder, f"{name}.png"))
            if plot:
                plt.show()
            plt.close(fig)

        for rgi_id, gdir in zip(inspectGlaciers, gdirs):
            print(f"Inspecting intermediate variables of {rgi_id}")
            df_inter = mbm.training.ti_intermediates_gridded(
                model, inspect_gdls[rgi_id], rgi_id
            )
            period = f"{df_inter.YEAR.min()}-{df_inter.YEAR.max()}"
            inspectFolder = os.path.join(pathFolder, "inspect")
            os.makedirs(inspectFolder, exist_ok=True)
            df_inter.to_parquet(
                os.path.join(inspectFolder, f"{rgi_id}_{period}.parquet"),
                engine="pyarrow",
                compression="snappy",
            )

            fig = mbm.plots.monthlyProfile(
                df_inter,
                "T_downscaled",
                xlabel="Temperature (°C)",
                title=f"{rgi_id}\ndownscaled temperature averaged over {period}",
                lapse_rate_unit="°C/km",
            )
            save_inspect_fig(fig, rgi_id, "temperature_downscaled_profile")

            fig = mbm.plots.monthlyMaps(
                df_inter,
                "T_downscaled",
                rgi_id,
                cfg,
                gdir=gdir,
                title=f"{rgi_id}\ndownscaled temperature averaged over {period}",
                label_cb="Temperature (°C)",
            )
            save_inspect_fig(fig, rgi_id, "temperature_downscaled_maps_monthly")

            fig = mbm.plots.periodMap(
                df_inter,
                "T_downscaled",
                rgi_id,
                cfg,
                gdir=gdir,
                title=f"{rgi_id}\ndownscaled temperature averaged over {period}",
                label_cb="Temperature (°C)",
            )
            save_inspect_fig(fig, rgi_id, "temperature_downscaled_map_period")

            fig = mbm.plots.monthlyProfile(
                df_inter,
                "T_bias",
                xlabel="Temperature bias (°C)",
                title=f"{rgi_id}\ntemperature bias averaged over {period}",
            )
            save_inspect_fig(fig, rgi_id, "temperature_bias_profile")

            fig = mbm.plots.monthlyProfile(
                df_inter,
                "P_scaling",
                xlabel="Precipitation scaling correction (-)",
                title=f"{rgi_id}\nprecipitation scaling correction averaged over {period}",
            )
            save_inspect_fig(fig, rgi_id, "precipitation_scaling_profile")

            if model.module.bias_cor is not None:
                label_pcor = "Precipitation correction $P_{cor}$ (-)"
                fig = mbm.plots.monthlyProfile(
                    df_inter,
                    "P_scaling",
                    xlabel=label_pcor,
                    title=f"{rgi_id}\nprecipitation correction averaged over {period}",
                )
                save_inspect_fig(fig, rgi_id, "precipitation_correction_profile")

                fig = mbm.plots.monthlyMaps(
                    df_inter,
                    "P_scaling",
                    rgi_id,
                    cfg,
                    gdir=gdir,
                    title=f"{rgi_id}\nprecipitation correction averaged over {period}",
                    label_cb=label_pcor,
                )
                save_inspect_fig(fig, rgi_id, "precipitation_correction_maps_monthly")

            fig = mbm.plots.monthlyProfile(
                df_inter,
                "P_corrected",
                xlabel="Precipitation (m w.e. month$^{-1}$)",
                title=f"{rgi_id}\nbias corrected precipitation averaged over {period}",
            )
            save_inspect_fig(fig, rgi_id, "precipitation_corrected_profile")

            fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
            mbm.plots.monthlyProfile(
                df_inter, "cor_acc", ax=axs[0], xlabel="Accumulation factor (-)"
            )
            mbm.plots.monthlyProfile(
                df_inter, "cor_abl", ax=axs[1], xlabel="Ablation factor (-)"
            )
            axs[1].set_ylabel(None)
            fig.suptitle(f"{rgi_id}\nTI factors averaged over {period}")
            fig.tight_layout()
            save_inspect_fig(fig, rgi_id, "accumulation_ablation_factors_profile")

            if model.module.R_sw_contrib is not None:
                label_sw = "Shortwave radiation contribution"
                fig = mbm.plots.monthlyProfile(
                    df_inter,
                    "R_sw",
                    xlabel=f"{label_sw} (m w.e. month$^{{-1}}$)",
                    title=f"{rgi_id}\nshortwave radiation contribution averaged over {period}",
                )
                save_inspect_fig(fig, rgi_id, "shortwave_contribution_profile")

                fig = mbm.plots.monthlyMaps(
                    df_inter,
                    "R_sw",
                    rgi_id,
                    cfg,
                    gdir=gdir,
                    title=f"{rgi_id}\nshortwave radiation contribution averaged over {period}",
                    label_cb=f"{label_sw} (m w.e. month$^{{-1}}$)",
                )
                save_inspect_fig(fig, rgi_id, "shortwave_contribution_maps_monthly")

                # Sum the months of each hydrological year so that the map shows the
                # annual contribution averaged over the years
                df_sw_annual = df_inter.groupby(
                    ["RGIId", "YEAR", "POINT_LAT", "POINT_LON"], as_index=False
                ).R_sw.sum()
                fig = mbm.plots.periodMap(
                    df_sw_annual,
                    "R_sw",
                    rgi_id,
                    cfg,
                    gdir=gdir,
                    title=f"{rgi_id}\nannual shortwave radiation contribution averaged over {period}",
                    label_cb=f"{label_sw} (m w.e. a$^{{-1}}$)",
                )
                save_inspect_fig(fig, rgi_id, "shortwave_contribution_map_annual")
                del df_sw_annual

            del df_inter

        # Temperature bias and precipitation correction over each region, at the
        # centroids of its RGI 6.2 glaciers, with the glaciers of the geodetic products
        # flagged. Without `regions`, the regions are the RGI regions of the inspected
        # glaciers.
        module = model.module
        if regions:
            regionBlocks = {
                name: (region["rgi_regions"], region)
                for name, region in regions.items()
            }
        else:
            regionBlocks = {
                f"RGI{r:02d}": ([r], training)
                for r in sorted(set(rgiRegionOf(inspectGlaciers)))
            }
        regionFolder = os.path.join(pathFolder, "inspect", "regions")
        os.makedirs(regionFolder, exist_ok=True)
        for name, (rgi_regions, block) in regionBlocks.items():
            glaciers = pd.concat(
                [
                    gpd.read_file(
                        mbm.data_processing.glacier_utils.get_region_shape_file(
                            f"{r:02d}"
                        ),
                        ignore_geometry=True,
                        columns=["RGIId", "CenLat", "CenLon", "Zmin", "Zmax"],
                    ).assign(RGI_REGION=r)
                    for r in rgi_regions
                ],
                ignore_index=True,
            )
            # A correction on ELEVATION_DIFFERENCE needs the altitude of the ERA5 cell
            # of each glacier, which each RGI region has its own file for
            correctionInputs = (module.inp_bias_T or []) + (module.inp_P_cor or [])
            if "ELEVATION_DIFFERENCE" in correctionInputs:
                glaciers["ALTITUDE_CLIMATE"] = np.nan
                for r, idx in glaciers.groupby("RGI_REGION").groups.items():
                    glaciers.loc[idx, "ALTITUDE_CLIMATE"] = (
                        mbm.data_processing.get_climate_data.altitude_climate_at(
                            r, glaciers.CenLat[idx], glaciers.CenLon[idx]
                        )
                    )
            # Elevations spanning the glaciers of the region (missing values are < 0)
            zmin = np.percentile(glaciers.Zmin[glaciers.Zmin > 0], 5)
            zmax = np.percentile(glaciers.Zmax[glaciers.Zmax > 0], 95)
            elevations = np.unique(np.round(np.linspace(zmin, zmax, 5), -2))

            # The geodetic glaciers are listed by the ids of their product, and placed
            # with the gridded product they were trained or tested on
            sources = block["geodetic_source"]
            sources = [sources] if isinstance(sources, str) else list(sources)
            testSources = (
                [block["geodetic_source_test"]]
                if block.get("geodetic_source_test")
                else sources
            )
            flagged = {}
            for side, sideSources in [
                ("train", sources),
                ("val", sources),
                ("test", testSources),
            ]:
                ids = block.get(f"{side}_glaciers_geo") or []
                loc = mbm.data_processing.geodetic_glacier_locations(ids, sideSources)
                if len(loc) < len(ids):
                    print(
                        f"{len(ids) - len(loc)} {side} geodetic glaciers of region "
                        f"{name} have no grid in {sideSources} and are not flagged."
                    )
                # Without `regions` the lists hold the glaciers of every RGI region
                inRegion = loc.POINT_LAT.between(
                    glaciers.CenLat.min() - 0.5, glaciers.CenLat.max() + 0.5
                ) & loc.POINT_LON.between(
                    glaciers.CenLon.min() - 0.5, glaciers.CenLon.max() + 0.5
                )
                flagged[side] = loc[inRegion]

            print(f"Mapping the corrections of region {name} at {elevations} m")
            title = f"{name} (RGI {', '.join(str(r) for r in rgi_regions)})"
            for variable, net in [
                ("temperature_bias", module.bias_T),
                ("precipitation_correction", module.bias_cor),
            ]:
                if net is None:
                    continue
                fig = mbm.plots.regionCorrectionMaps(
                    module, glaciers, elevations, variable, flagged=flagged, title=title
                )
                fig.savefig(os.path.join(regionFolder, f"{name}_{variable}.png"))
                if plot:
                    plt.show()
                plt.close(fig)

    if inspectOnly:
        inspect_glaciers()
        return
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
        from massbalancemachine.data_processing.pgo import pgo_target_file

        df = mbm.data_processing.geodetic_target_PGO(pgo_target_file())
        area = df.a_m2 / 1e6
        pgo_glaciers = df.RGIId.values
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
        df_geo = geodetic_table(geoTarget, geoErr, geoPred, pgo_gdl)
        if savePred:
            print("Saving gridded prediction...")
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
                            (np.datetime_as_string(start), np.datetime_as_string(end))
                            for start, end in pgo_gdl.periods_per_glacier[rgi_id]
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
            geo=geodetic_windows(df_geo),
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

    assert (
        not geodeticTestOnly or len(df_X_test_subset) > 0
    ), "The model has no test set."

    def plot_test_PMB(grouped_ids, folder):
        """Scores of the test stake predictions and their figure in `folder`."""
        scores = mbm.metrics.seasonal_scores(
            grouped_ids, target_col="target", pred_col="pred"
        )
        keys = ("rmse", "r2", "bias")
        fig = mbm.plots.predVSTruthTimeSeries(
            grouped_ids=grouped_ids,
            scores_annual=(
                {k: scores["annual"][k] for k in keys} if "annual" in scores else None
            ),
            scores_winter={k: scores["winter"][k] for k in keys},
            scores_summer=(
                {k: scores["summer"][k] for k in keys} if "summer" in scores else None
            ),
            ax_xlim=(-8, 6),
            ax_ylim=(-8, 6),
            precLegend=2,
        )
        fig.savefig(f"{folder}/prediction_test_PMB.pdf")
        if plot:
            plt.show()
        plt.close(fig)
        return scores

    def evaluate_test(testSet, testFolder):
        """Evaluate the model on one test set (see `testSplits`) and write its figures
        and predictions to `testFolder`.

        Returns the scores of `assessOnTest` and the stake predictions, or with
        geodeticTestOnly the glacier-wide geodetic predictions only.
        """
        test_glaciers = testSet["glaciers"]
        df_X_test_subset = testSet["stakes"]
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
            geodeticSource=testSet["geodeticSource"],
            geodeticSourceOptions=testSet["geodeticSourceOptions"],
        )
        if geodeticTestOnly:
            # Glacier-wide predictions only, without the gridded ones
            with torch.no_grad():
                geoPred, geoTarget, geoErr, _ = mbm.training.eval_geodetic(
                    model, test_gdl
                )
            df_geo = geodetic_table(geoTarget, geoErr, geoPred, test_gdl)
            df_geo.to_parquet(
                f"{testFolder}/gridded_geodetic_test.parquet",
                engine="pyarrow",
                compression="snappy",
            )
            print(f"Saved {testFolder}/gridded_geodetic_test.parquet")
            return None, None, df_geo
        grouped_ids = model.evaluate_group_pred(test_gdl)
        plot_test_PMB(grouped_ids, testFolder)

        # submission_df = grouped_ids[["ID", "pred"]].sort_values(by="ID")
        # submission_df.rename(columns={"pred": "POINT_BALANCE"}, inplace=True)
        # # change 'ID' to string
        # submission_df["ID"] = submission_df["ID"].astype(str)
        # # save solution
        # submission_df.to_csv(f"{testFolder}/submission.csv", index=False)

        # solution_df = grouped_ids[["ID", "target"]].sort_values(by="ID")
        # solution_df.rename(columns={"target": "POINT_BALANCE"}, inplace=True)
        # # change 'ID' to string
        # solution_df["ID"] = solution_df["ID"].astype(str)

        # # save solution
        # solution_df.to_csv(f"{testFolder}/solution.csv", index=False)

        test_glaciers_with_stakes = (
            set(datasetManager.test_glaciers)
            .intersection(datasetManager.mean_stakes_elevation.keys())
            .intersection(df_X_test_subset[keyGlacier])
        )
        test_gl_per_el = {
            k: datasetManager.mean_stakes_elevation[k]
            for k in test_glaciers_with_stakes
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
                f"{testFolder}/stakes_test.parquet",
                engine="pyarrow",
                compression="snappy",
            )

        fig = mbm.plots.predVSTruthPerGlacier(
            grouped_ids,
            custom_order=test_gl_per_el,
        )
        fig.savefig(f"{testFolder}/individual_glaciers_test_PMB.pdf")
        if plot:
            plt.show()
        plt.close(fig)

        # Geodetic performance
        with torch.no_grad():
            resTest = mbm.training.assessOnTest(
                testFolder, model, test_gdl, params, color=color
            )

        pathFolderPred = f"{testFolder}/pred"
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
        df_geo = geodetic_table(geoTarget, geoErr, geoPred, test_gdl)
        if savePred:
            print("Saving gridded prediction...")
            df_geo.to_parquet(
                f"{testFolder}/gridded_geodetic_test.parquet",
                engine="pyarrow",
                compression="snappy",
            )
            df_gridded_annual.to_parquet(
                f"{testFolder}/gridded_annual_test.parquet",
                engine="pyarrow",
                compression="snappy",
            )
            df_gridded_monthly.to_parquet(
                f"{testFolder}/gridded_monthly_test.parquet",
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
        fig.savefig(f"{testFolder}/PMB_profile_individual_glaciers_test.pdf")
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
        # fig.savefig(f"{testFolder}/PMB_profile_monthly_individual_glaciers_test.pdf")
        # if plot:
        #     plt.show()
        # plt.close(fig)
        # assert False

        # Plot cumulated mass change
        fig, _ = mbm.plots.cumulatedMassChange(
            df_gridded_monthly,
            geo=geodetic_windows(df_geo),
        )
        fig.savefig(f"{testFolder}/cumulated_mass_change_glaciers_test.pdf")
        if plot:
            plt.show()
        plt.close(fig)

        if any([m in test_glaciers for m in maps]):
            mapsFolder = f"{testFolder}/maps"
            os.makedirs(mapsFolder, exist_ok=True)
            # Initialize OGGM once for all to avoid repeated and useless computations
            mbm.data_processing.oggm_utils._initialize_oggm_config("")
            rgi_ids = list(set(test_glaciers).intersection(set(maps)))
            gdirs = mbm.data_processing.oggm_utils._initialize_glacier_directories(
                rgi_ids, cfg
            )
            for rgi_id, gdir in zip(rgi_ids, gdirs):
                years = df_gridded_annual[
                    df_gridded_annual.RGIId == rgi_id
                ].YEAR.unique()
                for year in yearsMaps:
                    # TODO: allow to generate maps outside of that range
                    assert year in years
                    fig = mbm.plots.mapGlacier(
                        df_gridded_annual, rgi_id, cfg, year=year, gdir=gdir
                    )
                    fig.savefig(f"{mapsFolder}/{rgi_id}_{year}.pdf")
                    plt.close(fig)
        del df_gridded_annual, df_gridded_monthly
        return resTest, grouped_ids, None

    test_glacierNames = {}
    if len(df_X_test_subset) > 0 and not noTest:
        test_glacierNames = mbm.data_processing.oggm_utils._glacier_name(
            list(data_test.RGIId.unique()), cfg
        )
        # One test set per region, each on its own geodetic test source, or a single one
        testSets = testSplits(params, df_X_test_subset, keyGlacier=keyGlacier)
        if list(testSets) == [None]:
            resTest, _, _ = evaluate_test(testSets[None], pathFolder)
            if geodeticTestOnly:
                return
        else:
            # The figures and predictions of every region go to a folder of its own
            resTest, grouped_ids_all, df_geo_all = {}, [], []
            for name, testSet in testSets.items():
                testFolder = os.path.join(pathFolder, name)
                os.makedirs(testFolder, exist_ok=True)
                print(f"Test set of region {name}")
                resTest[name], grouped_ids, df_geo = evaluate_test(testSet, testFolder)
                if geodeticTestOnly:
                    df_geo_all.append(df_geo.assign(region=name))
                else:
                    grouped_ids_all.append(grouped_ids.assign(region=name))
            if geodeticTestOnly:
                # Gathered where the single-region evaluation writes it as well
                pd.concat(df_geo_all, ignore_index=True).to_parquet(
                    f"{pathFolder}/gridded_geodetic_test.parquet",
                    engine="pyarrow",
                    compression="snappy",
                )
                print(f"Saved {pathFolder}/gridded_geodetic_test.parquet")
                return
            # Scores on the test stakes of all regions together
            scores = plot_test_PMB(
                pd.concat(grouped_ids_all, ignore_index=True), pathFolder
            )
            resTest["all_regions_stakes"] = {
                f"{k}_{period}": float(v)
                for period, period_scores in scores.items()
                for k, v in period_scores.items()
            }
    else:
        resTest = None

    if len(inspectGlaciers) > 0:
        inspect_glaciers()

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
        if params["training"]["splitVal"] == "group-rgi":
            train_glaciers = list(set(train_glaciers).difference(valid_glaciers))
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

    # Create dataloaders, one per side: with a split per year the same glacier carries a
    # geodetic window on each side, over different periods, which one dataloader cannot hold
    def buildGeoDataLoader(glacierList, stakesDf, sourceOptions):
        return mbm.dataloader.GeoDataLoader(
            cfg,
            glacierList,
            device=device,
            trainStakesDf=stakesDf,
            months_head_pad=months_head_pad,
            months_tail_pad=months_tail_pad,
            keyGlacierSel="GLACIER" if sourceData == "switzerland" else "RGIId",
            allStakesPerIter=(params["training"]["scalingStakes"] == "full"),
            additionalYears=additionalYears,
            geodeticSource=params["training"]["geodetic_source"],
            geodeticSourceOptions=sourceOptions,
        )

    geodeticSourceOptions = params["training"].get("geodetic_source_options")
    train_gdl = buildGeoDataLoader(train_glaciers, df_X_train, geodeticSourceOptions)
    val_gdl = buildGeoDataLoader(
        valid_glaciers,
        df_X_val,
        params["training"].get("geodetic_source_options_val") or geodeticSourceOptions,
    )

    with torch.no_grad():
        resVal = mbm.training.assessOnVal(model, val_gdl, params, separateLoader=True)
        with open(os.path.join(pathFolder, "perf.json"), "w") as f:
            json.dump({"test": resTest, "val": resVal}, f, indent=4)

    # PMB predictions. The validation dataloader holds the validation stakes as its own
    # stake data, so they are read through its train-side accessor.
    grouped_ids_train = model.evaluate_group_pred(train_gdl)
    grouped_ids_valid = model.evaluate_group_pred(val_gdl)

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

    # Both sides are evaluated, each through its own dataloader, and reported together as
    # before. A glacier of both sides appears twice, its validation entry marked as such.
    geoPred, geoTarget, geoErr, dict_df_gridded = mbm.training.eval_geodetic(
        model,
        train_gdl,
        return_grid_pred=["annual", "monthly"],
        callback_annual=(callback_save_geodetic_annual if savePred else None),
        callback_monthly=(callback_save_geodetic_monthly if savePred else None),
    )
    geoPredVal, geoTargetVal, geoErrVal, dict_df_gridded_val = (
        mbm.training.eval_geodetic(
            model,
            val_gdl,
            return_grid_pred=["annual", "monthly"],
            callback_annual=(callback_save_geodetic_annual if savePred else None),
            callback_monthly=(callback_save_geodetic_monthly if savePred else None),
        )
    )
    df_geo = pd.concat(
        [
            geodetic_table(geoTarget, geoErr, geoPred, train_gdl).assign(side="train"),
            geodetic_table(geoTargetVal, geoErrVal, geoPredVal, val_gdl).assign(
                side="val"
            ),
        ],
        ignore_index=True,
    )

    def _merge_geodetic(train_side, val_side):
        """The two sides in one dictionary, a glacier present in both keeping both entries."""
        merged = dict(train_side)
        for glacier, value in val_side.items():
            merged[f"{glacier} (val)" if glacier in merged else glacier] = value
        return merged

    geoPred = _merge_geodetic(geoPred, geoPredVal)
    geoTarget = _merge_geodetic(geoTarget, geoTargetVal)
    geoErr = _merge_geodetic(geoErr, geoErrVal)
    df_gridded_annual = pd.concat(
        [dict_df_gridded["annual"], dict_df_gridded_val["annual"]], ignore_index=True
    )
    df_gridded_monthly = pd.concat(
        [dict_df_gridded["monthly"], dict_df_gridded_val["monthly"]], ignore_index=True
    )
    del dict_df_gridded, dict_df_gridded_val
    if savePred:
        print("Saving gridded prediction...")
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
        geo=geodetic_windows(df_geo),
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
        gdirs = mbm.data_processing.oggm_utils._initialize_glacier_directories(
            rgi_ids, cfg
        )
        for rgi_id, gdir in zip(rgi_ids, gdirs):
            years = df_gridded_annual[df_gridded_annual.RGIId == rgi_id].YEAR.unique()
            for year in yearsMaps:
                # TODO: allow to generate maps outside of that range
                assert year in years
                fig = mbm.plots.mapGlacier(
                    df_gridded_annual, rgi_id, cfg, year=year, gdir=gdir
                )
                fig.savefig(f"{mapsFolder}/{rgi_id}_{year}.pdf")
                plt.close(fig)
    del df_gridded_annual, df_gridded_monthly

    # TODO: since we changed the iterator, is the evaluation consistent on train/test ?

    if onRegion:
        if params["training"].get("regions"):
            warnings.warn(
                "onRegion evaluates a single RGI region, that of the first training "
                "glacier, and not every region of the model."
            )
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
            geodeticSourceOptions=params["training"].get("geodetic_source_options"),
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


if __name__ == "__main__":
    main()
