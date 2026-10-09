import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from collections.abc import Mapping, Sequence
import math

import massbalancemachine as mbm
from massbalancemachine.cli.common import (
    geodetic_table,
    geodetic_windows,
    default_glacier_name,
    mergeRegions,
)

NETCFG_FOLDER = "scripts/netcfg"


def parseParams(params):
    if "regions" in params["training"]:
        params = {
            **params,
            "training": mergeRegions(params["training"], NETCFG_FOLDER),
        }
    lr = float(params["training"].get("lr", 1e-3))
    optim = params["training"].get("optim", "ADAM")
    momentum = float(params["training"].get("momentum", 0.0))
    beta1 = float(params["training"].get("beta1", 0.9))
    beta2 = float(params["training"].get("beta2", 0.999))
    scheduler = params["training"].get("scheduler", None)
    scheduler_gamma = float(params["training"].get("scheduler_gamma", 0.5))
    scheduler_step_size = int(params["training"].get("scheduler_step_size", 200))
    Nepochs = int(params["training"].get("Nepochs", 1000))
    source_data = params["training"].get("source_data", "iceland")
    geodetic_source = params["training"].get("geodetic_source", "Hugonnet21")
    geodetic_source_test = params["training"].get("geodetic_source_test")
    geodetic_source_options = params["training"].get("geodetic_source_options", {})
    geodetic_source_options_val = params["training"].get(
        "geodetic_source_options_val", {}
    )
    geodetic_source_test_options = params["training"].get(
        "geodetic_source_test_options", {}
    )
    inputs = params["model"].get("inputs") or mbm.dataloader._default_input(source_data)
    batch_size = int(params["training"].get("batch_size", 128))
    weight_decay = float(params["training"].get("weight_decay", 0.0))
    downscale = params["model"].get("downscale", None)
    scalingStakes = params["training"].get("scalingStakes", "glacier")
    modelParams = {
        "type": params["model"]["type"],
        "inputs": inputs,
    }
    if (
        modelParams["type"] == "sequential"
        or modelParams["type"] == "sequential_downscaled"
    ):
        modelParams["layers"] = params["model"]["layers"]
        modelParams["dropout"] = params["model"].get("dropout", 0.0)
        modelParams["downscale"] = downscale
    elif modelParams["type"] == "TIlike":
        if "cor_T" in params["model"]:
            modelParams["cor_T"] = params["model"]["cor_T"]
        elif "grad_T" in params["model"] and "bias_T" in params["model"]:
            modelParams["grad_T"] = params["model"]["grad_T"]
            modelParams["bias_T"] = {
                **params["model"]["bias_T"],
                "b_max": float(params["model"]["bias_T"].get("b_max", 5.0)),
            }
            if "cor_dir" in params["model"] and "cor_terrain" in params["model"]:
                modelParams["cor_dir"] = params["model"]["cor_dir"]
                modelParams["cor_terrain"] = params["model"]["cor_terrain"]
            if "sw_contrib" in params["model"]:
                modelParams["sw_contrib"] = params["model"]["sw_contrib"]
        else:
            ValueError("Cannot identify the type of temperature downscaling")
        if "cor_fac" in params["model"]:
            modelParams["cor_fac"] = {"layers": params["model"]["cor_fac"]["layers"]}
        else:
            modelParams["cor_acc"] = params["model"]["cor_acc"]
            modelParams["cor_abl"] = params["model"]["cor_abl"]
        if "bias_cor" in params["model"]:
            modelParams["bias_cor"] = {
                **params["model"]["bias_cor"],
                "p_min": float(params["model"]["bias_cor"].get("p_min", 0.0)),
                "p_max": float(params["model"]["bias_cor"].get("p_max", 2.0)),
            }
        if "bias_cor_elev" in params["model"]:
            modelParams["bias_cor_elev"] = params["model"]["bias_cor_elev"]
        if "snow_slope" in params["model"]:
            modelParams["snow_slope"] = params["model"]["snow_slope"]
        if "trainable" in params["model"]:
            modelParams["trainable"] = params["model"]["trainable"]
    trainingParams = {
        "source_data": source_data,
        "geodetic_source": geodetic_source,
        "geodetic_source_options": geodetic_source_options,
        "geodetic_source_options_val": geodetic_source_options_val,
        "geodetic_source_test": geodetic_source_test,
        "geodetic_source_test_options": geodetic_source_test_options,
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
        "test_glaciers": params["training"].get("test_glaciers"),
        "train_glaciers": params["training"].get("train_glaciers"),
        "val_glaciers": params["training"].get("val_glaciers"),
        "wGeo": params["training"].get("wGeo", 0.0),
        "scalingGeo": params["training"].get("scalingGeo", "quad"),
        "bestModelCriterion": params["training"].get("bestModelCriterion", "lossVal"),
        "splitVal": params["training"].get("splitVal", "group-meas-id"),
        "splitTest": params["training"].get("splitTest", "group-rgi"),
        "freqVal": params["training"].get("freqVal", 1),
        "log_suffix": params["training"].get("log_suffix", ""),
        "log_prefix": params["training"].get("log_prefix", ""),
        "log_dir": params["training"].get("log_dir"),
        "wWinter": params["training"].get("wWinter", 1.0),
        "wSummer": params["training"].get("wSummer", 1.0),
    }
    if "regions" in params["training"]:
        trainingParams["regions"] = params["training"]["regions"]
    if "val_years" in params["training"]:
        trainingParams["val_years"] = params["training"]["val_years"]
    if "test_glaciers_geo" in params["training"]:
        trainingParams["test_glaciers_geo"] = params["training"].get(
            "test_glaciers_geo"
        )
    if "train_glaciers_geo" in params["training"]:
        trainingParams["train_glaciers_geo"] = params["training"].get(
            "train_glaciers_geo"
        )
    if "val_glaciers_geo" in params["training"]:
        trainingParams["val_glaciers_geo"] = params["training"].get("val_glaciers_geo")
    return {
        "model": modelParams,
        "training": trainingParams,
    }


def loadParams(modelType):
    with open(os.path.join(NETCFG_FOLDER, modelType + ".yml")) as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    parsedParams = parseParams(params)
    return parsedParams
