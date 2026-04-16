import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

import massbalancemachine as mbm


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
    source_data = params["training"].get("source_data", "iceland")
    inputs = params["model"].get("inputs") or mbm.dataloader._default_input(source_data)
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
            "source_data": source_data,
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
            "wGeo": params["training"].get("wGeo", 0.0),
            "bestModelCriterion": params["training"].get(
                "bestModelCriterion", "lossVal"
            ),
            "freqVal": params["training"].get("freqVal", 1),
            "log_suffix": params["training"].get("log_suffix", ""),
            "log_dir": params["training"].get("log_dir"),
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
