import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from collections.abc import Mapping, Sequence
import math

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
            "dropout": params["model"].get("dropout", 0.0),
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
            "splitVal": params["training"].get("splitVal", "group-meas-id"),
            "freqVal": params["training"].get("freqVal", 1),
            "log_suffix": params["training"].get("log_suffix", ""),
            "log_prefix": params["training"].get("log_prefix", ""),
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


def default_glacier_name(rgi_id):
    return {
        # # Norway
        # RGI60-08.00038;  # Nigardsbreen
        # RGI60-08.00087;  # Jostedalsbreen
        # RGI60-08.00147;  # Folgefonna
        # RGI60-08.00203;  # Hardangerjøkulen
        # Italy
        "RGI60-11.00695": "Glatschiu dil segnas",
        "RGI60-11.03005": "Miage",
        "RGI60-11.03001": "Brenva",
        "RGI60-11.01473": "Laaser Ferner",
        "RGI60-11.00597": "Übeltalferner",
        "RGI60-11.01776": "Langenferner/Vedretta Lunga",
        # France, Mont Blanc
        "RGI60-11.03643": "Mer de Glace/Geant",
        "RGI60-11.03638": "Argentière",
        "RGI60-11.03646": "Bossons",
        "RGI60-11.03647": "Taconnaz",
        "RGI60-11.03296": "Tricot",
        "RGI60-11.03438": "Tete Rousse",
        "RGI60-11.03648": "Bionnassay",
        "RGI60-11.03601": "Armancette",
        "RGI60-11.03650": "Covagnet",
        "RGI60-11.03276": "Miage 1",
        "RGI60-11.03388": "Miage 2",
        "RGI60-11.03579": "Miage 3",
        "RGI60-11.03649": "Miage 4",
        "RGI60-11.03651": "Tré-la-Tête",
        "RGI60-11.03339": "Glaciers",
        # France, Belledonne
        "RGI60-11.03674": "Saint Sorlin",
        # France, Ecrins
        "RGI60-11.03677": "Meije",
        "RGI60-11.03684": "Blanc",
        # France, Pyrénées
        "RGI60-11.03232": "Ossoue",
        "RGI60-11.03208": "Aneto",
        # Austria
        "RGI60-11.00897": "Hintereisferner",
        "RGI60-11.00787": "Kesselwandferner",
        # Switzerland
        "RGI60-11.01270": "Grindelwald",
        "RGI60-11.01450": "Aletsch",
        "RGI60-11.01733": "Hangend",
        "RGI60-11.01328": "Unteraar",
        "RGI60-11.01238": "Rhone",
        "RGI60-11.02249": "Tsanfleuron",
        "RGI60-11.01702": "Kander",
        "RGI60-11.00872": "Hüfi",
        "RGI60-11.02774": "Giétro",
        "RGI60-11.01876": "Gries",
        "RGI60-11.02746": "Schwarzberg",
        "RGI60-11.02810": "Arolla",
        "RGI60-11.02775": "Orny",
        "RGI60-11.02507": "Brunegg",
        "RGI60-11.00804": "Silvretta",
        "RGI60-11.00752": "Vorab",
        "RGI60-11.02787": "Mont Collon",
        "RGI60-11.01267": "Porchabella",
        "RGI60-11.02634": "Prafleuri",
        "RGI60-11.01946": "Morteratsch",
    }.get(rgi_id)


def canonicalize(x):
    """Convert params into a stable, comparable representation."""
    if isinstance(x, Mapping):
        return tuple(sorted((str(k), canonicalize(v)) for k, v in x.items()))
    if isinstance(x, tuple):
        return tuple(canonicalize(v) for v in x)
    if isinstance(x, list):
        return tuple(canonicalize(v) for v in x)
    if isinstance(x, float):
        if math.isnan(x):
            return "__nan__"
        return x
    return x


def already_completed_trial(study, candidate_params: dict):
    from optuna.trial import TrialState

    candidate_key = canonicalize(candidate_params)

    for t in study.get_trials(
        deepcopy=False,
        states=(TrialState.COMPLETE,),
    ):
        if canonicalize(t.params) == candidate_key:
            return True, t

    return False, None
