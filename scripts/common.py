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
    geodetic_source = params["training"].get("geodetic_source", "Hugonnet21")
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
            modelParams["bias_T"] = params["model"]["bias_T"]
        else:
            ValueError("Cannot identify the type of temperature downscaling")
        if "cor_fac" in params["model"]:
            modelParams["cor_fac"] = {"layers": params["model"]["cor_fac"]["layers"]}
        else:
            modelParams["cor_acc"] = params["model"]["cor_acc"]
            modelParams["cor_abl"] = params["model"]["cor_abl"]
        if "bias_cor" in params["model"]:
            modelParams["bias_cor"] = params["model"]["bias_cor"]
    elif modelParams["type"] == "multi":
        modelParams["glacio"] = {
            "type": params["model"]["glacio"]["type"],
            # "inputs": params["model"]["glacio"]["inputs"],
            "layers": params["model"]["glacio"]["layers"],
            "dropout": params["model"]["glacio"].get("dropout", 0.0),
        }
        modelParams["geo"] = {
            "type": params["model"]["geo"]["type"],
            "inputs": params["model"]["geo"]["inputs"],
            "layers": params["model"]["geo"]["layers"],
            "dropout": params["model"]["geo"].get("dropout", 0.0),
        }
    # if "layers_cor_geodetic" in modelParams:
    #     modelParams["layers_cor_geodetic"] = params["model"]["layers_cor_geodetic"]
    trainingParams = {
        "source_data": source_data,
        "geodetic_source": geodetic_source,
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
        "freqVal": params["training"].get("freqVal", 1),
        "log_suffix": params["training"].get("log_suffix", ""),
        "log_prefix": params["training"].get("log_prefix", ""),
        "log_dir": params["training"].get("log_dir"),
        "wWinter": params["training"].get("wWinter", 1.0),
        "wSummer": params["training"].get("wSummer", 1.0),
    }
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
        "RGI60-11.03166": "Grand Etret",
        "RGI60-11.00647": "Gigante Occidentale (Ries Ovest) / Westl. Rieser",
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
        "RGI60-11.00781": "Jamtalferner",
        "RGI60-11.00116": "Venedigerkees",
        "RGI60-11.00251": "Kleinfleisskees",
        "RGI60-11.00289": "Goldbergkees",
        "RGI60-11.00006": "Schladminger",
        # Switzerland
        "RGI60-11.01270": "Grindelwald",
        "RGI60-11.01450": "Aletsch",
        "RGI60-11.01733": "Hangend",
        "RGI60-11.01328": "Unteraar",
        "RGI60-11.01238": "Rhone",
        "RGI60-11.02249": "Tsanfleuron",
        "RGI60-11.01702": "Kander",
        "RGI60-11.00872": "Hüfifirn",
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
        "RGI60-11.00878": "Claridenfirn I",
        "RGI60-11.00843": "Claridenfirn II",
        "RGI60-11.00819": "Claridenfirn III",
        "RGI60-11.01962": "Corvatsch",
        "RGI60-11.02740": "Trient",
        "RGI60-11.01367": "St. Annafirn",
        "RGI60-11.02745": "Allalin",
        "RGI60-11.01280": "Glatscher da Plattas",
        "RGI60-11.02679": "Hohlaubgletscher",
        "RGI60-11.02773": "Findelen",
        "RGI60-11.02448": "Plan Névé",
        "RGI60-11.02282": "Vadrec dal Castel Nord",
        "RGI60-11.02624": "Feegletscher",
    }.get(rgi_id)
