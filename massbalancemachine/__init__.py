import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import importlib

__all__ = [
    "dataloader",
    "models",
    "data_processing",
    "geodata",
    "config",
    "training",
    "plots",
    "metrics",
    "sampling",
    "utils",
]

import dataloader
import models
import data_retrieval
import data_preprocessing
import data_processing
import geodata
import training
import plots
import metrics

# import sampling # Do not import by default since this is an advanced feature
import utils
from .config import *  # Load config at the top level of the package


# Import only if the user asks for it
def __getattr__(name):
    if name == "sampling":
        return importlib.import_module(".sampling", __name__)
    raise AttributeError(f"module {__name__} has no attribute {name}")
