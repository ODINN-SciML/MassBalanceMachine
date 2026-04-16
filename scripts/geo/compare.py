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
# parser.add_argument(
#     "--cpu",
#     dest="cpu",
#     default=False,
#     action="store_true",
#     help="Force model to run on CPU, even if a GPU is available.",
# )
parser.add_argument(
    "--plot",
    dest="plot",
    default=False,
    action="store_true",
    help="Display figures in addition to saving.",
)
# parser.add_argument(
#     "--noTest",
#     dest="noTest",
#     default=False,
#     action="store_true",
#     help="Do not evaluate on test data.",
# )
# parser.add_argument(
#     "--onRegion",
#     dest="onRegion",
#     default=False,
#     action="store_true",
#     help="Evaluate prediction on the whole region in addition to classical plots.",
# )
args = parser.parse_args()

modelFolder1 = args.modelFolder1
modelFolder2 = args.modelFolder2
# cpu = args.cpu
plot = args.plot
# noTest = args.noTest
# onRegion = args.onRegion
pathFolder1 = os.path.join("logs", modelFolder1)
pathFolder2 = os.path.join("logs", modelFolder2)

pathFolder = os.path.join("results/comp/", f"{modelFolder1}_{modelFolder2}")
os.makedirs(pathFolder, exist_ok=True)

with open(f"{pathFolder1}/params.json", "r") as f:
    params1 = json.load(f)
with open(f"{pathFolder2}/params.json", "r") as f:
    params2 = json.load(f)


# Cumulated mass change on train data
df_gridded_monthly1 = pd.read_csv(f"{pathFolder1}/gridded_monthly_train.csv")
df_geo1 = pd.read_csv(f"{pathFolder1}/gridded_geodetic_train.csv")
geoTarget = df_geo1.set_index("RGIId").target.to_dict()
geoErr = df_geo1.set_index("RGIId").err.to_dict()

# Plot cumulated mass change
fig = mbm.plots.cumulatedMassChange(
    df_gridded_monthly1,
    geo={
        rgi_id: {"mean": geoTarget[rgi_id], "err": geoErr[rgi_id]}
        for rgi_id in geoTarget
    },
)
del df_gridded_monthly1
df_gridded_monthly2 = pd.read_csv(f"{pathFolder2}/gridded_monthly_train.csv")
mbm.plots.cumulatedMassChange(
    df_gridded_monthly2,
    geo=None,
    axs=fig.axes,
    color_pred="orange",
)


fig.savefig(f"{pathFolder}/cumulated_mass_change_glaciers_train.pdf")
fig.savefig(f"{pathFolder}/cumulated_mass_change_glaciers_train.png", dpi=300)
if plot:
    plt.show()
plt.close(fig)


# Cumulated mass change on test data
df_gridded_monthly1 = pd.read_csv(f"{pathFolder1}/gridded_monthly_test.csv")
df_geo1 = pd.read_csv(f"{pathFolder1}/gridded_geodetic_test.csv")
geoTarget = df_geo1.set_index("RGIId").target.to_dict()
geoErr = df_geo1.set_index("RGIId").err.to_dict()

# Plot cumulated mass change
fig = mbm.plots.cumulatedMassChange(
    df_gridded_monthly1,
    geo={
        rgi_id: {"mean": geoTarget[rgi_id], "err": geoErr[rgi_id]}
        for rgi_id in geoTarget
    },
)
del df_gridded_monthly1
df_gridded_monthly2 = pd.read_csv(f"{pathFolder2}/gridded_monthly_test.csv")
mbm.plots.cumulatedMassChange(
    df_gridded_monthly2,
    geo=None,
    axs=fig.axes,
    color_pred="orange",
)


fig.savefig(f"{pathFolder}/cumulated_mass_change_glaciers_test.pdf")
fig.savefig(f"{pathFolder}/cumulated_mass_change_glaciers_test.png", dpi=300)
if plot:
    plt.show()
plt.close(fig)
