import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

import warnings
import matplotlib.pyplot as plt
from cmcrameri import cm
import massbalancemachine as mbm
import logging
import torch
import json
import argparse
from datetime import datetime
from skorch.callbacks import EarlyStopping, LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from scripts.common import (
    getTrainTestSets,
    seed_all,
    loadParams,
)
from scripts.legacy.utils import (
    getMetaData,
    trainTestGlaciers,
    getDatasets,
    buildArgs,
    trainValData,
    setFeatures,
)

from regions.Switzerland.scripts.helpers import get_cmap_hex
from regions.Switzerland.scripts.nn_helpers import plot_training_history

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("modelType", type=str, help="Type of model to train.")
parser.add_argument(
    "--gpu",
    type=bool,
    default=False,
    help="Train on GPU. By default training runs on CPU.",
)
args = parser.parse_args()

params = loadParams(args.modelType)
featuresInpModel = params["model"]["inputs"]
runOnGpu = args.gpu

metaData = getMetaData(featuresInpModel)


cfg = mbm.SwitzerlandConfig(
    metaData=metaData,
    notMetaDataNotFeatures=["POINT_BALANCE"],
)
seed_all(cfg.seed)


# Plot styles:
path_style_sheet = "regions/Switzerland/scripts/example.mplstyle"
plt.style.use(path_style_sheet)
colors = get_cmap_hex(cm.batlow, 10)
color_dark_blue = colors[0]
color_pink = "#c51b7d"


if torch.cuda.is_available():
    print("CUDA is available")
    # free_up_cuda()
else:
    print("CUDA is NOT available")


# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

print(params)
train_glaciers, test_glaciers = trainTestGlaciers(params)

train_set, test_set, data_glamos = getTrainTestSets(
    train_glaciers,
    test_glaciers,
    params,
    cfg,
    "CH_wgms_dataset_monthly_NN_nongeo.csv",
    process=False,
)

data_train = train_set["df_X"]
data_train["y"] = train_set["y"]

feature_columns = setFeatures(cfg, data_train, featuresInpModel)
df_X_train, y_train, df_X_val, y_val = trainValData(cfg, train_set, feature_columns)


early_stop = EarlyStopping(
    monitor="valid_loss",
    patience=20,
    threshold=1e-4,  # Optional: stop only when improvement is very small
)
lr_scheduler_cb = LRScheduler(
    policy=ReduceLROnPlateau,
    monitor="valid_loss",
    mode="min",
    factor=0.5,
    patience=5,
    threshold=0.01,
    threshold_mode="rel",
    verbose=True,
)

dataset = dataset_val = None  # Initialized hereafter


def my_train_split(ds, y=None, **fit_params):
    return dataset, dataset_val


param_init = {"device": "cuda:0" if runOnGpu else "cpu"}
nInp = len(feature_columns)


model = mbm.models.buildModel(cfg, params=params)

callbacks = [
    ("early_stop", early_stop),
    ("lr_scheduler", lr_scheduler_cb),
]
args = buildArgs(cfg, params, model, my_train_split, callbacks=callbacks)


custom_nn = mbm.models.CustomNeuralNetRegressor(cfg, **args, **param_init)


dataset, dataset_val = getDatasets(
    cfg, df_X_train, y_train, df_X_val, y_val, test_set["df_X"], custom_nn
)


custom_nn.seed_all()

print("Training the model...")
print("Model parameters:")
for key, value in args.items():
    print(f"{key}: {value}")
custom_nn.fit(dataset.X, dataset.y)
# The dataset provided in fit is not used as the datasets are overwritten in the provided train_split function

# Generate filename with current date
current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = f"nn_{current_date}"
model_filename = f"{model_dir}/model.pt"

# Save the model
pathSave = custom_nn.save_model(model_filename)
pathFolder = os.path.dirname(pathSave)

plot_training_history(custom_nn, skip_first_n=5, save=False)
plt.savefig(os.path.join(pathFolder, "training_history.pdf"))
plt.close()

with open(os.path.join(pathFolder, "params.json"), "w") as f:
    json.dump(params, f, indent=4, sort_keys=True)
