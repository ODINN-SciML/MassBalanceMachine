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
import numpy as np
from datetime import datetime
from skorch.callbacks import EarlyStopping, LRScheduler, Callback
from skorch.helper import SliceDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from scripts.common import (
    getTrainTestSets,
    seed_all,
    loadParams,
)
from scripts.nongeo.utils import (
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
parser.add_argument(
    "-s",
    "--suffix",
    type=str,
    default=None,
    help="Suffix to add to the folder that contains the model once trained.",
)
args = parser.parse_args()

runOnGpu = args.gpu
suffix = args.suffix
params = loadParams(args.modelType)
featuresInpModel = params["model"]["inputs"]

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


class TensorBoardMetricLogger(Callback):
    def __init__(self, logdir=None):
        if logdir is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffixStr = f"_{suffix}" if suffix is not None else ""
            logdir = f"logs/nongeo_{run_name}{suffixStr}"
        self.writer = SummaryWriter(logdir)

    def on_batch_end(self, model, Xi=None, yi=None, training=False, **kwargs):
        if not training:
            return
        step = len(model.history[-1]["batches"]) + sum(
            len(e["batches"]) for e in model.history[:-1]
        )

        # training or validation loss
        loss = kwargs.get("loss")
        if loss is not None:
            tag = "Loss/train"
            self.writer.add_scalar(tag, float(loss), step)

        # Log learning rate (assuming one param group)
        lr = model.optimizer_.param_groups[0]["lr"]
        self.writer.add_scalar("Step", lr, step)

    def on_epoch_end(self, model, dataset_train=None, dataset_valid=None, **kwargs):
        if dataset_valid is None:
            return

        Xval = SliceDataset(dataset_valid, idx=0)
        yval = SliceDataset(dataset_valid, idx=1)

        # Make predictions
        y_pred = model.predict(Xval)
        y_pred_agg = model.aggrPredict(Xval)

        # Get true values
        batchIndex = np.arange(len(y_pred_agg))
        y_true = np.array([e for e in yval[batchIndex]])

        # Calculate scores
        loss = -model.score(Xval, yval)
        mse, rmse, mae, pearson, r2, bias = model.evalMetrics(y_pred, y_true)

        epoch = len(model.history)

        # Log to TensorBoard
        self.writer.add_scalar("Loss/val", loss, epoch)
        self.writer.add_scalar("RMSE/val", rmse, epoch)
        # self.writer.add_scalar("RMSE_annual/val", rmse_annual, epoch)
        # self.writer.add_scalar("RMSE_winter/val", rmse_winter, epoch)
        self.writer.add_scalar("MAE/val", mae, epoch)
        # self.writer.add_scalar("MAE_annual/val", mae_annual, epoch)
        # self.writer.add_scalar("MAE_winter/val", mae_winter, epoch)
        self.writer.add_scalar("Pearson/val", pearson, epoch)
        # self.writer.add_scalar(
        #     "Pearson_annual/val", pearson_corr_annual, epoch
        # )
        # self.writer.add_scalar(
        #     "Pearson_winter/val", pearson_corr_winter, epoch
        # )
        self.writer.add_scalar("R2/val", r2, epoch)
        # self.writer.add_scalar("R2_annual/val", r2_annual, epoch)
        # self.writer.add_scalar("R2_winter/val", r2_winter, epoch)
        self.writer.add_scalar("bias/val", bias, epoch)
        # self.writer.add_scalar("bias_annual/val", bias_annual, epoch)
        # self.writer.add_scalar("bias_winter/val", bias_winter, epoch)

    def on_train_end(self, model, **kwargs):
        self.writer.close()


logger = TensorBoardMetricLogger()

callbacks = [
    ("early_stop", early_stop),
    ("lr_scheduler", lr_scheduler_cb),
    ("logger", logger),
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
suffixStr = f"_{suffix}" if suffix is not None else ""
model_dir = f"nn_{current_date}{suffixStr}"
model_filename = f"{model_dir}/model.pt"

# Save the model
pathSave = custom_nn.save_model(model_filename)
pathFolder = os.path.dirname(pathSave)

plot_training_history(custom_nn, skip_first_n=5, save=False)
plt.savefig(os.path.join(pathFolder, "training_history.pdf"))
plt.close()

with open(os.path.join(pathFolder, "params.json"), "w") as f:
    json.dump(params, f, indent=4, sort_keys=True)
