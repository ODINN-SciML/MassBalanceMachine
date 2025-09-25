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
import git
import argparse
import numpy as np
from skorch.callbacks import EarlyStopping, LRScheduler, Callback
from skorch.helper import SliceDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from scripts.common import (
    trainTestGlaciers,
    getTrainTestSetsSwitzerland,
    seed_all,
    loadParams,
)
from scripts.nongeo.utils import (
    getMetaData,
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

train_set, test_set, data_glamos, months_head_pad, months_tail_pad = (
    getTrainTestSetsSwitzerland(
        train_glaciers,
        test_glaciers,
        params,
        cfg,
        "CH_wgms_dataset_monthly_NN_nongeo.csv",
        process=False,
    )
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


model = mbm.models.buildModel(cfg, params=params)


class TensorBoardMetricLogger(Callback):
    def __init__(self, logdir, additional_scores_funcs={}):
        self.writer = SummaryWriter(logdir)
        self.additional_scores_funcs = additional_scores_funcs

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
        self.writer.add_scalar("MAE/val", mae, epoch)
        self.writer.add_scalar("Pearson/val", pearson, epoch)
        self.writer.add_scalar("R2/val", r2, epoch)
        self.writer.add_scalar("bias/val", bias, epoch)

        for freq, func in self.additional_scores_funcs.items():
            if epoch % freq == freq - 1:
                scores = func(model, dataset_valid)
                for k, v in scores.items():
                    self.writer.add_scalar(k, v, epoch)

    def on_train_end(self, model, **kwargs):
        self.writer.close()


logdir = getLogDir(suffix)


def annual_winter_func(model, dataset):
    Xval = SliceDataset(dataset, idx=0)
    yval = SliceDataset(dataset, idx=1)
    Mval = [dataset.getMetadata(i) for i in range(len(dataset))]
    Mval = [Mval[i].reshape(len(yval[i]), -1) for i in range(len(Mval))]

    # Retrieve the period associated to each element of the batch
    posPeriod = dataset.metadataColumns.index("PERIOD")
    indAnnual = []
    indWinter = []
    for i in range(len(Mval)):
        period = Mval[i][0, posPeriod]
        if period == "annual":
            indAnnual.append(i)
        elif period == "winter":
            indWinter.append(i)
        else:
            raise ValueError(f"Period {period} is unknown.")

    # Make predictions
    y_pred = model.predict(Xval)
    y_pred_agg = model.aggrPredict(Xval)

    # Get true values
    batchIndex = np.arange(len(y_pred_agg))
    y_true = np.array([e for e in yval[batchIndex]])

    y_pred_annual = y_pred[indAnnual]
    y_true_annual = y_true[indAnnual]
    y_pred_winter = y_pred[indWinter]
    y_true_winter = y_true[indWinter]

    mse_annual, rmse_annual, mae_annual, pearson_annual, r2_annual, bias_annual = (
        model.evalMetrics(y_pred_annual, y_true_annual)
    )
    mse_winter, rmse_winter, mae_winter, pearson_winter, r2_winter, bias_winter = (
        model.evalMetrics(y_pred_winter, y_true_winter)
    )

    scores = {
        "RMSE_annual/val": rmse_annual,
        "RMSE_winter/val": rmse_winter,
        "MAE_annual/val": mae_annual,
        "MAE_winter/val": mae_winter,
        "Pearson_annual/val": pearson_annual,
        "Pearson_winter/val": pearson_winter,
        "R2_annual/val": r2_annual,
        "R2_winter/val": r2_winter,
        "bias_annual/val": bias_annual,
        "bias_winter/val": bias_winter,
    }
    return scores


additional_scores_funcs = {
    10: annual_winter_func
}  # Compute the annual/winter scores every 10 epochs
logger = TensorBoardMetricLogger(
    logdir, additional_scores_funcs=additional_scores_funcs
)

callbacks = [
    ("early_stop", early_stop),
    ("lr_scheduler", lr_scheduler_cb),
    ("logger", logger),
]
args = buildArgs(cfg, params, model, my_train_split, callbacks=callbacks)


custom_nn = mbm.models.CustomNeuralNetRegressor(cfg, **args, **param_init)


dataset, dataset_val = getDatasets(
    cfg,
    df_X_train,
    y_train,
    df_X_val,
    y_val,
    test_set["df_X"],
    custom_nn,
    months_head_pad,
    months_tail_pad,
)


custom_nn.seed_all()

print("Training the model...")
print("Model parameters:")
for key, value in args.items():
    print(f"{key}: {value}")
custom_nn.fit(dataset.X, dataset.y)
# The dataset provided in fit is not used as the datasets are overwritten in the provided train_split function

# Save the model
custom_nn.save_model(model_dir=logdir)

plot_training_history(custom_nn.history, skip_first_n=5, save=False)
plt.savefig(os.path.join(logdir, "training_history.pdf"))
plt.close()

repo = git.Repo(search_parent_directories=True)
params["commit_hash"] = repo.head.object.hexsha
with open(os.path.join(logdir, "params.json"), "w") as f:
    json.dump(params, f, indent=4, sort_keys=True)
