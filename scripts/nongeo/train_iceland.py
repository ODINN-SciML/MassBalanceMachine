import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

import warnings
import matplotlib.pyplot as plt
import pandas as pd
import massbalancemachine as mbm
import torch
import json
import git
import argparse
import numpy as np
from skorch.callbacks import EarlyStopping, LRScheduler, Callback
from skorch.helper import SliceDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from scripts.common import (
    trainTestGlaciers,
    getTrainTestSetsIceland,
    seed_all,
    loadParams,
)
from scripts.nongeo.utils import (
    getMetaData,
    getDatasets,
    buildArgs,
    trainValData,
    setFeatures,
    getLogDir,
)

from regions.Switzerland.scripts.nn_helpers import plot_training_history

# data = pd.read_csv('./notebooks/example_data/iceland/files/iceland_monthly_dataset.csv')
# print('Number of winter and annual samples:', len(data))


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


cfg = mbm.Config()
seed_all(cfg.seed)


if torch.cuda.is_available():
    print("CUDA is available")
    # free_up_cuda()
else:
    print("CUDA is NOT available")


print(params)
# train_glaciers, test_glaciers = trainTestGlaciers(params)


test_glaciers = []  #'RGI60-06.00228', 'RGI60-06.00232']

train_set, test_set, months_head_pad, months_tail_pad = getTrainTestSetsIceland(
    test_glaciers,
    params,
    cfg,
)

data_train = train_set["df_X"]
data_train["y"] = train_set["y"]

feature_columns = setFeatures(cfg, data_train, featuresInpModel)
df_X_train, y_train, df_X_val, y_val = trainValData(cfg, train_set, feature_columns)


# assert False


# # Create a new DataLoader object with the monthly stake data measurements.
# dataloader = mbm.dataloader.DataLoader(cfg, data=data)
# # Create a training and testing iterators. The parameters are optional. The default value of test_size is 0.3.
# train_itr, test_itr = dataloader.set_train_test_split(test_size=0.3)

# # Get all indices of the training and testing dataset at once from the iterators. Once called, the iterators are empty.
# train_indices, test_indices = list(train_itr), list(test_itr)

# # Get the features and targets of the training data for the indices as defined above, that will be used during the cross validation.
# df_X_train = data.iloc[train_indices]
# y_train = df_X_train['POINT_BALANCE'].values

# # Get test set
# df_X_test = data.iloc[test_indices]
# y_test = df_X_test['POINT_BALANCE'].values

# # Create the cross validation splits based on the training dataset. The default value for the number of splits is 5.
# type_fold = 'group-meas-id'  # 'group-rgi' # or 'group-meas-id'
# splits = dataloader.get_cv_split(n_splits=5, type_fold=type_fold)

# # Print size of train and test
# print(f"Size of training set: {len(train_indices)}")
# print(f"Size of test set: {len(test_indices)}")


# feature_columns = df_X_train.columns.difference(cfg.metaData)
# feature_columns = feature_columns.drop(cfg.notMetaDataNotFeatures)
# feature_columns = list(feature_columns)
# nInp = len(feature_columns)
# cfg.setFeatures(feature_columns)


dataset = dataset_val = None  # Initialized hereafter


def my_train_split(ds, y=None, **fit_params):
    return dataset, dataset_val


param_init = {"device": "cuda:0" if runOnGpu else "cpu"}


model = mbm.models.buildModel(cfg, params=params)


# # Create a CustomNeuralNetRegressor instance
# params_init = {"device": "cpu"}
# custom_nn = mbm.models.CustomNeuralNetRegressor(
#     cfg,
#     model,
#     nbFeatures=nInp,
#     train_split=
#     False,  # train_split is disabled since cross validation is handled by the splits variable hereafter
#     batch_size=16,
#     iterator_train__shuffle=True,
#     **params_init)


logdir = getLogDir(suffix)


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
    patience=100,
    threshold=0.01,
    threshold_mode="rel",
    verbose=True,
)


callbacks = [
    ("early_stop", early_stop),
    ("lr_scheduler", lr_scheduler_cb),
    # ("logger", logger),
]
args = buildArgs(cfg, params, model, my_train_split, callbacks=callbacks)


custom_nn = mbm.models.CustomNeuralNetRegressor(cfg, **args, **param_init)


# features, metadata = mbm.data_processing.utils.create_features_metadata(cfg, df_X_train)
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


# # Define the dataset for the NN
# dataset = mbm.data_processing.AggregatedDataset(
#     cfg,
#     features=features,
#     metadata=metadata,
#     targets=y_train
# )
# # splits = dataset.mapSplitsToDataset(splits)

# # Use SliceDataset to make the dataset accessible as a numpy array for scikit learn
# dataset = [SliceDataset(dataset, idx=0), SliceDataset(dataset, idx=1)]

# print(dataset[0].shape, dataset[1].shape)


# TODO: try to remove POINT_ELEVATION in the features
