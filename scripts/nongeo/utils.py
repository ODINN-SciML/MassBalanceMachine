import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

# import warnings
from datetime import datetime
import massbalancemachine as mbm
import numpy as np
import torch
import torch.nn as nn
from skorch.helper import SliceDataset
from massbalancemachine.cli.common import (
    getMetaData,
    setFeatures,
    trainValData,
    testData,
)


def getDatasets(
    cfg,
    df_X_train,
    y_train,
    df_X_val,
    y_val,
    df_test,
    custom_nn,
    months_head_pad,
    months_tail_pad,
):
    features, metadata = mbm.data_processing.utils.create_features_metadata(
        cfg, df_X_train
    )
    if np.isnan(features).any():
        print(
            f"Summary of the columns (out of {len(df_X_train)} rows) that contain NaN values:"
        )
        print(df_X_train.isna().sum())
        raise ValueError("Training features contain NaN, check the details above.")

    features_val, metadata_val = mbm.data_processing.utils.create_features_metadata(
        cfg, df_X_val
    )
    if np.isnan(features_val).any():
        print(
            f"Summary of the columns (out of {len(df_X_val)} rows) that contain NaN values:"
        )
        print(df_X_val.isna().sum())
        raise ValueError("Validation features contain NaN, check the details above.")

    # Define the dataset for the NN
    dataset = mbm.data_processing.AggregatedDataset(
        cfg,
        features=features,
        metadata=metadata,
        months_head_pad=months_head_pad,
        months_tail_pad=months_tail_pad,
        targets=y_train,
    )
    dataset = mbm.data_processing.SliceDatasetBinding(
        SliceDataset(dataset, idx=0),
        SliceDataset(dataset, idx=1),
        M=SliceDataset(dataset, idx=2),
        metadataColumns=dataset.metadataColumns,
    )
    print("train:", dataset.X.shape, dataset.y.shape)

    dataset_val = mbm.data_processing.AggregatedDataset(
        cfg,
        features=features_val,
        metadata=metadata_val,
        months_head_pad=months_head_pad,
        months_tail_pad=months_tail_pad,
        targets=y_val,
    )
    dataset_val = mbm.data_processing.SliceDatasetBinding(
        SliceDataset(dataset_val, idx=0),
        SliceDataset(dataset_val, idx=1),
        M=SliceDataset(dataset_val, idx=2),
        metadataColumns=dataset.metadataColumns,
    )
    print("validation:", dataset_val.X.shape, dataset_val.y.shape)
    return dataset, dataset_val


class NetworkBinding(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def buildArgs(cfg, params, model, train_split, callbacks=[]):
    lr = params["training"]["lr"]
    optimType = params["training"]["optim"]
    Nepochs = params["training"]["Nepochs"]
    batch_size = params["training"]["batch_size"]
    weight_decay = params["training"]["weight_decay"]
    if optimType == "ADAM":
        optim = torch.optim.Adam
    elif optimType == "SGD":
        optim = torch.optim.SGD
    else:
        raise ValueError(f"Optimizer {optimType} is not supported.")

    nInp = len(cfg.featureColumns)
    args = {
        "module": NetworkBinding,
        "nbFeatures": nInp,
        "module__model": model,
        "train_split": train_split,
        "batch_size": batch_size,
        "verbose": 1,
        "iterator_train__shuffle": True,
        "lr": lr,
        "max_epochs": Nepochs,
        "optimizer": optim,
        "optimizer__weight_decay": weight_decay,
        "callbacks": callbacks,
    }
    return args


def getLogDir(suffix=None):
    # Generate filename with current date
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffixStr = f"_{suffix}" if suffix is not None else ""
    logdir = f"logs/nongeo_{run_name}{suffixStr}"
    print(f"Logging in {logdir}")
    return logdir
