import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

# import warnings
from datetime import datetime
import massbalancemachine as mbm
import torch
import torch.nn as nn
from skorch.helper import SliceDataset

from scripts.common import (
    getTrainTestSetsSwitzerland,
    _default_input,
    seed_all,
)

# from regions.Switzerland.scripts.helpers import get_cmap_hex

# warnings.filterwarnings('ignore')


def getMetaData(featuresInpModel):
    featuresToRemove = list(set(_default_input) - set(featuresInpModel))
    metaData = list(
        set(
            [
                "RGIId",
                "POINT_ID",
                "ID",
                "GLWD_ID",
                "N_MONTHS",
                "MONTHS",
                "PERIOD",
                "GLACIER",
                "YEAR",
                "POINT_LAT",
                "POINT_LON",
            ]
        ).union(set(featuresToRemove))
    )
    return metaData


def setFeatures(cfg, data_train, featuresInpModel):
    feature_columns = list(
        data_train.columns.difference(cfg.metaData)
        .drop(cfg.notMetaDataNotFeatures)
        .drop("y")
    )
    assert set(feature_columns) == set(
        featuresInpModel
    ), f"Asked features are {featuresInpModel} but the one obtained from the dataframe are {feature_columns}"
    cfg.setFeatures(feature_columns)
    return feature_columns


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

    features_val, metadata_val = mbm.data_processing.utils.create_features_metadata(
        cfg, df_X_val
    )

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


def trainValData(cfg, train_set, feature_columns):
    """
    Split training dataset into train and validation sets.

    Args:
        - cfg: A configuration instance.
        - train_set: Dictionary with at least the following keys: `df_X` (pd.DataFrame) and `y` (pd.Series) which represent respectively the features and the targets.
        - feature_columns: List of string representing the columns to be used as features in the dataframe.
    """
    # Validation and train split:
    data_train = train_set["df_X"]
    data_train["y"] = train_set["y"]
    dataloader = mbm.dataloader.DataLoader(cfg, data=data_train)

    train_itr, val_itr = dataloader.set_train_test_split(test_size=0.2)

    # Get all indices of the training and valing dataset at once from the iterators. Once called, the iterators are empty.
    train_indices, val_indices = list(train_itr), list(val_itr)

    df_X_train = data_train.iloc[train_indices]
    y_train = df_X_train["POINT_BALANCE"].values

    # Get val set
    df_X_val = data_train.iloc[val_indices]
    y_val = df_X_val["POINT_BALANCE"].values

    assert all(data_train.POINT_BALANCE == train_set["y"])

    all_columns = feature_columns + cfg.fieldsNotFeatures
    print("Shape of training dataset:", df_X_train[all_columns].shape)
    print("Shape of validation dataset:", df_X_val[all_columns].shape)
    print("Running with features:", feature_columns)

    return df_X_train, y_train, df_X_val, y_val


def testData(cfg, test_set, feature_columns):
    all_columns = feature_columns + cfg.fieldsNotFeatures
    df_X_test_subset = test_set["df_X"][all_columns]
    print("Shape of testing dataset:", df_X_test_subset.shape)
    print("Running with features:", feature_columns)

    return df_X_test_subset


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
