import os
import pytest
import numpy as np
import torch
import pandas as pd
from torch import nn
from skorch.helper import SliceDataset
import massbalancemachine as mbm


@pytest.mark.order3
def test_iceland_train():
    data = pd.read_csv('./notebooks/example_data/iceland/files/iceland_monthly_dataset.csv')
    print('Number of winter and annual samples:', len(data))

    cfg = mbm.Config()

    # Create a new DataLoader object with the monthly stake data measurements.
    dataloader = mbm.dataloader.DataLoader(cfg, data=data)
    # Create a training and testing iterators. The parameters are optional. The default value of test_size is 0.3.
    train_itr, test_itr = dataloader.set_train_test_split(test_size=0.3)

    # Get all indices of the training and testing dataset at once from the iterators. Once called, the iterators are empty.
    train_indices, test_indices = list(train_itr), list(test_itr)

    # Get the features and targets of the training data for the indices as defined above, that will be used during the cross validation.
    df_X_train = data.iloc[train_indices]
    y_train = df_X_train['POINT_BALANCE'].values

    # Get test set
    df_X_test = data.iloc[test_indices]
    y_test = df_X_test['POINT_BALANCE'].values

    # Create the cross validation splits based on the training dataset. The default value for the number of splits is 5.
    type_fold = 'group-meas-id'  # 'group-rgi' # or 'group-meas-id'
    splits = dataloader.get_cv_split(n_splits=5, type_fold=type_fold)

    # Print size of train and test
    print(f"Size of training set: {len(train_indices)}")
    print(f"Size of test set: {len(test_indices)}")

    feature_columns = df_X_train.columns.difference(cfg.metaData)
    feature_columns = feature_columns.drop(cfg.notMetaDataNotFeatures)
    feature_columns = list(feature_columns)
    nInp = len(feature_columns)
    print(f"{feature_columns=}")
    cfg.setFeatures(feature_columns)

    network = nn.Sequential(
        nn.Linear(nInp, 12),
        nn.ReLU(),
        nn.Linear(12, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
    )

    # Create a CustomNeuralNetRegressor instance
    params_init = {"device": "cpu"}
    custom_nn = mbm.models.CustomNeuralNetRegressor(
        cfg,
        network,
        nbFeatures=nInp,
        train_split=False,  # train_split is disabled since cross validation is handled by the splits variable hereafter
        batch_size=16,
        iterator_train__shuffle=True,
        **params_init
    )

    features, metadata = mbm.data_processing.utils.create_features_metadata(cfg, df_X_train)

    # Define the dataset for the NN
    dataset = mbm.data_processing.AggregatedDataset(
        cfg,
        features=features,
        metadata=metadata,
        targets=y_train
    )
    splits = dataset.mapSplitsToDataset(splits)

    # Use SliceDataset to make the dataset accessible as a numpy array for scikit learn
    dataset = [SliceDataset(dataset, idx=0), SliceDataset(dataset, idx=1)]
    print(dataset[0].shape, dataset[1].shape)


    custom_nn.set_params(lr=0.01, max_epochs=8)
    custom_nn.fit(dataset[0], dataset[1])


if __name__ == "__main__":
    test_iceland_train()
