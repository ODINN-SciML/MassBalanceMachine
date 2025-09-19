import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from skorch.helper import SliceDataset
import massbalancemachine as mbm


@pytest.mark.order3
def test_iceland_train():
    data = pd.read_csv(
        "./notebooks/example_data/iceland/files/iceland_monthly_dataset.csv"
    )
    print("Number of winter and annual samples:", len(data))

    months_head_pad, months_tail_pad = (
        mbm.data_processing.utils.build_head_tail_pads_from_monthly_df(data)
    )

    cfg = mbm.Config()

    # Create a new DataLoader object with the monthly stake data measurements.
    dataloader = mbm.dataloader.DataLoader(cfg, data=data)
    # Create a training and testing iterators. The parameters are optional. The default value of test_size is 0.3.
    train_itr, test_itr = dataloader.set_train_test_split(test_size=0.3)

    # Get all indices of the training and testing dataset at once from the iterators. Once called, the iterators are empty.
    train_indices, test_indices = list(train_itr), list(test_itr)

    # Get the features and targets of the training data for the indices as defined above, that will be used during the cross validation.
    df_X_train = data.iloc[train_indices]
    y_train = df_X_train["POINT_BALANCE"].values

    # Get test set
    df_X_test = data.iloc[test_indices]
    y_test = df_X_test["POINT_BALANCE"].values

    # Create the cross validation splits based on the training dataset. The default value for the number of splits is 5.
    type_fold = "group-meas-id"  # 'group-rgi' # or 'group-meas-id'
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
        **params_init,
    )

    features, metadata = mbm.data_processing.utils.create_features_metadata(
        cfg, df_X_train
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
    splits = dataset.mapSplitsToDataset(splits)

    # Use SliceDataset to make the dataset accessible as a numpy array for scikit learn
    dataset = [SliceDataset(dataset, idx=0), SliceDataset(dataset, idx=1)]
    print(dataset[0].shape, dataset[1].shape)

    custom_nn.set_params(lr=0.01, max_epochs=8)
    custom_nn.fit(dataset[0], dataset[1])

    ## Test plot function

    # Make predictions on test
    features_test, metadata_test = mbm.data_processing.utils.create_features_metadata(
        cfg, df_X_test
    )

    dataset_test = mbm.data_processing.AggregatedDataset(
        cfg,
        features=features_test,
        metadata=metadata_test,
        months_head_pad=months_head_pad,
        months_tail_pad=months_tail_pad,
        targets=y_test,
    )
    month_pos = dataset_test.month_pos
    col_idx_rgiid = dataset_test.metadataColumns.index("RGIId")

    dataset_test = [
        SliceDataset(dataset_test, idx=0),  # Features
        SliceDataset(dataset_test, idx=1),  # Target
        SliceDataset(dataset_test, idx=2),  # Metadata
    ]

    # Make predictions aggr to meas ID
    y_pred = custom_nn.predict(dataset_test[0])
    y_pred_agg = custom_nn.aggrPredict(dataset_test[0])

    batchIndex = np.arange(len(y_pred_agg))
    y_true = np.array([e for e in dataset_test[1][batchIndex]])

    # Calculate scores
    score = custom_nn.score(dataset_test[0], dataset_test[1])
    mse, rmse, mae, pearson, r2, bias = custom_nn.evalMetrics(y_pred, y_true)

    # Aggregate predictions
    ID = dataset_test[0].dataset.indexToId(batchIndex)
    data = {
        "target": [e[0] for e in dataset_test[1]],
        "ID": ID,
        "pred": y_pred_agg,
        "RGIId": [e[col_idx_rgiid] for e in dataset_test[2]],
    }
    grouped_ids = pd.DataFrame(data)

    scores = {"rmse": rmse, "mae": mae, "R2": r2}
    fig = mbm.plots.predVSTruth(
        grouped_ids,
        scores=scores,
        marker="o",
        title="NN on test",
        alpha=0.5,
    )

    scores = {}
    for i, test_gl in enumerate(grouped_ids["RGIId"].unique()):
        df_gl = grouped_ids[grouped_ids["RGIId"] == test_gl]
        glacier_scores = mbm.metrics.scores(df_gl["target"], df_gl["pred"])
        scores[test_gl] = {
            "rmse": glacier_scores["rmse"],
            "R2": glacier_scores["r2"],
            "B": glacier_scores["bias"],
        }

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    mbm.plots.predVSTruthPerGlacier(
        grouped_ids,
        axs=axs,
        scores=scores,
    )


if __name__ == "__main__":
    test_iceland_train()
