import os
import logging
import pandas as pd
import massbalancemachine as mbm
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    train_test_split,
    GroupShuffleSplit,
)
import geopandas as gpd
import xarray as xr

from regions.Iceland.scripts.config_ICE import *


def process_or_load_data(
    run_flag,
    df,
    paths,
    cfg,
    vois_climate,
    vois_topographical,
    output_file="ICE_dataset_monthly_full.csv",
):
    """
    Process or load the data based on the RUN flag.
    """
    if run_flag:
        logging.info("Number of annual and seasonal samples: %d", len(df))

        # Filter data
        logging.info(
            "Running on %d glaciers:\n%s", len(df.GLACIER.unique()), df.GLACIER.unique()
        )

        # Create dataset
        dataset_gl = mbm.data_processing.Dataset(
            cfg=cfg, data=df, region_name="ICE", data_path=paths["csv_path"]  # Region
        )
        for period in df["PERIOD"].unique():
            count = len(df[df.PERIOD == period])
            logging.info("Number of %s samples: %d", period, count)

        # Add climate data
        logging.info("Adding climate features...")
        try:
            dataset_gl.get_climate_features(
                climate_data=paths["era5_climate_data"],
                geopotential_data=paths["geopotential_data"],
                change_units=True,
            )
        except Exception as e:
            logging.error("Failed to add climate features: %s", e)
            return None

        # Convert to monthly resolution
        logging.info("Converting to monthly resolution...")
        dataset_gl.convert_to_monthly(
            meta_data_columns=cfg.metaData,
            vois_climate=vois_climate,
            vois_topographical=vois_topographical,
        )

        # Create DataLoader
        dataloader_gl = mbm.dataloader.DataLoader(
            cfg,
            data=dataset_gl.data,
            random_seed=cfg.seed,
            meta_data_columns=cfg.metaData,
        )
        logging.info("Number of monthly rows: %d", len(dataloader_gl.data))
        logging.info("Columns in the dataset: %s", dataloader_gl.data.columns)

        # Save processed data
        output_file = os.path.join(paths["csv_path"], output_file)
        dataloader_gl.data.to_csv(output_file, index=False)
        logging.info("Processed data saved to: %s", output_file)

        return dataloader_gl
    else:
        # Load preprocessed data
        try:
            input_file = os.path.join(paths["csv_path"], output_file)
            data_monthly = pd.read_csv(input_file)
            dataloader_gl = mbm.dataloader.DataLoader(
                cfg,
                data=data_monthly,
                random_seed=cfg.seed,
                meta_data_columns=cfg.metaData,
            )
            logging.info("Loaded preprocessed data.")
            logging.info("Number of monthly rows: %d", len(dataloader_gl.data))
            for period in dataloader_gl.data["PERIOD"].unique():
                count = len(dataloader_gl.data[dataloader_gl.data.PERIOD == period])
                logging.info("Number of %s samples: %d", period, count)

            return dataloader_gl
        except FileNotFoundError as e:
            logging.error("Preprocessed data file not found: %s", e)
            return None


def get_CV_splits(
    dataloader_gl, test_split_on="YEAR", test_splits=None, random_state=0, test_size=0.2
):
    # Split into training and test splits with train_test_split
    if test_splits is None:
        train_splits, test_splits = train_test_split(
            dataloader_gl.data[test_split_on].unique(),
            test_size=test_size,
            random_state=random_state,
        )
    else:
        split_data = dataloader_gl.data[test_split_on].unique()
        train_splits = [x for x in split_data if x not in test_splits]
    train_indices = dataloader_gl.data[
        dataloader_gl.data[test_split_on].isin(train_splits)
    ].index
    test_indices = dataloader_gl.data[
        dataloader_gl.data[test_split_on].isin(test_splits)
    ].index

    dataloader_gl.set_custom_train_test_indices(train_indices, test_indices)

    # Get the features and targets of the training data for the indices as defined above, that will be used during the cross validation.
    df_X_train = dataloader_gl.data.iloc[train_indices]
    y_train = df_X_train["POINT_BALANCE"].values
    train_meas_id = df_X_train["ID"].unique()

    # Get test set
    df_X_test = dataloader_gl.data.iloc[test_indices]
    y_test = df_X_test["POINT_BALANCE"].values
    test_meas_id = df_X_test["ID"].unique()

    # Values split in training and test set
    train_splits = df_X_train[test_split_on].unique()
    test_splits = df_X_test[test_split_on].unique()

    # Create the CV splits based on the training dataset. The default value for the number of splits is 5.
    cv_splits = dataloader_gl.get_cv_split(n_splits=5, type_fold="group-meas-id")

    test_set = {
        "df_X": df_X_test,
        "y": y_test,
        "meas_id": test_meas_id,
        "splits_vals": test_splits,
    }
    train_set = {
        "df_X": df_X_train,
        "y": y_train,
        "splits_vals": train_splits,
        "meas_id": train_meas_id,
    }

    return cv_splits, test_set, train_set


def getDfAggregatePred(test_set, y_pred_agg, all_columns):
    # Aggregate predictions to annual or winter:
    df_pred = test_set["df_X"][all_columns].copy()
    df_pred["target"] = test_set["y"]
    grouped_ids = df_pred.groupby("ID").agg(
        {"target": "mean", "YEAR": "first", "POINT_ID": "first", "GLACIER": "first"}
    )
    grouped_ids["pred"] = y_pred_agg
    grouped_ids["PERIOD"] = (
        test_set["df_X"][all_columns].groupby("ID")["PERIOD"].first()
    )

    return grouped_ids
