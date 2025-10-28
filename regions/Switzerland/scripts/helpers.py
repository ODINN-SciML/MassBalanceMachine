import torch
import numpy as np
import random as rd
import os
import gc
import shutil
from matplotlib.colors import to_hex
from matplotlib import pyplot as plt
import random
import logging
import massbalancemachine as mbm
import pandas as pd

from sklearn.model_selection import train_test_split


def seed_all(seed=None):
    # Python
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN deterministic kernels
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Forbid nondeterministic ops (warn if an op has no deterministic impl)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Setting CUBLAS environment variable (helps in newer versions)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"


def free_up_cuda():
    """Frees up unused CUDA memory in PyTorch."""
    gc.collect()  # Run garbage collection
    torch.cuda.empty_cache()  # Free unused cached memory
    torch.cuda.ipc_collect()  # Collect inter-process memory


def get_cmap_hex(cmap, length):
    """
    Function to get a get a list of colours as hex codes

    :param cmap:    name of colourmap
    :type cmap:     str

    :return:        list of hex codes
    :rtype:         list
    """
    # Get cmap
    rgb = plt.get_cmap(cmap)(np.linspace(0, 1, length))

    # Convert to hex
    hex_codes = [to_hex(rgb[i, :]) for i in range(rgb.shape[0])]

    return hex_codes


def emptyfolder(path):
    """Removes all files and subdirectories in the given folder."""
    if os.path.exists(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)  # Remove file
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # Remove folder and all contents
            except Exception as e:
                print(f"Error removing {item_path}: {e}")
    else:
        os.makedirs(path, exist_ok=True)  # Ensure directory exists


# difference between two lists
def Diff(li1, li2):
    li_dif = list(set(li1) - set(li2))
    return li_dif


def format_rgi_code(X):
    # Convert X to a string, and pad with leading zeros if its length is less than 5
    Y = str(X).zfill(5)
    # Return the final formatted string
    return f"RGI60-11.{Y}"


def process_or_load_data(
    run_flag,
    data_glamos,
    paths,
    cfg,
    vois_climate,
    vois_topographical,
    add_pcsr=True,
    output_file="CH_wgms_dataset_monthly_full.csv",
):
    """
    Process or load the data based on the RUN flag.
    """
    if run_flag:
        logging.info("Number of annual and seasonal samples: %d", len(data_glamos))

        # Filter data
        logging.info(
            "Running on %d glaciers:\n%s",
            len(data_glamos.GLACIER.unique()),
            data_glamos.GLACIER.unique(),
        )

        # Add a glacier-wide ID (used for geodetic MB)
        data_glamos["GLWD_ID"] = data_glamos.apply(
            lambda x: mbm.data_processing.utils.get_hash(f"{x.GLACIER}_{x.YEAR}"),
            axis=1,
        )
        data_glamos["GLWD_ID"] = data_glamos["GLWD_ID"].astype(str)

        # Create dataset
        dataset_gl = mbm.data_processing.Dataset(
            cfg=cfg,
            data=data_glamos,
            region_name="CH",
            region_id=11,
            data_path=paths["csv_path"],
        )
        logging.info("Number of winter and annual samples: %d", len(data_glamos))
        logging.info(
            "Number of annual samples: %d",
            len(data_glamos[data_glamos.PERIOD == "annual"]),
        )
        logging.info(
            "Number of winter samples: %d",
            len(data_glamos[data_glamos.PERIOD == "winter"]),
        )

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

        if add_pcsr:
            # Add radiation data
            logging.info("Adding potential clear sky radiation...")
            logging.info("Shape before adding radiation: %s", dataset_gl.data.shape)
            dataset_gl.get_potential_rad(paths["radiation_save_path"])
            logging.info("Shape after adding radiation: %s", dataset_gl.data.shape)

        # Convert to monthly resolution
        logging.info("Converting to monthly resolution...")
        if add_pcsr:
            dataset_gl.convert_to_monthly(
                meta_data_columns=cfg.metaData,
                vois_climate=vois_climate + ["pcsr"],
                vois_topographical=vois_topographical,
            )
        else:
            dataset_gl.convert_to_monthly(
                meta_data_columns=cfg.metaData,
                vois_climate=vois_climate,
                vois_topographical=vois_topographical,
            )

        # add glwd_id
        data_monthly = dataset_gl.data

        data_monthly["GLWD_ID"] = data_monthly.apply(
            lambda x: mbm.data_processing.utils.get_hash(f"{x.GLACIER}_{x.YEAR}"),
            axis=1,
        )
        data_monthly["GLWD_ID"] = data_monthly["GLWD_ID"].astype(str)

        logging.info("Number of monthly rows: %d", len(data_monthly))
        logging.info("Columns in the dataset: %s", data_monthly.columns)

        # Save processed data
        output_file = os.path.join(paths["csv_path"], output_file)
        data_monthly.to_csv(output_file, index=False)
        logging.info("Processed data saved to: %s", output_file)

        return data_monthly
    else:
        # Load preprocessed data
        try:
            input_file = os.path.join(paths["csv_path"], output_file)
            data_monthly = pd.read_csv(input_file)
            filt = data_monthly.filter(["YEAR.1", "POINT_LAT.1", "POINT_LON.1"])
            data_monthly.drop(filt, inplace=True, axis=1)
            logging.info("Loaded preprocessed data.")
            logging.info("Number of monthly rows: %d", len(data_monthly))
            logging.info(
                "Number of annual rows: %d",
                len(data_monthly[data_monthly.PERIOD == "annual"]),
            )
            logging.info(
                "Number of winter rows: %d",
                len(data_monthly[data_monthly.PERIOD == "winter"]),
            )

            return data_monthly
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
