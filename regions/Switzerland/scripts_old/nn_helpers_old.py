import os
import ast
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pandas.api.types import CategoricalDtype

import massbalancemachine as mbm
from regions.Switzerland.scripts.plots import *


def plot_training_history(history, skip_first_n=0, save=True):

    # Skip first N entries if specified
    if skip_first_n > 0:
        history = history[skip_first_n:]

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry.get("train_loss") for entry in history]
    valid_loss = [entry.get("valid_loss") for entry in history if "valid_loss" in entry]

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, train_loss, label="Training Loss", marker="o")

    if valid_loss:
        # Align epochs with valid_loss length
        valid_epochs = epochs[: len(valid_loss)]
        plt.plot(valid_epochs, valid_loss, label="Validation Loss", marker="x")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"Training and Validation Loss (Skipped first {skip_first_n} epochs)"
        if skip_first_n > 0
        else "Training and Validation Loss"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save:
        # save the plot
        # Create a folder to save figures (optional)
        save_dir = "figures"
        os.makedirs(save_dir, exist_ok=True)

        # Save the figure
        plt.savefig(
            os.path.join(save_dir, "training_history.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()  # closes the plot to avoid display in notebooks/scripts


def PlotPredictions_NN(grouped_ids):
    fig = plt.figure(figsize=(15, 10))
    colors_glacier = [
        "#a6cee3",
        "#1f78b4",
        "#b2df8a",
        "#33a02c",
        "#fb9a99",
        "#e31a1c",
        "#fdbf6f",
        "#ff7f00",
        "#cab2d6",
        "#6a3d9a",
        "#ffff99",
        "#b15928",
    ]
    color_palette_glaciers = dict(zip(grouped_ids.GLACIER.unique(), colors_glacier))
    ax1 = plt.subplot(2, 2, 1)
    grouped_ids_annual = grouped_ids[grouped_ids.PERIOD == "annual"]

    y_true_mean = grouped_ids_annual["target"]
    y_pred_agg = grouped_ids_annual["pred"]

    scores_annual = mbm.metrics.scores(y_true_mean, y_pred_agg)
    scores_annual.pop("bias")
    scores_annual.pop("r2")
    predVSTruth(
        ax1,
        grouped_ids_annual,
        scores_annual,
        hue="GLACIER",
        palette=color_palette_glaciers,
    )
    ax1.set_title("Annual PMB", fontsize=24)

    grouped_ids_annual.sort_values(by="YEAR", inplace=True)
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title("Mean annual PMB", fontsize=24)
    plotMeanPred(grouped_ids_annual, ax2)

    grouped_ids_winter = grouped_ids[grouped_ids.PERIOD == "winter"]
    y_true_mean = grouped_ids_winter["target"]
    y_pred_agg = grouped_ids_winter["pred"]

    ax3 = plt.subplot(2, 2, 3)
    scores_winter = mbm.metrics.scores(y_true_mean, y_pred_agg)
    scores_winter.pop("bias")
    scores_winter.pop("r2")
    predVSTruth(
        ax3,
        grouped_ids_winter,
        scores_winter,
        hue="GLACIER",
        palette=color_palette_glaciers,
    )
    ax3.set_title("Winter PMB", fontsize=24)

    ax4 = plt.subplot(2, 2, 4)
    ax4.set_title("Mean winter PMB", fontsize=24)
    grouped_ids_winter.sort_values(by="YEAR", inplace=True)
    plotMeanPred(grouped_ids_winter, ax4)

    plt.tight_layout()


def process_glacier_grids(
    cfg,
    glacier_list,
    periods_per_glacier,
    all_columns,
    loaded_model,
    path_glacier_grid_glamos,
    path_save_glw,
    path_xr_grids,
):
    """
    Process distributed MB grids for a list of glaciers using pre-trained models.

    Parameters
    ----------
    cfg : object
        Configuration object with dataPath attribute.
    glacier_list : list of str
        List of glacier names to process.
    periods_per_glacier : dict
        Dictionary mapping glacier names to periods (years) for processing.
    all_columns : list of str
        List of required column names in glacier grid files.
    loaded_model : object
        Pre-trained model to use for prediction.
    path_glacier_grid_glamos : str
        Relative path to glacier grids within cfg.dataPath.
    emptyfolder : function
        Function to empty a folder.
    path_save_glw : str
        Path where results will be saved.
    path_xr_grids : str
        Path to xr_masked_grids.
    """
    # Ensure save path exists
    os.makedirs(path_save_glw, exist_ok=True)

    emptyfolder(path_save_glw)

    for glacier_name in glacier_list:
        glacier_path = os.path.join(
            cfg.dataPath, path_glacier_grid_glamos, glacier_name
        )

        if not os.path.exists(glacier_path):
            print(f"Folder not found for {glacier_name}, skipping...")
            continue

        glacier_files = sorted(
            [f for f in os.listdir(glacier_path) if glacier_name in f]
        )

        geodetic_range = range(
            np.min(periods_per_glacier[glacier_name]),
            np.max(periods_per_glacier["aletsch"]) + 1,
        )

        years = [
            int(file_name.split("_")[2].split(".")[0]) for file_name in glacier_files
        ]
        years = [y for y in years if y in geodetic_range]

        print(f"Processing {glacier_name} ({len(years)} files)")

        for year in tqdm(years, desc=f"Processing {glacier_name}", leave=False):
            file_name = f"{glacier_name}_grid_{year}.parquet"
            file_path = os.path.join(
                cfg.dataPath, path_glacier_grid_glamos, glacier_name, file_name
            )

            df_grid_monthly = pd.read_parquet(file_path)
            df_grid_monthly.drop_duplicates(inplace=True)

            # Keep only necessary columns
            df_grid_monthly = df_grid_monthly[
                [col for col in all_columns if col in df_grid_monthly.columns]
            ]
            df_grid_monthly = df_grid_monthly.dropna()

            # Create geodata object
            geoData = mbm.geodata.GeoData(df_grid_monthly)

            # Compute and save gridded MB
            path_glacier_dem = os.path.join(
                path_xr_grids, f"{glacier_name}_{year}.zarr"
            )

            geoData.gridded_MB_pred(
                df_grid_monthly,
                loaded_model,
                glacier_name,
                year,
                all_columns,
                path_glacier_dem,
                path_save_glw,
                save_monthly_pred=True,
                type_model="NN",
            )


def retrieve_best_params(path, sort_values="valid_loss"):
    # Open grid_search results
    gs_results = pd.read_csv(path).sort_values(by=sort_values, ascending=True)

    # Take best row
    best_params = gs_results.iloc[0].to_dict()

    # Clean it up into a proper dict
    params = {}

    for key, value in best_params.items():
        if key in ["valid_loss", "train_loss", "test_rmse", "status", "error"]:
            continue  # skip these

        if isinstance(value, str):
            # Convert optimizer string to actual torch class
            if "torch.optim" in value:
                # e.g. "<class 'torch.optim.adamw.AdamW'>" → torch.optim.AdamW
                cls_name = value.split("'")[1].split(".")[-1]
                params[key] = getattr(torch.optim, cls_name)
            else:
                # Convert string representations of lists, bools, numbers
                try:
                    params[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    params[key] = value
        else:
            params[key] = value

    return params


def plot_topk_param_distributions(
    log: Union[str, Path, pd.DataFrame],
    k: int = 10,
    metric: str = "valid_loss",
    minimize: bool = True,
    num_params: Optional[List[str]] = None,
    cat_params: Optional[List[str]] = None,
):
    """
    Plot distributions of parameters for the top-k models in a grid-search log.

    Args:
        log: Either a path to the CSV log file or a DataFrame of search results.
        k: Number of top rows to select.
        metric: Column name to rank by.
        minimize: If True, lower metric is better; if False, higher is better.
        num_params: List of numeric parameter names to histogram.
        cat_params: List of categorical/boolean parameter names to countplot.
    """
    # load CSV if a path is passed
    if isinstance(log, (str, Path)):
        df_log = pd.read_csv(log)
    else:
        df_log = log.copy()

    if metric == "avg_test_loss":
        if {"test_rmse_a", "test_rmse_w"}.issubset(df_log.columns):
            df_log["avg_test_loss"] = (
                df_log["test_rmse_a"] + df_log["test_rmse_w"]
            ) / 2

    if metric not in df_log.columns:
        raise ValueError(
            f"Metric '{metric}' not found in columns: {list(df_log.columns)}"
        )

    # rank models
    df_sorted = df_log.sort_values(metric, ascending=minimize)
    topk = df_sorted.head(k).copy()

    print(f"[Top {k}] average {metric}: {topk[metric].mean():.4f}")

    # defaults
    if num_params is None:
        num_params = [
            "hidden_size",
            "num_layers",
            "dropout",
            "static_layers",
            "static_hidden",
            "lr",
            "weight_decay",
        ]
    # if cat_params is None:
    #     cat_params = ["bidirectional", "loss_name"]

    # numeric distributions
    numeric_avail = [c for c in num_params if c in topk.columns]
    if numeric_avail:
        topk[numeric_avail].hist(figsize=(12, 10), bins=10)
        plt.suptitle(f"Top {k}: Numerical Parameter Distributions")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    # # categorical/boolean distributions
    # for col in cat_params:
    #     if col in topk.columns:
    #         plt.figure(figsize=(5, 4))
    #         sns.countplot(x=col, data=topk)
    #         plt.title(f"Top {k}: {col} distribution")
    #         plt.show()


def prepare_monthly_long_df(
    df_lstm, df_nn, df_xgb, df_glamos_w, df_glamos_a, month_order=None
):
    """
    Convert LSTM, NN, XGB, and GLAMOS glacier–month DataFrames into a long-format DataFrame
    for plotting.

    GLAMOS winter contains April data, GLAMOS annual contains September data.

    Parameters
    ----------
    df_lstm : pd.DataFrame
    df_nn   : pd.DataFrame
    df_xgb  : pd.DataFrame
    df_glamos_w : pd.DataFrame   # ['glacier', 'year', 'apr']
    df_glamos_a : pd.DataFrame   # ['glacier', 'year', 'sep']
    month_order : list, optional

    Returns
    -------
    pd.DataFrame with columns:
    ['glacier', 'year', 'Month', 'mb_nn', 'mb_lstm', 'mb_xgb', 'mb_glamos']
    """

    if month_order is None:
        month_order = [
            "oct",
            "nov",
            "dec",
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
        ]

    common_cols = ["glacier", "year"]

    # Keep only glacier–year pairs present in GLAMOS
    valid_pairs = pd.concat(
        [df_glamos_w[common_cols], df_glamos_a[common_cols]], ignore_index=True
    ).drop_duplicates()

    df_lstm = df_lstm.merge(valid_pairs, on=common_cols, how="inner")
    df_nn = df_nn.merge(valid_pairs, on=common_cols, how="inner")
    df_xgb = df_xgb.merge(valid_pairs, on=common_cols, how="inner")

    # --- arrays to populate long DF ---
    array_nn, array_lstm, array_xgb = [], [], []
    months, glaciers, years = [], [], []

    for col in month_order:
        array_nn.append(df_nn[col].values)
        array_lstm.append(df_lstm[col].values)
        array_xgb.append(df_xgb[col].values)

        months.append(np.tile(col, len(df_nn)))
        glaciers.append(df_nn["glacier"].values)
        years.append(df_nn["year"].values)

    # Build long-format DataFrame
    df_long = pd.DataFrame(
        {
            "glacier": np.concatenate(glaciers),
            "year": np.concatenate(years),
            "Month": np.concatenate(months),
            "mb_nn": np.concatenate(array_nn),
            "mb_lstm": np.concatenate(array_lstm),
            "mb_xgb": np.concatenate(array_xgb),
            "mb_glamos": np.nan,
        }
    )

    # ---- Inject GLAMOS observations ----
    # April → winter balance
    if "apr" in df_glamos_w.columns:
        mask = df_long["Month"] == "apr"
        df_apr = df_long.loc[mask, ["glacier", "year"]].merge(
            df_glamos_w[["glacier", "year", "apr"]],
            on=["glacier", "year"],
            how="left",
        )
        df_long.loc[mask, "mb_glamos"] = df_apr["apr"].values

    # September → annual balance
    if "sep" in df_glamos_a.columns:
        mask = df_long["Month"] == "sep"
        df_sep = df_long.loc[mask, ["glacier", "year"]].merge(
            df_glamos_a[["glacier", "year", "sep"]],
            on=["glacier", "year"],
            how="left",
        )
        df_long.loc[mask, "mb_glamos"] = df_sep["sep"].values

    # ---- Order categorical month column ----
    df_long["Month"] = df_long["Month"].astype(
        CategoricalDtype(month_order, ordered=True)
    )

    return df_long


def build_combined_LSTM_dataset_2(
    df_loss,
    df_full,
    monthly_cols,
    static_cols,
    months_head_pad,
    months_tail_pad,
    normalize_target=True,
    expect_target=True,
):
    # Clean copies
    df_loss = df_loss.copy()
    df_full = df_full.copy()
    df_loss["PERIOD"] = df_loss["PERIOD"].str.lower().str.strip()
    df_full["PERIOD"] = df_full["PERIOD"].str.lower().str.strip()

    # --------------------------------------
    # STEP 1 — Remove POINT_BALANCE from df_full
    # --------------------------------------
    df_full_clean = df_full.drop(columns=["POINT_BALANCE", "y"], errors="ignore")

    # --------------------------------------
    # STEP 2 — Keep only the POINT_BALANCE information from df_loss
    # --------------------------------------
    df_loss_reduced = df_loss[
        ["GLACIER", "YEAR", "ID", "PERIOD", "MONTHS", "POINT_BALANCE"]
    ].copy()

    # --------------------------------------
    # STEP 3 — Merge
    # padded months will have POINT_BALANCE = NaN
    # --------------------------------------
    df_combined = df_full_clean.merge(
        df_loss_reduced, on=["GLACIER", "YEAR", "ID", "PERIOD", "MONTHS"], how="left"
    )

    # --------------------------------------
    # STEP 4 — Build dataset
    # --------------------------------------
    ds = mbm.data_processing.MBSequenceDataset_2.from_dataframe(
        df=df_combined,
        monthly_cols=monthly_cols,
        static_cols=static_cols,
        months_head_pad=months_head_pad,
        months_tail_pad=months_tail_pad,
        expect_target=expect_target,
        normalize_target=normalize_target,
    )

    return ds
