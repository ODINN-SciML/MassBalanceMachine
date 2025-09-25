import matplotlib.pyplot as plt
import os
import seaborn as sns
from skorch.helper import SliceDataset
from datetime import datetime
import massbalancemachine as mbm
from tqdm.notebook import tqdm
import ast
from typing import Optional, Dict, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union

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


def evaluate_model_and_group_predictions(
    custom_NN_model,
    df_X_subset,
    y,
    cfg,
    months_head_pad,
    months_tail_pad,
):
    return custom_NN_model.evaluate_group_pred(
        df_X_subset,
        y,
        months_head_pad,
        months_tail_pad,
        group_by_col=["PERIOD", "GLACIER", "YEAR"],
    )


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


def get_best_params_for_lstm(
    log_path: Union[str, Path],
    select_by: str = "valid_loss",  # or "avg_test_loss"
    minimize: bool = True,
) -> Dict[str, Any]:
    """
    Return best hyperparameters with original names (incl. 'two_heads').
    - Parses types (bools, floats, ints, list-like strings).
    - If 'two_heads' missing but 'simple' present, sets two_heads = not simple.
      'simple' is NOT included in the output to keep original param names.
    - 'static_hidden' becomes list[int] or None (treat "0", 0, "", "none", NaN as None).
    - 'loss_spec' parsed to Python object or None. If loss_name == "weighted" and
      loss_spec is missing/empty, defaults to ("weighted", {}).
    """

    def _as_bool(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return bool(int(x))
        return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}

    def _as_opt_list(x):
        # map "0", 0, "", "none", NaN -> None
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        s = str(x).strip()
        if s.lower() in {"", "none", "nan", "0"}:
            return None
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple)):
                return [int(v) for v in val]
            # single int string like "128"
            if isinstance(val, int):
                return [val]
        except Exception:
            pass
        return None

    def _as_opt_float(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        s = str(x).strip().lower()
        if s in {"", "none", "nan"}:
            return None
        return float(x)

    def _as_opt_literal(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        s = str(x).strip()
        if s.lower() in {"", "none", "nan"}:
            return None
        try:
            return ast.literal_eval(s)
        except Exception:
            return s

    log_path = Path(log_path)
    df = pd.read_csv(log_path)

    if select_by == "avg_test_loss":
        if {"test_rmse_a", "test_rmse_w"}.issubset(df.columns):
            df["avg_test_loss"] = (df["test_rmse_a"] + df["test_rmse_w"]) / 2
        else:
            raise ValueError(
                "Need columns 'test_rmse_a' and 'test_rmse_w' to compute avg_test_loss."
            )

    if select_by not in df.columns:
        raise ValueError(
            f"Column '{select_by}' not found. Available: {list(df.columns)}"
        )

    idx = df[select_by].idxmin() if minimize else df[select_by].idxmax()
    r = df.loc[idx].to_dict()

    # print loss metrics of the best run
    print(f"Best run {idx} by '{select_by}' (value: {r[select_by]:.4f}):")
    print(f"Best run: test_rmse_a (value: {r['test_rmse_a']:.4f}):")
    print(f"Best run: test_rmse_w (value: {r['test_rmse_w']:.4f}):")
    print(f"Best run: valid_loss (value: {r['valid_loss']:.4f}):")

    best_params: Dict[str, Any] = {
        "Fm": int(r["Fm"]),
        "Fs": int(r["Fs"]),
        "hidden_size": int(r["hidden_size"]),
        "num_layers": int(r["num_layers"]),
        "bidirectional": _as_bool(r["bidirectional"]),
        "dropout": float(r["dropout"]),
        "static_layers": int(r["static_layers"]),
        "static_hidden": _as_opt_list(r.get("static_hidden")),
        "static_dropout": _as_opt_float(r.get("static_dropout")),
        "lr": float(r["lr"]),
        "weight_decay": float(r["weight_decay"]),
        "loss_name": str(r.get("loss_name", "neutral")),
        # loss_spec handled below
    }

    loss_spec_val = _as_opt_literal(r.get("loss_spec"))
    if best_params["loss_name"] == "weighted" and loss_spec_val is None:
        loss_spec_val = ("weighted", {})
    best_params["loss_spec"] = loss_spec_val

    return best_params


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


def plot_history_lstm(history):
    """
    Plot training and validation loss curves (and learning rate if available).

    Parameters
    ----------
    history : dict
        Dictionary with keys 'train_loss', 'val_loss', and optionally 'lr'.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot losses
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="tab:blue")
    ax1.plot(epochs, history["val_loss"], label="Validation Loss", color="tab:orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # If LR is present, plot on secondary axis
    if "lr" in history:
        ax2 = ax1.twinx()
        ax2.plot(
            epochs,
            history["lr"],
            label="Learning Rate",
            color="tab:green",
            linestyle="--",
        )
        ax2.set_ylabel("Learning Rate")
        ax2.legend(loc="upper center")

    plt.show()


def compute_seasonal_scores(df, target_col="target", pred_col="pred"):
    """
    Computes regression scores separately for annual and winter data.

    Parameters:
    - df: DataFrame with at least 'PERIOD', target_col, and pred_col columns.
    - target_col: name of the column with ground truth values.
    - pred_col: name of the column with predicted values.

    Returns:
    - scores_annual: dict of metrics for annual data.
    - scores_winter: dict of metrics for winter data.
    """

    scores = {}
    for season in ["annual", "winter"]:
        df_season = df[df["PERIOD"] == season]
        y_true = df_season[target_col]
        y_pred = df_season[pred_col]
        scores_season = mbm.metrics.scores(y_true, y_pred)
        # Rename to match with where this function is used
        scores_season["R2"] = scores_season.pop("r2")
        scores_season["Bias"] = scores_season.pop("bias")
        scores[season] = scores_season
    return scores["annual"], scores["winter"]
