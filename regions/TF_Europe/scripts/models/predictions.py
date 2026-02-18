import re
import logging
import xarray as xr
from tqdm.auto import tqdm
import pandas as pd
from pandas.api.types import CategoricalDtype

from sklearn.model_selection import (
    GroupKFold,
    KFold,
    train_test_split,
    GroupShuffleSplit,
)
import numpy as np

import massbalancemachine as mbm
from regions.TF_Europe.scripts.config_TF_Europe import *
from regions.TF_Europe.scripts.utils import *
from regions.TF_Europe.scripts.geodata import *
from regions.TF_Europe.scripts.plotting import *


def get_df_aggregate_pred(test_set, y_pred_agg, all_columns):
    """
    Aggregate point-level predictions to measurement-level summaries.

    This function takes model predictions produced at the individual-row level
    (typically monthly or point-based) and aggregates them to a higher level
    using the unique measurement identifier "ID". The aggregation is performed
    to match the temporal resolution of the original observations (e.g.,
    annual or winter balances).

    The function:
      - Extracts relevant columns from the test feature DataFrame.
      - Adds the true target values.
      - Groups all rows by measurement ID.
      - Averages the target values within each group.
      - Attaches aggregated model predictions.
      - Preserves metadata such as YEAR, PERIOD, and GLACIER.

    Parameters
    ----------
    test_set : dict
        Dictionary describing the test set, as returned by `get_CV_splits`.
        Must contain at least:
        - "df_X": pandas.DataFrame with feature data
        - "y": array-like of true target values corresponding to rows of df_X

    y_pred_agg : array-like
        Aggregated model predictions, one value per unique measurement ID.
        The length of this array must match the number of unique IDs
        in `test_set["df_X"]`.

    all_columns : list of str
        List of columns from `df_X` to retain for aggregation
        (e.g., metadata such as YEAR, PERIOD, POINT_ID).

    Returns
    -------
    pandas.DataFrame
        A DataFrame indexed by measurement ID with the following columns:

        - target : float
            Mean observed target value for each ID.

        - pred : float
            Aggregated model prediction corresponding to each ID.

        - YEAR : int
            Year associated with the measurement (taken as the first entry
            within each group).

        - POINT_ID : str
            Original point identifier.

        - PERIOD : str
            Temporal period of the observation (e.g., "annual" or "winter").

        - GLACIER : str
            Glacier name inferred from the POINT_ID prefix.
    """
    # Aggregate predictions to annual or winter:
    df_pred = test_set["df_X"][all_columns].copy()
    df_pred["target"] = test_set["y"]
    grouped_ids = df_pred.groupby("ID").agg(
        {"target": "mean", "YEAR": "first", "POINT_ID": "first"}
    )
    grouped_ids["pred"] = y_pred_agg
    grouped_ids["PERIOD"] = (
        test_set["df_X"][all_columns].groupby("ID")["PERIOD"].first()
    )
    grouped_ids["GLACIER"] = grouped_ids["POINT_ID"].apply(lambda x: x.split("_")[0])

    return grouped_ids


def compute_seasonal_scores(df, target_col="target", pred_col="pred"):
    """
    Computes regression scores separately for annual and winter data.

    If one season is missing (0 samples), returns NaNs for that season instead
    of crashing.

    Returns:
    - scores_annual: dict of metrics for annual data.
    - scores_winter: dict of metrics for winter data.
    """

    scores = {}

    for season in ["annual", "winter"]:
        df_season = df[df["PERIOD"] == season]

        if len(df_season) == 0:
            # Keep same keys as mbm.metrics.scores (plus n)
            scores[season] = {
                "mse": np.nan,
                "rmse": np.nan,
                "mae": np.nan,
                "R2": np.nan,
                "Bias": np.nan,
                "n": 0,
            }
            continue

        y_true = df_season[target_col]
        y_pred = df_season[pred_col]

        scores_season = mbm.metrics.scores(y_true, y_pred)

        # Rename to match existing usage
        scores_season["R2"] = scores_season.pop("r2")
        scores_season["Bias"] = scores_season.pop("bias")
        scores_season["n"] = int(len(df_season))

        scores[season] = scores_season

    return scores["annual"], scores["winter"]


def evaluate_NN_and_group_predictions(
    custom_NN_model,
    df_X_subset,
    y,
    months_head_pad,
    months_tail_pad,
):
    """
    Evaluate a trained neural network model by computing grouped predictions
    and metrics over predefined temporal and spatial groupings.

    This function is a thin wrapper around
    ``custom_NN_model.evaluate_group_pred``. It evaluates model predictions
    grouped by period, glacier, and year, while accounting for padded months
    at the beginning and end of the time series.

    Parameters
    ----------
    custom_NN_model : CustomNeuralNetRegressor
        Trained neural network model implementing the
        ``evaluate_group_pred`` method.
    df_X_subset : pandas.DataFrame or array-like
        Input feature data used for evaluation. Must be aligned with ``y``
        and contain the metadata required for grouping (PERIOD, GLACIER, YEAR).
    y : array-like
        Target values corresponding to ``df_X_subset``.
    months_head_pad : int
        Number of months to ignore (pad) at the beginning of each time series
        when computing grouped predictions.
    months_tail_pad : int
        Number of months to ignore (pad) at the end of each time series
        when computing grouped predictions.

    Returns
    -------
    Any
        Output of ``custom_NN_model.evaluate_group_pred``, typically containing
        grouped predictions and/or aggregated evaluation metrics.
    """
    return custom_NN_model.evaluate_group_pred(
        df_X_subset,
        y,
        months_head_pad,
        months_tail_pad,
        group_by=["PERIOD", "GLACIER", "YEAR"],
    )


def load_glwd_nn_predictions(PATH_PREDICTIONS_NN, hydro_months):
    """
    Load gridded neural network (NN) glacier mass-balance predictions stored
    as Zarr files and aggregate them into a single pandas DataFrame.

    The function iterates over glacier-specific subdirectories and reads
    monthly Zarr datasets following the naming convention:

        <glacier_name>_<year>_<month>.zarr

    For each glacier and hydrological year, all available monthly predictions
    are extracted at grid-cell level, filtered to valid (non-NaN) prediction
    pixels, and combined into a tabular format. Elevation values corresponding
    to the predicted pixels are also included.

    Parameters
    ----------
    PATH_PREDICTIONS_NN : str
        Path to the directory containing one subdirectory per glacier.
        Each glacier subdirectory is expected to contain Zarr files with
        neural network predictions and auxiliary variables.
    hydro_months : list of str
        Ordered list of hydrological month identifiers (e.g.
        ['oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr', 'may', 'jun',
         'jul', 'aug', 'sep']). Only months present on disk are loaded.

    Returns
    -------
    df_months_NN : pandas.DataFrame
        Long-format DataFrame containing neural network predictions for all
        glaciers and years. Each row corresponds to a glacier grid cell and
        year, with columns:
            - one column per hydrological month (predicted values),
            - 'year' (int),
            - 'glacier' (str),
            - 'elevation' (float, meters).

    Notes
    -----
    - Only pixels with valid (non-NaN) values in ``pred_masked`` are retained.
    - Elevation values are taken from ``masked_elev`` and aligned to the same
      valid prediction pixels.
    - Years are inferred automatically from the filenames found in each
      glacier directory.
    - Missing months for a given year are silently skipped.
    - The function assumes that all monthly Zarr files for a given
      glacier-year share the same spatial mask and pixel ordering.

    Raises
    ------
    ValueError
        If no valid glacier data are found in the provided directory structure.
    FileNotFoundError
        If ``PATH_PREDICTIONS_NN`` does not exist.

    """
    glaciers = os.listdir(PATH_PREDICTIONS_NN)

    # Initialize final storage for all glacier data
    all_glacier_data = []

    # Loop over glaciers
    for glacier_name in tqdm(glaciers):
        glacier_path = os.path.join(PATH_PREDICTIONS_NN, glacier_name)
        if not os.path.isdir(glacier_path):
            continue  # skip non-directories

        # Regex pattern adapted for current glacier name
        pattern = re.compile(rf"{glacier_name}_(\d{{4}})_[a-z]{{3}}\.zarr")

        # Extract available years
        years = set()
        for fname in os.listdir(glacier_path):
            match = pattern.match(fname)
            if match:
                years.add(int(match.group(1)))
        years = sorted(years)

        # Collect all year-month data
        all_years_data = []
        for year in years:
            monthly_data = {}
            for month in hydro_months:
                zarr_path = os.path.join(
                    glacier_path, f"{glacier_name}_{year}_{month}.zarr"
                )
                if not os.path.exists(zarr_path):
                    continue

                ds = xr.open_dataset(zarr_path)
                df = (
                    ds.pred_masked.to_dataframe().drop(["x", "y"], axis=1).reset_index()
                )
                df_pred_months = df[df.pred_masked.notna()]

                df_el = (
                    ds.masked_elev.to_dataframe().drop(["x", "y"], axis=1).reset_index()
                )
                df_elv_months = df_el[df.pred_masked.notna()]

                df_pred_months["elevation"] = df_elv_months.masked_elev.values

                monthly_data[month] = df_pred_months.pred_masked.values

            if monthly_data:
                df_months = pd.DataFrame(monthly_data)
                df_months["year"] = year
                df_months["glacier"] = glacier_name  # add glacier name
                df_months["elevation"] = df_pred_months.elevation.values
                all_years_data.append(df_months)

        # Concatenate this glacier's data
        if all_years_data:
            df_glacier = pd.concat(all_years_data, axis=0, ignore_index=True)
            all_glacier_data.append(df_glacier)

    # Final full DataFrame for all glaciers
    df_months_NN = pd.concat(all_glacier_data, axis=0, ignore_index=True)
    return df_months_NN


def load_glwd_lstm_predictions(PATH_PREDICTIONS_LSTM, hydro_months):
    """
    Load gridded LSTM glacier mass-balance predictions stored as Zarr files
    and aggregate them into a single pandas DataFrame.

    The function iterates over glacier-specific subdirectories and reads
    monthly Zarr datasets following the naming convention:

        <glacier_name>_<year>_<month>.zarr

    For each glacier and hydrological year, all available monthly predictions
    are extracted at grid-cell level, filtered to valid (non-NaN) prediction
    pixels, and combined into a tabular format. Elevation values corresponding
    to the predicted pixels are also included.

    Parameters
    ----------
    PATH_PREDICTIONS_LSTM : str
        Path to the directory containing one subdirectory per glacier.
        Each glacier subdirectory is expected to contain Zarr files with
        LSTM predictions and auxiliary variables.
    hydro_months : list of str
        Ordered list of hydrological month identifiers (e.g.
        ['oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep']).
        Only months present on disk are loaded.

    Returns
    -------
    df_months_LSTM : pandas.DataFrame
        Long-format DataFrame containing LSTM predictions for all glaciers
        and years. Each row corresponds to a glacier grid cell and year, with
        columns:
            - one column per hydrological month (predicted values),
            - 'year' (int),
            - 'glacier' (str),
            - 'elevation' (float, meters).

    Notes
    -----
    - Only pixels with valid (non-NaN) values in ``pred_masked`` are retained.
    - Elevation values are taken from ``masked_elev`` and aligned to the same
      valid prediction pixels.
    - Years are inferred automatically from the filenames found in each
      glacier directory.
    - Missing months for a given year are silently skipped.
    - The function assumes that all monthly Zarr files for a given glacier-year
      share the same spatial mask and elevation ordering.

    Raises
    ------
    ValueError
        If no valid glacier data are found in the provided directory structure.
    FileNotFoundError
        If ``PATH_PREDICTIONS_LSTM`` does not exist.

    """
    glaciers = os.listdir(PATH_PREDICTIONS_LSTM)

    # Initialize final storage for all glacier data
    all_glacier_data = []

    # Loop over glaciers
    for glacier_name in tqdm(glaciers):
        glacier_path = os.path.join(PATH_PREDICTIONS_LSTM, glacier_name)
        if not os.path.isdir(glacier_path):
            continue  # skip non-directories

        # Regex pattern adapted for current glacier name
        pattern = re.compile(rf"{glacier_name}_(\d{{4}})_[a-z]{{3}}\.zarr")

        # Extract available years
        years = set()
        for fname in os.listdir(glacier_path):
            match = pattern.match(fname)
            if match:
                years.add(int(match.group(1)))
        years = sorted(years)

        # Collect all year-month data
        all_years_data = []
        for year in years:
            monthly_data = {}
            for month in hydro_months:
                zarr_path = os.path.join(
                    glacier_path, f"{glacier_name}_{year}_{month}.zarr"
                )
                if not os.path.exists(zarr_path):
                    continue

                ds = xr.open_dataset(zarr_path)
                df = (
                    ds.pred_masked.to_dataframe().drop(["x", "y"], axis=1).reset_index()
                )
                df_pred_months = df[df.pred_masked.notna()]

                df_el = (
                    ds.masked_elev.to_dataframe().drop(["x", "y"], axis=1).reset_index()
                )
                df_elv_months = df_el[df.pred_masked.notna()]

                df_pred_months["elevation"] = df_elv_months.masked_elev.values

                monthly_data[month] = df_pred_months.pred_masked.values

            if monthly_data:
                df_months = pd.DataFrame(monthly_data)
                df_months["year"] = year
                df_months["glacier"] = glacier_name  # add glacier name
                df_months["elevation"] = df_pred_months.elevation.values
                all_years_data.append(df_months)

        # Concatenate this glacier's data
        if all_years_data:
            df_glacier = pd.concat(all_years_data, axis=0, ignore_index=True)
            all_glacier_data.append(df_glacier)

    # Final full DataFrame for all glaciers
    df_months_LSTM = pd.concat(all_glacier_data, axis=0, ignore_index=True)
    return df_months_LSTM


# --------------------------- MULTI REGION HANDLING ---------------------------


def make_test_loader_for_key(cfg, lstm_assets_for_key, batch_size=128):
    """
    Creates ds_train_copy (fit scalers on train) and a test_dl that uses train scalers.
    Returns (ds_train_copy, ds_test_copy, test_dl).
    """
    mbm.utils.seed_all(cfg.seed)

    ds_train = lstm_assets_for_key["ds_train"]
    ds_test = lstm_assets_for_key["ds_test"]
    train_idx = lstm_assets_for_key["train_idx"]
    val_idx = lstm_assets_for_key["val_idx"]

    ds_train_copy = mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
        ds_train
    )
    ds_test_copy = mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
        ds_test
    )

    # Fit scalers on TRAIN split (important!)
    _train_dl, _val_dl = ds_train_copy.make_loaders(
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size_train=64,
        batch_size_val=128,
        seed=cfg.seed,
        fit_and_transform=True,
        shuffle_train=True,
        use_weighted_sampler=True,
    )

    test_dl = mbm.data_processing.MBSequenceDataset.make_test_loader(
        ds_test_copy, ds_train_copy, batch_size=batch_size, seed=cfg.seed
    )
    return ds_train_copy, ds_test_copy, test_dl


def evaluate_one_model(
    cfg,
    model,
    device,
    lstm_assets_for_key,
    ax=None,
    ax_xlim=(-16, 9),
    ax_ylim=(-16, 9),
    title=None,
    legend_fontsize=16,
):
    """
    Evaluate a single model and optionally plot pred-vs-truth density on an axis.

    If `ax` is None, a new (fig, ax) is created and returned.
    If `ax` is provided, plotting happens into that axis and fig may be None.
    """
    ds_train_copy, ds_test_copy, test_dl = make_test_loader_for_key(
        cfg, lstm_assets_for_key
    )

    test_metrics, test_df_preds = model.evaluate_with_preds(
        device, test_dl, ds_test_copy
    )

    scores_annual, scores_winter = compute_seasonal_scores(
        test_df_preds, target_col="target", pred_col="pred"
    )

    out = {
        "RMSE_annual": float(test_metrics.get("RMSE_annual", scores_annual["rmse"])),
        "RMSE_winter": float(test_metrics.get("RMSE_winter", scores_winter["rmse"])),
        "R2_annual": float(scores_annual["R2"]),
        "R2_winter": float(scores_winter["R2"]),
        "Bias_annual": float(scores_annual["Bias"]),
        "Bias_winter": float(scores_winter["Bias"]),
        "n_preds": int(len(test_df_preds)),
        # useful if you added n in your safe scorer
        "n_annual": (
            int(scores_annual.get("n", np.nan))
            if isinstance(scores_annual, dict)
            else np.nan
        ),
        "n_winter": (
            int(scores_winter.get("n", np.nan))
            if isinstance(scores_winter, dict)
            else np.nan
        ),
    }

    # Plot
    created_fig = None
    if ax is None:
        created_fig = plt.figure(figsize=(15, 10))
        ax = plt.subplot(1, 1, 1)

    pred_vs_truth_density(
        ax,
        test_df_preds,
        scores_annual,
        add_legend=False,
        palette=[mbm.plots.COLOR_ANNUAL, mbm.plots.COLOR_WINTER],
        ax_xlim=ax_xlim,
        ax_ylim=ax_ylim,
    )

    # Legend text (handle NaNs nicely)
    def _fmt(x):
        return (
            "NA"
            if (x is None or (isinstance(x, float) and np.isnan(x)))
            else f"{x:.2f}"
        )

    legend_NN = "\n".join(
        [
            rf"$\mathrm{{RMSE_a}}={_fmt(scores_annual['rmse'])},\ \mathrm{{RMSE_w}}={_fmt(scores_winter['rmse'])}$",
            rf"$\mathrm{{R^2_a}}={_fmt(scores_annual['R2'])},\ \mathrm{{R^2_w}}={_fmt(scores_winter['R2'])}$",
            rf"$\mathrm{{Bias_a}}={_fmt(scores_annual['Bias'])},\ \mathrm{{Bias_w}}={_fmt(scores_winter['Bias'])}$",
        ]
    )

    ax.text(
        0.02,
        0.98,
        legend_NN,
        transform=ax.transAxes,
        va="top",
        fontsize=legend_fontsize,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    if title:
        ax.set_title(title, fontsize=20)

    return out, test_df_preds, created_fig, ax


def evaluate_all_models(
    cfg,
    models_by_key: dict,
    lstm_assets_by_key: dict,
    device,
    save_dir=None,  # e.g. "figures/eval"
    grid_shape=(2, 3),  # 2x3 for 6 regions
    grid_figsize=(20, 12),
    ax_xlim=(-16, 9),
    ax_ylim=(-16, 9),
):
    if save_dir:
        save_abs = os.path.join(save_dir)
        os.makedirs(save_abs, exist_ok=True)
    else:
        save_abs = None

    keys = sorted(models_by_key.keys())

    # --- combined grid ---
    nrows, ncols = grid_shape
    fig_grid, axes = plt.subplots(
        nrows, ncols, figsize=grid_figsize, sharex=True, sharey=True
    )
    axes = np.array(axes).reshape(-1)  # flat
    # If fewer/more keys than slots, we handle gracefully
    n_slots = len(axes)

    rows = []
    preds_by_key = {}
    figs_by_key = {}

    for i, key in enumerate(keys):
        model = models_by_key[key]
        print(f"\nEvaluating {key} ...")

        # --- individual fig ---
        metrics, df_preds, fig_ind, ax_ind = evaluate_one_model(
            cfg=cfg,
            model=model,
            device=device,
            lstm_assets_for_key=lstm_assets_by_key[key],
            ax=None,
            ax_xlim=ax_xlim,
            ax_ylim=ax_ylim,
            title=f"{key} – Pred vs Truth (Test)",
            legend_fontsize=14,
        )
        metrics["key"] = key
        rows.append(metrics)
        preds_by_key[key] = df_preds
        figs_by_key[key] = fig_ind

        if save_abs:
            out_png = os.path.join(save_abs, f"pred_vs_truth_{key}.png")
            fig_ind.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig_ind)  # prevents duplicate display in notebooks

        # --- grid subplot (if slot available) ---
        if i < n_slots:
            ax_grid = axes[i]
            evaluate_one_model(
                cfg=cfg,
                model=model,
                device=device,
                lstm_assets_for_key=lstm_assets_by_key[key],
                ax=ax_grid,
                ax_xlim=ax_xlim,
                ax_ylim=ax_ylim,
                title=key,
                legend_fontsize=15,
            )

    # turn off unused axes
    for j in range(len(keys), n_slots):
        axes[j].axis("off")

    for j in range(3):
        axes[j].set_xlabel("")
    for j in range(6):
        if j == 0 or j == 3:
            axes[j].set_ylabel("Modeled PMB [m w.e.]")
        else:
            axes[j].set_ylabel("")

    fig_grid.suptitle("Pred vs Truth (Test) — All Regions", fontsize=20)
    fig_grid.tight_layout()

    if save_abs:
        out_grid = os.path.join(save_abs, "pred_vs_truth_ALL_REGIONS_grid.png")
        fig_grid.savefig(out_grid, dpi=200, bbox_inches="tight")

    df_metrics = pd.DataFrame(rows).set_index("key").sort_index()

    return df_metrics, preds_by_key, figs_by_key, fig_grid
