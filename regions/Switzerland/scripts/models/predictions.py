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

import massbalancemachine as mbm
from regions.Switzerland.scripts.config_CH import *
from regions.Switzerland.scripts.utils import *


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


def prepare_monthly_long_df(
    df_lstm, df_nn, df_xgb, df_glamos_w, df_glamos_a, month_order=None
):
    """
    Convert modelled and observed glacier mass-balance tables from wide
    (one column per month) to a single long-format DataFrame suitable for
    plotting and grouped analyses.

    The function takes monthly model predictions (NN, LSTM, XGB) provided in
    wide format and stacks them into a long table with one row per
    glacier–year–month. GLAMOS observations are then injected for the
    corresponding months:
      - winter GLAMOS is stored in the 'apr' column (April),
      - annual GLAMOS is stored in the 'sep' column (September).

    Only glacier–year pairs that exist in GLAMOS (winter and/or annual) are
    retained; the model DataFrames are filtered to these valid pairs.

    Parameters
    ----------
    df_lstm : pandas.DataFrame
        Wide-format LSTM predictions with at least the columns
        ['glacier', 'year'] plus one column per month in ``month_order``
        (e.g. 'oct', 'nov', ..., 'sep').
    df_nn : pandas.DataFrame
        Wide-format NN predictions with the same structure/columns as ``df_lstm``.
    df_xgb : pandas.DataFrame
        Wide-format XGB predictions with the same structure/columns as ``df_lstm``.
    df_glamos_w : pandas.DataFrame
        GLAMOS winter mass-balance observations with columns
        ['glacier', 'year', 'apr'] (April values).
    df_glamos_a : pandas.DataFrame
        GLAMOS annual mass-balance observations with columns
        ['glacier', 'year', 'sep'] (September values).
    month_order : list of str, optional
        Ordered list of month labels to use when stacking. If None, defaults to
        hydrological-year order:
        ['oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
         'aug', 'sep'].

    Returns
    -------
    df_long : pandas.DataFrame
        Long-format DataFrame with one row per glacier–year–month and columns:
            - 'glacier' (str)
            - 'year' (int)
            - 'Month' (pandas.Categorical): ordered by ``month_order``
            - 'mb_nn' (float): NN prediction for the month
            - 'mb_lstm' (float): LSTM prediction for the month
            - 'mb_xgb' (float): XGB prediction for the month
            - 'mb_glamos' (float): GLAMOS observation (only populated for
              'apr' from winter and 'sep' from annual; NaN otherwise)

    Notes
    -----
    - Model inputs are inner-joined with the set of GLAMOS glacier–year pairs,
      so any modelled glacier–year not present in GLAMOS is dropped.
    - GLAMOS values are merged in month-specific steps:
        * Month == 'apr' gets values from ``df_glamos_w['apr']``
        * Month == 'sep' gets values from ``df_glamos_a['sep']``
      All other months remain NaN in ``mb_glamos``.
    - The function assumes the model DataFrames contain all months listed in
      ``month_order`` as columns. If a month column is missing, a KeyError will
      be raised when indexing (e.g. ``df_nn[col]``).

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
