import re
import logging

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
