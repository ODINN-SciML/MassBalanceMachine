import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
)


def scores(true, pred):
    """
    Compute a set of regression metrics between true and predicted values.

    Parameters
    ----------
    true : array-like
        Array of true (ground truth) target values.
    pred : array-like
        Array of predicted target values.

    Returns
    -------
    dict
        Dictionary containing:
            - "mse": Mean Squared Error
            - "rmse": Root Mean Squared Error
            - "mae": Mean Absolute Error
            - "pearson_corr": Pearson correlation coefficient between true and pred
            - "r2": R^2 (coefficient of determination) regression score
            - "bias": Mean model bias (mean of pred - true)
    """

    mse = mean_squared_error(true, pred)
    return {
        "mse": mse,
        "rmse": mse**0.5,
        "mae": mean_absolute_error(true, pred),
        "pearson_corr": np.corrcoef(true, pred)[0, 1],  # Pearson correlation
        "r2": r2_score(true, pred),  # R2 regression score
        "bias": np.mean(pred - true),  # Model bias
    }
