from pathlib import Path
from typing import Any, Dict, Union
import ast

import pandas as pd


def get_best_params_for_lstm(
    log_path: Union[str, Path],
    select_by: str = "valid_loss",
    minimize: bool = True,
) -> Dict[str, Any]:
    """
    Load a hyperparameter search log and return the best LSTM configuration.

    The function reads a CSV log produced by an LSTM hyperparameter search
    (grid/random search), selects the best run according to `select_by`,
    converts values to the correct Python types, and returns a parameter
    dictionary that matches the expected API of the LSTM model.

    Notes
    -----
    - `static_hidden` is returned as an `int` or `None` (NOT a list), to match
      the LSTM_MB model API.
    - If `select_by="avg_test_loss"`, the function requires `test_rmse_a` and
      `test_rmse_w` columns and computes:
        avg_test_loss = (test_rmse_a + test_rmse_w) / 2
    - The function prints a short summary of the selected best run.

    Parameters
    ----------
    log_path : str or pathlib.Path
        Path to the CSV log file containing one row per hyperparameter run.
    select_by : str, optional
        Column name used to rank runs. Common values are:
        - "valid_loss" (default)
        - "avg_test_loss" (computed from test_rmse_a and test_rmse_w)
        Any existing numeric column in the CSV may be used.
    minimize : bool, optional
        If True, the best run is the minimum of `select_by`.
        If False, the best run is the maximum of `select_by`.

    Returns
    -------
    dict
        Best hyperparameters in a dictionary with keys matching the LSTM model
        API, including:
        - Fm, Fs, hidden_size, num_layers, bidirectional, dropout
        - static_layers, static_hidden, static_dropout
        - lr, weight_decay
        - loss_name, loss_spec
        - two_heads, head_dropout

    Raises
    ------
    FileNotFoundError
        If `log_path` does not exist.
    ValueError
        If `select_by` is not a column in the log, or if `select_by="avg_test_loss"`
        but required columns are missing.
    """

    def _as_bool(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return bool(int(x))
        return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}

    def _as_opt_int(x):
        """
        Parse optional integer hyperparameters.
        Maps: None, NaN, "", "none", "nan", "0" -> None
        Maps: 128, 128.0, "128" -> 128
        """
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        s = str(x).strip().lower()
        if s in {"", "none", "nan", "0"}:
            return None
        try:
            return int(float(x))
        except Exception:
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
    if not log_path.exists():
        raise FileNotFoundError(f"Grid-search log file not found: {log_path}")

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

    # Print summary
    def _fmt(name):
        return f"{r[name]:.4f}" if name in r and pd.notna(r[name]) else "n/a"

    print(f"Best run {idx} by '{select_by}' (value: {_fmt(select_by)}):")
    print(
        f"  test_rmse_a: {_fmt('test_rmse_a')}  |  "
        f"test_rmse_w: {_fmt('test_rmse_w')}  |  "
        f"valid_loss: {_fmt('valid_loss')}"
    )

    # Core params (MATCH MODEL API)
    best_params: Dict[str, Any] = {
        "Fm": int(r["Fm"]),
        "Fs": int(r["Fs"]),
        "hidden_size": int(r["hidden_size"]),
        "num_layers": int(r["num_layers"]),
        "bidirectional": _as_bool(r["bidirectional"]),
        "dropout": float(r["dropout"]),
        "static_layers": int(r["static_layers"]),
        "static_hidden": _as_opt_int(r.get("static_hidden")),
        "static_dropout": _as_opt_float(r.get("static_dropout")),
        "lr": float(r["lr"]),
        "weight_decay": float(r["weight_decay"]),
        "loss_name": str(r.get("loss_name", "neutral")),
    }

    # two_heads & head_dropout
    if "two_heads" in r and pd.notna(r["two_heads"]):
        two_heads = _as_bool(r["two_heads"])
    elif "simple" in r and pd.notna(r["simple"]):
        two_heads = not _as_bool(r["simple"])
    else:
        two_heads = False

    head_dropout = _as_opt_float(r.get("head_dropout"))
    if head_dropout is None:
        head_dropout = 0.0

    best_params["two_heads"] = two_heads
    best_params["head_dropout"] = float(head_dropout)

    # loss_spec
    loss_spec_val = _as_opt_literal(r.get("loss_spec"))
    if best_params["loss_name"] == "weighted" and loss_spec_val is None:
        loss_spec_val = ("weighted", {})
    best_params["loss_spec"] = loss_spec_val

    return best_params
