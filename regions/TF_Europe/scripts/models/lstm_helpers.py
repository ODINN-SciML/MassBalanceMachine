from pathlib import Path
from typing import Any, Dict, Union
import ast
import os
import re
import pandas as pd
import xarray as xr
from tqdm.notebook import tqdm
import copy
import massbalancemachine as mbm
import torch
from datetime import datetime

from regions.TF_Europe.scripts.plotting import *


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


# --------------------------- MULTI REGION HANDLING ---------------------------


def iter_dataset_keys_from_config(RGI_REGIONS: dict):
    """
    Yields dataset keys like:
      '06_ISL', '07_SJM', '08_NOR', '11_CH', ...
    """
    for rid, spec in RGI_REGIONS.items():
        rid2 = str(rid).zfill(2)
        sub_codes = spec.get("subregions_codes", []) or []

        if sub_codes:
            for code in sub_codes:
                yield f"{rid2}_{code.upper()}"
        else:
            yield f"{rid2}_{spec['code'].upper()}"


def build_lstm_params_by_key(
    default_params: dict,
    log_path_gs_results: dict,  # keyed by CODE only
    RGI_REGIONS: dict,
    select_by: str = "avg_test_loss",
):
    """
    Returns dict:
        'RID_CODE' -> params dict

    If a grid-search log exists for the CODE,
    best params override defaults.
    """

    params_by_key = {}

    for key in sorted(iter_dataset_keys_from_config(RGI_REGIONS)):
        rid, code = key.split("_", 1)

        params = copy.deepcopy(default_params)
        log_path = log_path_gs_results.get(code)

        if log_path and os.path.exists(log_path):
            print(f"Loading tuned params for {key} (code={code})")
            best_params = get_best_params_for_lstm(log_path, select_by=select_by)
            params.update(best_params)
        else:
            print(f"No grid-search log for {key}. Using defaults.")

        params_by_key[key] = params

    return params_by_key


def train_or_load_one_within_region(
    cfg,
    key: str,  # e.g. "08_NOR"
    lstm_assets: dict,  # dict with ds_train/ds_test/train_idx/val_idx for this key
    best_params: dict,
    device,
    models_dir="models",
    prefix="lstm_within",
    train_flag=True,  # if False: only load (must exist)
    force_retrain=False,  # if True: retrain even if checkpoint exists
    epochs=150,
    batch_size_train=64,
    batch_size_val=128,
    batch_size_test=128,
    verbose=True,
):
    current_date = datetime.now().strftime("%Y-%m-%d")
    out_dir = os.path.join(models_dir)
    os.makedirs(out_dir, exist_ok=True)
    model_filename = os.path.join(out_dir, f"{prefix}_{key}_{current_date}.pt")

    # --- Build model + loss fn ---
    model = mbm.models.LSTM_MB.build_model_from_params(
        cfg, best_params, device, verbose=verbose
    )
    loss_fn = mbm.models.LSTM_MB.resolve_loss_fn(best_params)

    # --- If not training: just load ---
    if (not train_flag) and os.path.exists(model_filename):
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state)
        return model, model_filename, None

    # --- If training but we can reuse existing checkpoint ---
    if train_flag and (not force_retrain) and os.path.exists(model_filename):
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state)
        return model, model_filename, None

    if not train_flag and (not os.path.exists(model_filename)):
        raise FileNotFoundError(f"No checkpoint found for {key}: {model_filename}")

    # --- loaders (fit scalers on TRAIN) ---
    mbm.utils.seed_all(cfg.seed)

    ds_train = lstm_assets["ds_train"]
    ds_test = lstm_assets["ds_test"]
    train_idx = lstm_assets["train_idx"]
    val_idx = lstm_assets["val_idx"]

    ds_train_copy = mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
        ds_train
    )
    ds_test_copy = mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
        ds_test
    )

    train_dl, val_dl = ds_train_copy.make_loaders(
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size_train=batch_size_train,
        batch_size_val=batch_size_val,
        seed=cfg.seed,
        fit_and_transform=True,
        shuffle_train=True,
        use_weighted_sampler=True,
        verbose=verbose,
    )

    test_dl = mbm.data_processing.MBSequenceDataset.make_test_loader(
        ds_test_copy, ds_train_copy, batch_size=batch_size_test, seed=cfg.seed
    )

    # fresh checkpoint
    if os.path.exists(model_filename):
        os.remove(model_filename)
        if verbose:
            print(f"Deleted existing model file: {model_filename}")

    history, best_val, best_state = model.train_loop(
        device=device,
        train_dl=train_dl,
        val_dl=val_dl,
        epochs=epochs,
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        clip_val=1,
        # scheduler
        sched_factor=0.5,
        sched_patience=6,
        sched_threshold=0.01,
        sched_threshold_mode="rel",
        sched_cooldown=1,
        sched_min_lr=1e-6,
        # early stopping
        es_patience=15,
        es_min_delta=1e-4,
        # logging
        log_every=5,
        verbose=verbose,
        # checkpoint
        save_best_path=model_filename,
        loss_fn=loss_fn,
    )

    if verbose:
        plot_history_lstm(history)

    # Load best checkpoint
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state)

    return (
        model,
        model_filename,
        {
            "history": history,
            "best_val": best_val,
            "test_dl": test_dl,
            "ds_test": ds_test_copy,
        },
    )


# Run for all dataset keys (train subset, load others)
def train_within_region_models_all(
    cfg,
    lstm_assets_by_key: dict,  # e.g. lstm_assets["08_NOR"] -> {...}
    params_by_key: dict,  # e.g. params_by_key["08_NOR"] -> {...}
    device,
    train_keys=None,  # e.g. ["08_NOR"] to retrain only Norway
    force_retrain=False,
    models_dir="models",
    prefix="lstm_within",
    epochs=150,
):
    models = {}
    infos = {}

    train_keys_set = set(train_keys) if train_keys else None

    for key in sorted(lstm_assets_by_key.keys()):
        best_params = params_by_key[key]

        # train only selected keys; others will load if checkpoint exists
        train_flag = True
        if train_keys_set is not None:
            train_flag = key in train_keys_set

        print(f"\n=== {key} === train_flag={train_flag}, force_retrain={force_retrain}")

        model, path, info = train_or_load_one_within_region(
            cfg=cfg,
            key=key,
            lstm_assets=lstm_assets_by_key[key],
            best_params=best_params,
            device=device,
            models_dir=models_dir,
            prefix=prefix,
            train_flag=train_flag,
            force_retrain=force_retrain if train_flag else False,
            epochs=epochs,
        )

        models[key] = model
        infos[key] = {"model_path": path, **(info or {})}

    return models, infos


def train_crossregional_models_all(
    cfg,
    lstm_assets_by_key: dict,  # outputs_xreg["FR"] -> {"ds_train","ds_test","train_idx","val_idx",...}
    default_params: dict,
    device,
    train_keys=None,  # e.g. ["FR"] to retrain only France target
    force_retrain=False,
    models_dir="models",
    prefix="lstm_xreg_CH_to",
    epochs=150,
):
    models = {}
    infos = {}

    train_keys_set = set(train_keys) if train_keys else None

    for key in sorted(lstm_assets_by_key.keys()):
        assets = lstm_assets_by_key[key]

        # skip empty regions (if you kept those)
        if assets is None or assets.get("ds_test", None) is None:
            print(f"\n=== {key} === skipped (no test dataset)")
            models[key] = None
            infos[key] = {"model_path": None, "note": "No test dataset / skipped"}
            continue

        # train only selected keys; others load if checkpoint exists
        train_flag = True
        if train_keys_set is not None:
            train_flag = key in train_keys_set

        print(
            f"\n=== CH -> {key} === train_flag={train_flag}, force_retrain={force_retrain}"
        )

        # IMPORTANT: key becomes target-specific so model filenames are unique
        model_key = key

        model, path, info = train_or_load_one_within_region(
            cfg=cfg,
            key=model_key,
            lstm_assets=assets,
            best_params=default_params,
            device=device,
            models_dir=models_dir,
            prefix=prefix,
            train_flag=train_flag,
            force_retrain=force_retrain if train_flag else False,
            epochs=epochs,
        )

        models[key] = model
        infos[key] = {"model_path": path, **(info or {})}

    return models, infos
