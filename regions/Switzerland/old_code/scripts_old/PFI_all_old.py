# Parallel PFI for MBM LSTM (CPU, Linux)

# ------------------------------- Imports -------------------------------
import os
import sys
import random
import multiprocessing as mp
import collections
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Callable, Any, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm.auto import tqdm

# ------------------------------- determinism helpers -------------------------------


def _set_cpu_env_threads():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # deterministic CPU path
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# ------------------------------- globals for worker -------------------------------
# Filled by initializer once per process
_PFI_G = {
    "model": None,
    "device": None,
    "ds_train_copy": None,
    "MONTHLY_COLS": None,
    "STATIC_COLS": None,
    "months_head_pad": None,
    "months_tail_pad": None,
    "target_col": None,
    "baseline": None,
    "df_eval": None,
    "seed": None,
    "batch_size": None,
}


def _pfi_worker_init(
    cfg,
    custom_params: Dict[str, Any],
    model_filename: str,
    ds_train,
    train_idx,
    MONTHLY_COLS: List[str],
    STATIC_COLS: List[str],
    months_head_pad: int,
    months_tail_pad: int,
    df_eval: pd.DataFrame,
    target_col: str,
    seed: int,
    batch_size: int,
):
    """Initializer: quiet stdout, set threads, build scalers, load model, compute baseline once."""
    # local import avoids pickling a module from parent
    import massbalancemachine as mbm

    # silence worker prints
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    _set_cpu_env_threads()
    _set_seeds(seed)

    # Fit scalers on TRAIN only
    ds_train_copy = mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
        ds_train
    )
    ds_train_copy.fit_scalers(train_idx)

    # Load model on CPU
    device = torch.device("cpu")
    model = mbm.models.LSTM_MB.build_model_from_params(
        cfg, custom_params, device, verbose=False
    )
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Build eval ds/loader with targets
    ds_eval = mbm.data_processing.MBSequenceDataset.from_dataframe(
        df_eval,
        MONTHLY_COLS,
        STATIC_COLS,
        months_tail_pad=months_tail_pad,
        months_head_pad=months_head_pad,
        expect_target=True,
        show_progress=False,
    )
    dl_eval = mbm.data_processing.MBSequenceDataset.make_test_loader(
        ds_eval, ds_train_copy, seed=seed, batch_size=batch_size
    )
    with torch.no_grad():
        df_pred_base = model.predict_with_keys(device, dl_eval, ds_eval)

    # Try to pick targets directly from pred df, otherwise from df_eval by ID merge
    if _PFI_G["target_col"] is None:
        pass  # set below
    y_true = None
    if target_col in df_pred_base.columns:
        y_true = df_pred_base[target_col].to_numpy()
    else:
        merged = df_pred_base.merge(
            df_eval[["ID", target_col]].drop_duplicates("ID"), on="ID", how="left"
        )
        if merged[target_col].isna().any():
            raise ValueError(
                "Missing targets after merge; ensure df_eval contains per-ID targets."
            )
        y_true = merged[target_col].to_numpy()

    y_pred = df_pred_base["pred"].to_numpy()
    baseline = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # store globals
    _PFI_G.update(
        dict(
            model=model,
            device=device,
            ds_train_copy=ds_train_copy,
            MONTHLY_COLS=MONTHLY_COLS,
            STATIC_COLS=STATIC_COLS,
            months_head_pad=months_head_pad,
            months_tail_pad=months_tail_pad,
            target_col=target_col,
            baseline=baseline,
            df_eval=df_eval,
            seed=seed,
            batch_size=batch_size,
        )
    )


def _permute_within_groups(
    values: np.ndarray, groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    out = np.empty_like(values)
    # group by group label; shuffle within each block
    # to preserve seasonal distribution
    u, inv = np.unique(groups, return_inverse=True)
    for gi, g in enumerate(u):
        idx = np.where(inv == gi)[0]
        shuf = idx.copy()
        rng.shuffle(shuf)
        out[idx] = values[shuf]
    return out


def _pfi_worker_task(task: Tuple[str, str, int]) -> Tuple[str, str, float]:
    """
    Task = (feature, type, repeat_seed_offset) -> returns (feature, type, delta_rmse)
    """
    # local import
    import massbalancemachine as mbm

    print(f"[PFI worker running] {task}")

    feat, ftype, seed_offset = task
    rng = np.random.default_rng(int(_PFI_G["seed"]) + int(seed_offset))

    df = _PFI_G["df_eval"].copy()
    if ftype == "monthly":
        if "MONTHS" not in df.columns:
            raise ValueError("MONTHS column required for monthly feature permutation.")
        df[feat] = _permute_within_groups(
            df[feat].to_numpy(), df["MONTHS"].to_numpy(), rng
        )
    elif ftype == "static":
        idx = np.arange(len(df))
        rng.shuffle(idx)
        df[feat] = df[feat].to_numpy()[idx]
    else:
        raise ValueError("ftype must be 'monthly' or 'static'.")

    # Rebuild ds/loader for permuted df
    ds_p = mbm.data_processing.MBSequenceDataset.from_dataframe(
        df,
        _PFI_G["MONTHLY_COLS"],
        _PFI_G["STATIC_COLS"],
        months_tail_pad=_PFI_G["months_tail_pad"],
        months_head_pad=_PFI_G["months_head_pad"],
        expect_target=True,
        show_progress=False,
    )
    dl_p = mbm.data_processing.MBSequenceDataset.make_test_loader(
        ds_p,
        _PFI_G["ds_train_copy"],
        seed=_PFI_G["seed"],
        batch_size=_PFI_G["batch_size"],
    )
    with torch.no_grad():
        df_pred = _PFI_G["model"].predict_with_keys(_PFI_G["device"], dl_p, ds_p)

    # Targets
    tcol = _PFI_G["target_col"]
    if tcol in df_pred.columns:
        y_true = df_pred[tcol].to_numpy()
    else:
        merged = df_pred.merge(
            df[["ID", tcol]].drop_duplicates("ID"), on="ID", how="left"
        )
        if merged[tcol].isna().any():
            raise ValueError(
                "Missing targets after merge; ensure df_eval contains per-ID targets."
            )
        y_true = merged[tcol].to_numpy()
    y_pred = df_pred["pred"].to_numpy()

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    delta = rmse - float(_PFI_G["baseline"])
    return feat, ftype, delta


# ------------------------------- user-facing function -------------------------------


def permutation_feature_importance_mbm_parallel(
    cfg,
    custom_params: Dict[str, Any],
    model_filename: str,
    df_eval: pd.DataFrame,
    MONTHLY_COLS: List[str],
    STATIC_COLS: List[str],
    ds_train,
    train_idx,
    target_col: str,
    months_head_pad: int,
    months_tail_pad: int,
    seed: int = 42,
    n_repeats: int = 5,
    batch_size: int = 256,
    max_workers: int = None,
) -> pd.DataFrame:
    """
    Parallel Permutation Feature Importance (CPU).
    Returns DataFrame: ['feature','type','mean_delta','std_delta','baseline','metric_name'].
    """

    # Build list of all tasks (feature x repeat)
    feats = [(f, "monthly") for f in MONTHLY_COLS] + [
        (f, "static") for f in STATIC_COLS
    ]
    tasks = []
    for feat, ftype in feats:
        for r in range(n_repeats):
            tasks.append((feat, ftype, r))

    # Use Linux fork so df_eval stays shared copy-on-write
    ctx = mp.get_context("fork")
    if max_workers is None:
        max_workers = min(max(1, (os.cpu_count() or 2) - 1), 32)

    # Run
    results = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_pfi_worker_init,
        initargs=(
            cfg,
            custom_params,
            model_filename,
            ds_train,
            train_idx,
            MONTHLY_COLS,
            STATIC_COLS,
            months_head_pad,
            months_tail_pad,
            df_eval,
            target_col,
            seed,
            batch_size,
        ),
    ) as ex:
        futs = [ex.submit(_pfi_worker_task, t) for t in tasks]
        for fut in tqdm(
            as_completed(futs), total=len(futs), desc=f"PFI (workers={max_workers})"
        ):
            results.append(fut.result())

    # Aggregate
    rows = []
    baseline = _PFI_G.get(
        "baseline"
    )  # won't be set in parent; recompute baseline here:
    # baseline computed inside workers, but not shared; recompute baseline serially in parent:
    # Minimal recompute on parent with single pass:

    import massbalancemachine as mbm

    _set_cpu_env_threads()
    _set_seeds(seed)
    ds_train_copy = mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
        ds_train
    )
    ds_train_copy.fit_scalers(train_idx)
    device = torch.device("cpu")
    model = mbm.models.LSTM_MB.build_model_from_params(
        cfg, custom_params, device, verbose=False
    )
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state)
    model.eval()
    ds_eval_parent = mbm.data_processing.MBSequenceDataset.from_dataframe(
        df_eval,
        MONTHLY_COLS,
        STATIC_COLS,
        months_tail_pad=months_tail_pad,
        months_head_pad=months_head_pad,
        expect_target=True,
        show_progress=False,
    )
    dl_eval_parent = mbm.data_processing.MBSequenceDataset.make_test_loader(
        ds_eval_parent, ds_train_copy, seed=seed, batch_size=batch_size
    )
    with torch.no_grad():
        df_pred_base = model.predict_with_keys(device, dl_eval_parent, ds_eval_parent)
    if target_col in df_pred_base.columns:
        y_true = df_pred_base[target_col].to_numpy()
    else:
        merged = df_pred_base.merge(
            df_eval[["ID", target_col]].drop_duplicates("ID"), on="ID", how="left"
        )
        if merged[target_col].isna().any():
            raise ValueError(
                "Missing targets after merge; ensure df_eval has per-ID targets."
            )
        y_true = merged[target_col].to_numpy()
    y_pred = df_pred_base["pred"].to_numpy()
    baseline = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Build table
    bucket: Dict[Tuple[str, str], List[float]] = collections.defaultdict(list)
    for feat, ftype, delta in results:
        bucket[(feat, ftype)].append(float(delta))

    for (feat, ftype), deltas in bucket.items():
        mu = float(np.mean(deltas))
        sd = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
        rows.append({"feature": feat, "type": ftype, "mean_delta": mu, "std_delta": sd})

    out = (
        pd.DataFrame(rows)
        .sort_values("mean_delta", ascending=False)
        .reset_index(drop=True)
    )
    out["baseline"] = baseline
    out["metric_name"] = "rmse"
    return out
