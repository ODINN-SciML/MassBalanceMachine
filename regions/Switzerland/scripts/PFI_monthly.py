from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch


def _pfi_monthly_worker_task(
    task,
    cfg,
    custom_params,
    model_filename,
    df_eval,
    ds_train,
    train_idx,
    MONTHLY_COLS,
    STATIC_COLS,
    months_head_pad,
    months_tail_pad,
    seed,
    batch_size,
    denorm=True,
):
    """Worker function for monthly feature permutation."""
    import numpy as np
    import torch
    import massbalancemachine as mbm

    feat, m_idx, repeat = task
    np.random.seed(seed + repeat)

    # --- rebuild train scalers ---
    ds_train_copy = mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
        ds_train
    )
    ds_train_copy.fit_scalers(train_idx)

    # --- copy dataframe ---
    df_mod = df_eval.copy()

    # Expect df_eval to have "MONTH_IDX" or similar (0–14 per row)
    if "MONTH_IDX" not in df_mod.columns:
        raise ValueError(
            "df_eval must include MONTH_IDX for month-specific permutation."
        )

    # Select rows corresponding to this month
    mask = df_mod["MONTH_IDX"] == m_idx
    if mask.any():
        # Permute this feature only within this month
        vals = df_mod.loc[mask, feat].values
        np.random.shuffle(vals)
        df_mod.loc[mask, feat] = vals

    # --- rebuild dataset ---
    ds_perm = mbm.data_processing.MBSequenceDataset.from_dataframe(
        df_mod,
        MONTHLY_COLS,
        STATIC_COLS,
        months_tail_pad=months_tail_pad,
        months_head_pad=months_head_pad,
        expect_target=True,
        show_progress=False,
    )
    dl_perm = mbm.data_processing.MBSequenceDataset.make_test_loader(
        ds_perm, ds_train_copy, seed=seed + repeat, batch_size=batch_size
    )

    # --- rebuild model ---
    device = torch.device("cpu")
    model = mbm.models.LSTM_MB.build_model_from_params(
        cfg, custom_params, device, verbose=False
    )
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # --- predict ---
    with torch.no_grad():
        df_pred = model.predict_with_keys(device, dl_perm, ds_perm, denorm=denorm)

    df_eval_perm = df_pred.merge(
        df_eval[["ID", "PERIOD", "GLACIER", "YEAR", "y"]].drop_duplicates("ID"),
        on=["ID", "PERIOD", "GLACIER", "YEAR"],
        how="left",
    )

    def _rmse(period):
        d = df_eval_perm[df_eval_perm["PERIOD"] == period]
        return np.sqrt(np.mean((d["pred"] - d["y"]) ** 2)) if len(d) else np.nan

    # --- compute RMSEs ---
    rmse_w = _rmse("winter")
    rmse_a = _rmse("annual")
    rmse_global = (
        np.sqrt(np.mean((df_eval_perm["pred"] - df_eval_perm["y"]) ** 2))
        if len(df_eval_perm)
        else np.nan
    )

    return feat, m_idx, rmse_w, rmse_a, rmse_global


def permutation_feature_importance_mbm_monthly_parallel(
    cfg,
    custom_params,
    model_filename,
    df_eval,
    MONTHLY_COLS,
    STATIC_COLS,
    ds_train,
    train_idx,
    months_head_pad,
    months_tail_pad,
    seed=42,
    n_repeats=3,
    batch_size=256,
    denorm=True,
    max_workers=None,
):
    """Parallel monthly permutation feature importance (includes global RMSE).
    Now also permutes static (dynamic-invariant) features on a per-month basis
    to assess cross-temporal interactions.
    """

    import massbalancemachine as mbm

    # --- Prepare baseline ---
    ds_train_copy = mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
        ds_train
    )
    ds_train_copy.fit_scalers(train_idx)

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

    device = torch.device("cpu")
    model = mbm.models.LSTM_MB.build_model_from_params(
        cfg, custom_params, device, verbose=False
    )
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        df_pred_base = model.predict_with_keys(device, dl_eval, ds_eval, denorm=denorm)

    df_eval_base = df_pred_base.merge(
        df_eval[["ID", "PERIOD", "GLACIER", "YEAR", "y"]].drop_duplicates("ID"),
        on=["ID", "PERIOD", "GLACIER", "YEAR"],
        how="left",
    )

    def _rmse(period):
        d = df_eval_base[df_eval_base["PERIOD"] == period]
        return np.sqrt(np.mean((d["pred"] - d["y"]) ** 2)) if len(d) else np.nan

    # --- Baseline RMSEs ---
    baseline_w = _rmse("winter")
    baseline_a = _rmse("annual")
    baseline_global = np.sqrt(np.mean((df_eval_base["pred"] - df_eval_base["y"]) ** 2))

    print(
        f"[Baseline RMSE] winter={baseline_w:.3f} | annual={baseline_a:.3f} | global={baseline_global:.3f}"
    )

    # --- Build tasks ---
    month_names = [
        "aug_",
        "sep_",
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
        "oct_",
    ]
    n_months = len(month_names)

    # all_features = MONTHLY_COLS + STATIC_COLS
    all_features = MONTHLY_COLS

    tasks = [
        (feat, m_idx, r)
        for feat in all_features
        for m_idx in range(n_months)
        for r in range(n_repeats)
    ]

    # --- Run in parallel ---
    ctx = torch.multiprocessing.get_context("fork")
    if max_workers is None:
        import os

        max_workers = min(max(1, (os.cpu_count() or 2) - 1), 32)

    worker = partial(
        _pfi_monthly_worker_task,
        cfg=cfg,
        custom_params=custom_params,
        model_filename=model_filename,
        df_eval=df_eval,
        ds_train=ds_train,
        train_idx=train_idx,
        MONTHLY_COLS=MONTHLY_COLS,
        STATIC_COLS=STATIC_COLS,
        months_head_pad=months_head_pad,
        months_tail_pad=months_tail_pad,
        seed=seed,
        batch_size=batch_size,
        denorm=denorm,
    )

    results = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        for res in tqdm(ex.map(worker, tasks), total=len(tasks), desc="Monthly PFI"):
            results.append(res)

    # --- Aggregate ---
    df = pd.DataFrame(
        results,
        columns=["feature", "month_idx", "rmse_winter", "rmse_annual", "rmse_global"],
    )
    df["month"] = [month_names[m % len(month_names)] for m in df["month_idx"]]

    df["delta_winter"] = df["rmse_winter"] - baseline_w
    df["delta_annual"] = df["rmse_annual"] - baseline_a
    df["delta_global"] = df["rmse_global"] - baseline_global

    out = (
        df.groupby(["feature", "month"])
        .agg(
            mean_delta_winter=("delta_winter", "mean"),
            std_delta_winter=("delta_winter", "std"),
            mean_delta_annual=("delta_annual", "mean"),
            std_delta_annual=("delta_annual", "std"),
            mean_delta_global=("delta_global", "mean"),
            std_delta_global=("delta_global", "std"),
        )
        .reset_index()
    )

    out["baseline_rmse_winter"] = baseline_w
    out["baseline_rmse_annual"] = baseline_a
    out["baseline_rmse_global"] = baseline_global

    return out
