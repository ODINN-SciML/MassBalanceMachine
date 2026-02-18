import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import warnings
import logging
import massbalancemachine as mbm


# <---------- LSTM DATASET HELPERS ---------->
def build_combined_LSTM_dataset(
    df_loss,
    df_full,
    monthly_cols,
    static_cols,
    months_head_pad,
    months_tail_pad,
    normalize_target=True,
    expect_target=True,
):
    """
    Build an LSTM-ready sequence dataset by merging full feature data with
    point mass balance observations.

    This function:
    1. Removes target columns from the full feature DataFrame.
    2. Keeps only POINT_BALANCE information from the loss DataFrame.
    3. Merges both on glacier, year, ID, period, and month.
    4. Constructs an `MBSequenceDataset` with appropriate padding and masks.

    Padded months will have POINT_BALANCE = NaN and are excluded from the
    corresponding loss masks.

    Parameters
    ----------
    df_loss : pandas.DataFrame
        DataFrame containing point mass balance observations.
    df_full : pandas.DataFrame
        DataFrame containing full monthly and static feature information.
    monthly_cols : list of str
        Names of monthly (time-varying) feature columns.
    static_cols : list of str
        Names of static (time-invariant) feature columns.
    months_head_pad : int
        Number of padded months at the beginning of each sequence.
    months_tail_pad : int
        Number of padded months at the end of each sequence.
    normalize_target : bool, optional
        Whether to normalize the target variable.
    expect_target : bool, optional
        Whether the dataset should expect a target variable.

    Returns
    -------
    MBSequenceDataset
        LSTM-compatible dataset with masks for valid, winter, and annual loss.
    """
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
    ds = mbm.data_processing.MBSequenceDataset.from_dataframe(
        df=df_combined,
        monthly_cols=monthly_cols,
        static_cols=static_cols,
        months_head_pad=months_head_pad,
        months_tail_pad=months_tail_pad,
        expect_target=expect_target,
        normalize_target=normalize_target,
    )

    return ds


def inspect_LSTM_sample(ds, idx, month_labels=None):
    """
    Inspect and visualize a single LSTM dataset sample.

    Displays:
    - Dataset key (GLACIER, YEAR, ID, PERIOD)
    - Target value
    - Monthly validity, winter, and annual loss masks
    - Line plot of the masks over time

    Parameters
    ----------
    ds : MBSequenceDataset
        LSTM dataset.
    idx : int
        Index of the sample to inspect.
    month_labels : list of str, optional
        Labels for months on the x-axis. If None, generic labels are used.
    """
    x_m = ds.Xm[idx].numpy()
    mv = ds.mv[idx].numpy()
    mw = ds.mw[idx].numpy()
    ma = ds.ma[idx].numpy()
    key = ds.keys[idx]

    if month_labels is None:
        # Infer from dataset order: (MONTHS => pos_map) → sorted by pos
        month_labels = [f"m{i}" for i in range(x_m.shape[0])]

    df = pd.DataFrame(
        {"Month": month_labels, "mv(valid)": mv, "mw(winter)": mw, "ma(annual)": ma}
    )

    print("=== Sample info ===")
    print("Key:", key)
    print("Target y:", float(ds.y[idx].numpy()))
    print()
    print(df)

    # Plot masks
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(mv, label="mv (valid inputs)")
    ax.plot(mw, label="mw (winter MB loss mask)")
    ax.plot(ma, label="ma (annual MB loss mask)")
    ax.set_title(
        f"Masks for sample {idx} (GLACIER={key[0]}, YEAR={key[1]}, PERIOD={key[3]})"
    )
    ax.set_xticks(range(len(month_labels)))
    ax.set_xticklabels(month_labels, rotation=45)
    ax.legend()
    plt.show()


def inspect_LSTM_padded_months(ds, n_samples=5, tol_zero=1e-6):
    """
    Inspect padded versus real months in an LSTM dataset.

    For randomly selected samples, this function prints per-month:
    - Validity mask (mv)
    - Winter mask (mw)
    - Annual mask (ma)
    - Mean feature value
    - Presence of NaNs
    - Whether all features are effectively zero

    Useful for debugging padding, masking, and normalization.

    Parameters
    ----------
    ds : MBSequenceDataset
        LSTM dataset.
    n_samples : int, optional
        Number of random samples to inspect.
    tol_zero : float, optional
        Tolerance below which values are considered zero.
    """
    Xm = ds.Xm.detach().cpu().numpy()  # (B, T, Fm)
    mv = ds.mv.detach().cpu().numpy()  # (B, T)
    mw = ds.mw.detach().cpu().numpy()  # (B, T)
    ma = ds.ma.detach().cpu().numpy()  # (B, T)

    B, T, F = Xm.shape
    idxs = random.sample(range(B), min(n_samples, B))

    print(f"Inspecting {len(idxs)} random samples")
    for i in idxs:
        print("\n────────────────────────────────────")
        print(f"Sample {i} — {ds.keys[i]}")
        print("MONTH | mv | mw | ma | mean(X) | has_NaN | all_zero")
        for t in range(T):
            x = Xm[i, t, :]
            mean_val = float(np.nanmean(x))
            nan_mask = np.isnan(x).any()
            zero_mask = np.all(np.abs(x) < tol_zero)
            print(
                f"{t:2d} | {int(mv[i, t])}  | {int(mw[i, t])}  | {int(ma[i, t])}  | "
                f"{mean_val:7.3f} |  {nan_mask}  |  {zero_mask}"
            )


# --------------------------- MULTI REGION HANDLING ---------------------------
def _check_for_nans(key, df_loss, df_full, monthly_cols, static_cols, strict=True):
    """
    Checks for NaNs/Infs in features and targets.
    Raises ValueError if strict=True, otherwise prints warning.
    """
    feat_cols = [c for c in (monthly_cols + static_cols) if c in df_full.columns]

    # --- feature NaNs ---
    n_nan_feat = df_full[feat_cols].isna().sum().sum()
    n_inf_feat = np.isinf(
        df_full[feat_cols].to_numpy(dtype="float64", copy=False)
    ).sum()

    # --- target NaNs ---
    n_nan_target = df_loss["POINT_BALANCE"].isna().sum()
    n_inf_target = np.isinf(
        df_loss["POINT_BALANCE"].to_numpy(dtype="float64", copy=False)
    ).sum()

    if any([n_nan_feat, n_inf_feat, n_nan_target, n_inf_target]):

        msg = (
            f"[{key}] Data integrity issue:\n"
            f"  Feature NaNs: {n_nan_feat}\n"
            f"  Feature Infs: {n_inf_feat}\n"
            f"  Target  NaNs: {n_nan_target}\n"
            f"  Target  Infs: {n_inf_target}"
        )

        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg)


def _lstm_cache_paths(cfg, key: str, cache_dir: str):
    out_dir = os.path.join(cache_dir)
    os.makedirs(out_dir, exist_ok=True)
    train_p = os.path.join(out_dir, f"{key}_train.joblib")
    test_p = os.path.join(out_dir, f"{key}_test.joblib")
    split_p = os.path.join(out_dir, f"{key}_split.joblib")
    return train_p, test_p, split_p


def build_or_load_lstm_for_key(
    cfg,
    key: str,
    res: dict,
    MONTHLY_COLS,
    STATIC_COLS,
    val_ratio=0.2,
    cache_dir="logs/LSTM_cache",
    force_recompute=False,
    normalize_target=True,
    expect_target=True,
    strict_nan=True,  # <-- new
):

    train_p, test_p, split_p = _lstm_cache_paths(cfg, key, cache_dir=cache_dir)

    if (not force_recompute) and all(
        os.path.exists(p) for p in [train_p, test_p, split_p]
    ):
        ds_train = joblib.load(train_p)
        ds_test = joblib.load(test_p)
        split = joblib.load(split_p)
        return ds_train, ds_test, split["train_idx"], split["val_idx"]

    # required pieces from monthly prep
    df_train = res["df_train"]
    df_test = res["df_test"]
    df_train_aug = res["df_train_aug"]
    df_test_aug = res["df_test_aug"]
    months_head_pad = res["months_head_pad"]
    months_tail_pad = res["months_tail_pad"]

    # --- safety check ---
    _check_for_nans(
        key,
        df_loss=df_train,
        df_full=df_train_aug,
        monthly_cols=MONTHLY_COLS,
        static_cols=STATIC_COLS,
        strict=strict_nan,
    )

    _check_for_nans(
        key,
        df_loss=df_test,
        df_full=df_test_aug,
        monthly_cols=MONTHLY_COLS,
        static_cols=STATIC_COLS,
        strict=strict_nan,
    )

    mbm.utils.seed_all(cfg.seed)

    ds_train = build_combined_LSTM_dataset(
        df_loss=df_train,
        df_full=df_train_aug,
        monthly_cols=MONTHLY_COLS,
        static_cols=STATIC_COLS,
        months_head_pad=months_head_pad,
        months_tail_pad=months_tail_pad,
        normalize_target=normalize_target,
        expect_target=expect_target,
    )

    ds_test = build_combined_LSTM_dataset(
        df_loss=df_test,
        df_full=df_test_aug,
        monthly_cols=MONTHLY_COLS,
        static_cols=STATIC_COLS,
        months_head_pad=months_head_pad,
        months_tail_pad=months_tail_pad,
        normalize_target=normalize_target,
        expect_target=expect_target,
    )

    train_idx, val_idx = mbm.data_processing.MBSequenceDataset.split_indices(
        len(ds_train), val_ratio=val_ratio, seed=cfg.seed
    )

    joblib.dump(ds_train, train_p, compress=3)
    joblib.dump(ds_test, test_p, compress=3)
    joblib.dump({"train_idx": train_idx, "val_idx": val_idx}, split_p, compress=3)

    return ds_train, ds_test, train_idx, val_idx


def build_or_load_lstm_all(
    cfg,
    res_all: dict,  # e.g. {"07_SJM": res, "08_NOR": res, ...}
    MONTHLY_COLS,
    STATIC_COLS,
    cache_dir="logs/LSTM_cache",
    only_keys=None,  # e.g. ["08_NOR"] to recompute only Norway
    force_recompute=False,  # global default
    val_ratio=0.2,
):
    outputs = {}
    only_keys_set = set(only_keys) if only_keys else None

    for key, res in res_all.items():
        if res is None:
            continue

        # recompute only some keys; others load if possible
        fr = force_recompute
        if only_keys_set is not None:
            fr = key in only_keys_set

        logging.info(f"\nLSTM prep: {key} (force_recompute={fr})")

        ds_train, ds_test, train_idx, val_idx = build_or_load_lstm_for_key(
            cfg=cfg,
            key=key,
            res=res,
            MONTHLY_COLS=MONTHLY_COLS,
            STATIC_COLS=STATIC_COLS,
            val_ratio=val_ratio,
            cache_dir=cache_dir,
            force_recompute=fr,
        )

        outputs[key] = {
            "ds_train": ds_train,
            "ds_test": ds_test,
            "train_idx": train_idx,
            "val_idx": val_idx,
        }

    return outputs
