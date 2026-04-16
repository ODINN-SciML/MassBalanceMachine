import os
import joblib
import warnings
import pandas as pd
import numpy as np
import logging
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import random as rd
import torch

import massbalancemachine as mbm

from regions.TF_Europe.scripts.dataset import build_combined_LSTM_dataset


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


# ------------------------------------------------------------
# 1) Build/load a PRISTINE dataset only (no scalers inside)
# ------------------------------------------------------------
def build_or_load_lstm_dataset_only(
    cfg,
    key: str,
    df_loss,
    df_full,
    months_head_pad,
    months_tail_pad,
    MONTHLY_COLS,
    STATIC_COLS,
    cache_dir="logs/LSTM_cache",
    force_recompute=False,
    normalize_target=True,
    expect_target=True,
    strict_nan=True,
    kind="dataset",  # keep kind to avoid duplicate functions; default "dataset"
    show_progress=True,
):
    out_dir = os.path.join(cache_dir)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{key}_{kind}.joblib")

    # ---- Load cached (must be pristine) ----
    if (not force_recompute) and os.path.exists(p):
        ds = joblib.load(p)
        if (
            (ds.month_mean is not None)
            or (ds.static_mean is not None)
            or (ds.y_mean is not None)
        ):
            raise ValueError(
                f"{key}_{kind}: cached dataset already has scalers set. "
                "Cache should store pristine datasets only."
            )
        return ds

    # ---- Build fresh ----
    _check_for_nans(
        key,
        df_loss=df_loss,
        df_full=df_full,
        monthly_cols=MONTHLY_COLS,
        static_cols=STATIC_COLS,
        strict=strict_nan,
    )

    mbm.utils.seed_all(cfg.seed)

    ds = build_combined_LSTM_dataset(
        df_loss=df_loss,
        df_full=df_full,
        monthly_cols=MONTHLY_COLS,
        static_cols=STATIC_COLS,
        months_head_pad=months_head_pad,
        months_tail_pad=months_tail_pad,
        normalize_target=normalize_target,
        expect_target=expect_target,
        show_progress=show_progress,
    )

    # sanity: ensure pristine before caching
    if (
        (ds.month_mean is not None)
        or (ds.static_mean is not None)
        or (ds.y_mean is not None)
    ):
        raise ValueError(
            f"{key}_{kind}: newly built dataset unexpectedly has scalers set."
        )

    joblib.dump(ds, p, compress=3)
    return ds


# ------------------------------------------------------------
# 2) Transfer-learning slicing (no scaling logic here)
# ------------------------------------------------------------
def make_res_transfer_learning(
    res_xreg: dict,
    target_code: str,
    ft_glaciers: list,
    source_col="SOURCE_CODE",
    glacier_col="GLACIER",
):
    """
    Returns:
      res_pretrain: CH-only (df_train/df_train_aug + pads)
      res_ft: target finetune subset (df_train/df_train_aug + pads)
      res_test: target holdout (df_test/df_test_aug + pads)
    """
    res_pretrain = {
        "df_train": res_xreg["df_train"],
        "df_train_aug": res_xreg["df_train_aug"],
        "months_head_pad": res_xreg["months_head_pad"],
        "months_tail_pad": res_xreg["months_tail_pad"],
    }

    df_t_all = res_xreg["df_test"]
    df_t_all_aug = res_xreg["df_test_aug"]

    df_target = df_t_all.loc[df_t_all[source_col] == target_code].copy()
    df_target_aug = df_t_all_aug.loc[df_t_all_aug[source_col] == target_code].copy()

    df_ft = df_target.loc[df_target[glacier_col].isin(ft_glaciers)].copy()
    df_ft_aug = df_target_aug.loc[df_target_aug[glacier_col].isin(ft_glaciers)].copy()

    df_hold = df_target.loc[~df_target[glacier_col].isin(ft_glaciers)].copy()
    df_hold_aug = df_target_aug.loc[
        ~df_target_aug[glacier_col].isin(ft_glaciers)
    ].copy()

    res_ft = {
        "df_train": df_ft,
        "df_train_aug": df_ft_aug,
        "months_head_pad": res_xreg["months_head_pad"],
        "months_tail_pad": res_xreg["months_tail_pad"],
    }

    res_test = {
        "df_test": df_hold,
        "df_test_aug": df_hold_aug,
        "months_head_pad": res_xreg["months_head_pad"],
        "months_tail_pad": res_xreg["months_tail_pad"],
    }

    return res_pretrain, res_ft, res_test


# ------------------------------------------------------------
# 3) Build/load CH train dataset + split + SCALER DONOR (Option 2)
# ------------------------------------------------------------
def build_or_load_lstm_train_only(
    cfg,
    key_train: str,
    res_train: dict,  # must contain df_train, df_train_aug, pads
    MONTHLY_COLS,
    STATIC_COLS,
    val_ratio=0.2,
    cache_dir="logs/LSTM_cache",
    force_recompute=False,
    normalize_target=True,
    expect_target=True,
    strict_nan=True,
    show_progress=True,
):
    train_p, _, split_p = _lstm_cache_paths(cfg, key_train, cache_dir=cache_dir)
    scaler_p = os.path.join(cache_dir, f"{key_train}_scalers.joblib")

    # ---- Load cached assets (train ds must be pristine; scalers ds must have scalers) ----
    if (not force_recompute) and all(
        os.path.exists(p) for p in [train_p, split_p, scaler_p]
    ):
        ds_train = joblib.load(train_p)
        split = joblib.load(split_p)
        ds_scalers = joblib.load(scaler_p)

        # guards
        if (
            (ds_train.month_mean is not None)
            or (ds_train.static_mean is not None)
            or (ds_train.y_mean is not None)
        ):
            raise ValueError(
                f"{key_train}: cached TRAIN dataset has scalers set. "
                "train_p cache must store pristine dataset only."
            )
        if (
            (ds_scalers.month_mean is None)
            or (ds_scalers.static_mean is None)
            or (ds_scalers.y_mean is None)
        ):
            raise ValueError(f"{key_train}: cached SCALER donor is missing scalers.")

        return ds_train, split["train_idx"], split["val_idx"], ds_scalers

    # ---- Build fresh ----
    df_train = res_train["df_train"]
    df_train_aug = res_train["df_train_aug"]
    months_head_pad = res_train["months_head_pad"]
    months_tail_pad = res_train["months_tail_pad"]

    _check_for_nans(
        key_train,
        df_loss=df_train,
        df_full=df_train_aug,
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
        show_progress=show_progress,
    )

    # split indices
    train_idx, val_idx = mbm.data_processing.MBSequenceDataset.split_indices(
        len(ds_train), val_ratio=val_ratio, seed=cfg.seed
    )

    # ---- NEW: create scaler donor and fit scalers on CH TRAIN split only ----
    ds_scalers = mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
        ds_train
    )
    ds_scalers.fit_scalers(train_idx)

    # ---- Cache ----
    joblib.dump(ds_train, train_p, compress=3)
    joblib.dump({"train_idx": train_idx, "val_idx": val_idx}, split_p, compress=3)
    joblib.dump(ds_scalers, scaler_p, compress=3)

    return ds_train, train_idx, val_idx, ds_scalers


def build_source_codes_for_dataset(ds, df_monthly, source_col="SOURCE_CODE"):
    """
    Returns a list[str] of SOURCE_CODE aligned with ds.keys.
    Assumes SOURCE_CODE is constant per (GLACIER, YEAR, ID, PERIOD).
    """
    if source_col not in df_monthly.columns:
        raise KeyError(f"df_monthly is missing '{source_col}'")

    # mapping per sequence key
    key_cols = ["GLACIER", "YEAR", "ID", "PERIOD"]
    miss = [c for c in key_cols if c not in df_monthly.columns]
    if miss:
        raise KeyError(f"df_monthly missing required key cols: {miss}")

    tmp = df_monthly[key_cols + [source_col]].copy()
    tmp["PERIOD"] = tmp["PERIOD"].astype(str).str.strip().str.lower()

    # If a key appears with multiple source codes, that's a data issue
    nun = tmp.groupby(key_cols)[source_col].nunique()
    bad = nun[nun > 1]
    if len(bad) > 0:
        ex = bad.index[:5].tolist()
        raise ValueError(
            f"Found keys with multiple SOURCE_CODE values (showing first 5): {ex}"
        )

    key_to_sc = tmp.groupby(key_cols)[source_col].first().to_dict()

    out = []
    for g, yr, mid, per in ds.keys:
        k = (g, int(yr), int(mid), str(per).strip().lower())
        if k not in key_to_sc:
            raise KeyError(f"Missing SOURCE_CODE for ds key {k}")
        out.append(key_to_sc[k])

    return out


def build_transfer_learning_assets(
    cfg,
    res_xreg,
    FT_GLACIERS,
    MONTHLY_COLS,
    STATIC_COLS,
    cache_dir="logs/LSTM_cache_TL",
    force_recompute=False,
    val_ratio=0.2,
    show_progress=True,
):
    logging.info("\n" + "=" * 70)
    logging.info("TRANSFER LEARNING ASSET PREPARATION")
    logging.info("=" * 70)
    logging.info(f"Cache directory: {cache_dir}")
    logging.info(f"Regions in FT_GLACIERS: {list(FT_GLACIERS.keys())}")

    assets = {}

    # ------------------------------------------------------------------
    # 1) CH PRETRAIN DATASET (shared across all TL experiments)
    # ------------------------------------------------------------------
    key_train = "TL_CH_TRAIN"

    logging.info("\n--- CH PRETRAIN DATASET ---")
    logging.info(f"Cache key: {key_train}")
    logging.info(f"Force recompute: {force_recompute}")

    res_train = {
        "df_train": res_xreg["df_train"],
        "df_train_aug": res_xreg["df_train_aug"],
        "months_head_pad": res_xreg["months_head_pad"],
        "months_tail_pad": res_xreg["months_tail_pad"],
    }

    logging.info(
        f"CH train rows: {len(res_train['df_train'])} | "
        f"Aug rows: {len(res_train['df_train_aug'])}"
    )

    # ---- Option 2: also returns ds_ch_scalers (cached) ----
    ds_ch, train_idx, val_idx, ds_ch_scalers = build_or_load_lstm_train_only(
        cfg=cfg,
        key_train=key_train,
        res_train=res_train,
        MONTHLY_COLS=MONTHLY_COLS,
        STATIC_COLS=STATIC_COLS,
        val_ratio=val_ratio,
        cache_dir=cache_dir,
        force_recompute=force_recompute,
    )

    ch_source_codes = build_source_codes_for_dataset(
        ds_ch, res_xreg["df_train_aug"], source_col="SOURCE_CODE"
    )

    # IMPORTANT: do NOT fit scalers on ds_ch here anymore
    # ds_ch_scalers is the scaler donor; ds_ch stays pristine.

    logging.info(
        f"CH dataset size (sequences): {len(ds_ch)} | "
        f"Train split: {len(train_idx)} | Val split: {len(val_idx)}"
    )

    # ------------------------------------------------------------------
    # 2) PER REGION × SPLIT
    # ------------------------------------------------------------------
    for reg, splits in FT_GLACIERS.items():

        logging.info("\n" + "-" * 60)
        logging.info(f"TARGET REGION: {reg}")
        logging.info("-" * 60)

        for split_name, ft_gls in splits.items():

            exp_key = f"TL_CH_to_{reg}_{split_name}"

            logging.info("\n" + "-" * 40)
            logging.info(f"Experiment: {exp_key}")
            logging.info(f"Finetune glacier count: {len(ft_gls)}")

            # ----------------------------------------------------------
            # Slice finetune + holdout
            # ----------------------------------------------------------
            res_pre, res_ft, res_test = make_res_transfer_learning(
                res_xreg=res_xreg,
                target_code=reg,
                ft_glaciers=ft_gls,
            )

            logging.info(
                f"FT rows: {len(res_ft['df_train'])} | "
                f"FT aug rows: {len(res_ft['df_train_aug'])}"
            )

            logging.info(
                f"Holdout rows: {len(res_test['df_test'])} | "
                f"Holdout aug rows: {len(res_test['df_test_aug'])}"
            )

            if len(res_ft["df_train"]) == 0:
                logging.warning(f"{exp_key}: EMPTY FINETUNE SET -> skipping.")
                continue

            # ----------------------------------------------------------
            # Finetune dataset (PRISTINE)
            # ----------------------------------------------------------
            ft_cache_key = f"{exp_key}_FT"
            logging.info(f"Finetune cache key: {ft_cache_key}")

            ds_ft = build_or_load_lstm_dataset_only(
                cfg=cfg,
                key=ft_cache_key,
                df_loss=res_ft["df_train"],
                df_full=res_ft["df_train_aug"],
                months_head_pad=res_ft["months_head_pad"],
                months_tail_pad=res_ft["months_tail_pad"],
                MONTHLY_COLS=MONTHLY_COLS,
                STATIC_COLS=STATIC_COLS,
                cache_dir=cache_dir,
                force_recompute=force_recompute,
                kind="ft",
                show_progress=show_progress,
            )

            logging.info(f"Finetune dataset size (sequences): {len(ds_ft)}")

            ft_train_idx, ft_val_idx = (
                mbm.data_processing.MBSequenceDataset.split_indices(
                    len(ds_ft), val_ratio=val_ratio, seed=cfg.seed
                )
            )

            logging.info(
                f"FT train split: {len(ft_train_idx)} | "
                f"FT val split: {len(ft_val_idx)}"
            )

            # ----------------------------------------------------------
            # Holdout test dataset (PRISTINE)
            # ----------------------------------------------------------
            ds_test = None
            if len(res_test["df_test"]) > 0:

                test_cache_key = f"{exp_key}_TEST"
                logging.info(f"Holdout cache key: {test_cache_key}")

                ds_test = build_or_load_lstm_dataset_only(
                    cfg=cfg,
                    key=test_cache_key,
                    df_loss=res_test["df_test"],
                    df_full=res_test["df_test_aug"],
                    months_head_pad=res_test["months_head_pad"],
                    months_tail_pad=res_test["months_tail_pad"],
                    MONTHLY_COLS=MONTHLY_COLS,
                    STATIC_COLS=STATIC_COLS,
                    cache_dir=cache_dir,
                    force_recompute=force_recompute,
                    kind="test",
                    show_progress=show_progress,
                )

                logging.info(f"Holdout dataset size (sequences): {len(ds_test)}")

            else:
                logging.warning(f"{exp_key}: No holdout test set available.")

            ft_source_codes = build_source_codes_for_dataset(
                ds_ft, res_ft["df_train_aug"], source_col="SOURCE_CODE"
            )

            test_source_codes = None
            if ds_test is not None:
                test_source_codes = build_source_codes_for_dataset(
                    ds_test, res_test["df_test_aug"], source_col="SOURCE_CODE"
                )

            domain_vocab = sorted(
                set(ch_source_codes)
                | set(ft_source_codes)
                | (set(test_source_codes) if test_source_codes is not None else set())
            )

            # ----------------------------------------------------------
            # Store assets (include ds_ch_scalers!)
            # ----------------------------------------------------------
            assets[exp_key] = {
                "ds_pretrain": ds_ch,  # pristine CH dataset
                "ds_pretrain_scalers": ds_ch_scalers,  # <-- IMPORTANT: scaler donor
                "pretrain_train_idx": train_idx,
                "pretrain_val_idx": val_idx,
                "ds_finetune": ds_ft,  # pristine FT dataset
                "finetune_train_idx": ft_train_idx,
                "finetune_val_idx": ft_val_idx,
                "ds_test": ds_test,  # pristine test dataset
                "target_code": reg,
                "split_name": split_name,
                "ft_glaciers": ft_gls,
                "cache_keys": {
                    "pretrain": key_train,
                    "finetune": ft_cache_key,
                    "test": f"{exp_key}_TEST",
                },
                "ft_source_codes": ft_source_codes,
                "test_source_codes": test_source_codes,
                "pretrain_source_codes": ch_source_codes,
                "domain_vocab": domain_vocab,
            }

    logging.info("\nFinished building transfer learning assets.")
    logging.info("=" * 70 + "\n")

    return assets


# ----------------- For cross-regional modelling only -----------------
def build_xreg_res_all(
    res_xreg: dict,
    target_source_codes=None,
    region_groups: dict | None = None,
    source_col: str = "SOURCE_CODE",
    ch_code: str = "CH",
    key_prefix: str = "XREG_CH_TO",
) -> dict:
    """
    Build a res_all dict: {key: res_like_dict} where each entry contains
    df_train (CH) and df_test (one target region or a merged group of regions).

    Individual regions and groups can coexist — groups simply pool the rows
    of their member codes into a single df_test.

    Parameters
    ----------
    res_xreg : dict
        Output of prepare_monthly_df_crossregional_tgt_to_EU (or similar).
        Must contain df_train, df_train_aug, df_test, df_test_aug,
        months_head_pad, months_tail_pad.
    target_source_codes : list[str] or None
        Individual region codes to include as standalone entries.
        If None, auto-discovers all codes in df_test except ch_code,
        MINUS any codes that are already covered by region_groups.
    region_groups : dict[str, list[str]] or None
        Mapping of group name -> list of SOURCE_CODE values to pool.
        e.g. {"CEU": ["FR", "IT_AT"], "USCA": ["ALA", "CAW"]}
        Group entries are keyed as "{key_prefix}_{group_name}".
    source_col : str
        Column in df_test identifying the source region (default: SOURCE_CODE).
    ch_code : str
        Code for the training region to exclude from targets (default: CH).
    key_prefix : str
        Prefix for all result keys (default: XREG_CH_TO).

    Returns
    -------
    dict
        Keys are "{key_prefix}_{code_or_group}", values are res-like dicts
        with df_train, df_train_aug, df_test, df_test_aug, pad keys.
    """
    df_test_full = res_xreg["df_test"]
    df_test_aug_full = res_xreg["df_test_aug"]

    if source_col not in df_test_full.columns:
        raise ValueError(f"Missing '{source_col}' in res_xreg['df_test'].")

    region_groups = region_groups or {}

    # codes already covered by a group — excluded from individual entries
    # unless explicitly listed in target_source_codes
    grouped_codes = {c.upper() for codes in region_groups.values() for c in codes}

    # auto-discover individual codes, excluding CH and grouped codes
    if target_source_codes is None:
        all_codes = set(
            df_test_full[source_col].dropna().astype(str).str.upper().unique()
        )
        target_source_codes = sorted(all_codes - {ch_code.upper()} - grouped_codes)

    res_all = {}

    # --- individual region entries ---
    for sc in target_source_codes:
        sc_upper = sc.upper()
        key = f"{key_prefix}_{sc_upper}"

        mask = df_test_full[source_col].astype(str).str.upper() == sc_upper
        mask_aug = df_test_aug_full[source_col].astype(str).str.upper() == sc_upper

        df_test_sc = df_test_full.loc[mask].copy()
        df_test_aug_sc = df_test_aug_full.loc[mask_aug].copy()

        if len(df_test_sc) == 0:
            print(f"Warning: no test rows found for code '{sc}', skipping.")
            continue

        res_all[key] = {
            "df_train": res_xreg["df_train"],
            "df_train_aug": res_xreg["df_train_aug"],
            "df_test": df_test_sc,
            "df_test_aug": df_test_aug_sc,
            "months_head_pad": res_xreg["months_head_pad"],
            "months_tail_pad": res_xreg["months_tail_pad"],
        }

    # --- grouped region entries ---
    for group_name, codes in region_groups.items():
        key = f"{key_prefix}_{group_name}"

        codes_upper = [c.upper() for c in codes]

        mask = df_test_full[source_col].astype(str).str.upper().isin(codes_upper)
        mask_aug = (
            df_test_aug_full[source_col].astype(str).str.upper().isin(codes_upper)
        )

        df_test_group = df_test_full.loc[mask].copy()
        df_test_aug_group = df_test_aug_full.loc[mask_aug].copy()

        if len(df_test_group) == 0:
            print(
                f"Warning: no test rows found for group '{group_name}' "
                f"(codes: {codes_upper}), skipping."
            )
            continue

        print(
            f"Group '{group_name}': pooled {len(df_test_group)} rows "
            f"from {codes_upper}."
        )

        res_all[key] = {
            "df_train": res_xreg["df_train"],
            "df_train_aug": res_xreg["df_train_aug"],
            "df_test": df_test_group,
            "df_test_aug": df_test_aug_group,
            "months_head_pad": res_xreg["months_head_pad"],
            "months_tail_pad": res_xreg["months_tail_pad"],
        }

    return res_all


def build_or_load_lstm_test_only(
    cfg,
    key_test: str,
    res_test: dict,  # must contain df_test, df_test_aug, pads
    MONTHLY_COLS,
    STATIC_COLS,
    cache_dir="logs/LSTM_cache",
    force_recompute=False,
    normalize_target=True,
    expect_target=True,
    strict_nan=True,
    show_progress=True,
):
    _, test_p, _ = _lstm_cache_paths(cfg, key_test, cache_dir=cache_dir)

    if (not force_recompute) and os.path.exists(test_p):
        return joblib.load(test_p)

    df_test = res_test["df_test"]
    df_test_aug = res_test["df_test_aug"]
    months_head_pad = res_test["months_head_pad"]
    months_tail_pad = res_test["months_tail_pad"]

    _check_for_nans(
        key_test,
        df_loss=df_test,
        df_full=df_test_aug,
        monthly_cols=MONTHLY_COLS,
        static_cols=STATIC_COLS,
        strict=strict_nan,
    )

    mbm.utils.seed_all(cfg.seed)

    ds_test = build_combined_LSTM_dataset(
        df_loss=df_test,
        df_full=df_test_aug,
        monthly_cols=MONTHLY_COLS,
        static_cols=STATIC_COLS,
        months_head_pad=months_head_pad,
        months_tail_pad=months_tail_pad,
        normalize_target=normalize_target,
        expect_target=expect_target,
        show_progress=show_progress,
    )

    joblib.dump(ds_test, test_p, compress=3)
    return ds_test


def build_or_load_lstm_all_xreg(
    cfg,
    res_xreg: dict,
    MONTHLY_COLS,
    STATIC_COLS,
    target_source_codes=None,
    region_groups: dict | None = None,
    source_col="SOURCE_CODE",
    ch_code="CH",
    cache_dir="logs/LSTM_cache",
    force_recompute_train=False,
    force_recompute_tests=False,
    only_test_keys=None,
    val_ratio=0.2,
    normalize_target=True,
    expect_target=True,
    strict_nan=True,
):
    """
    Build or load LSTM datasets for cross-regional transfer (CH → targets).

    Supports both individual region codes and grouped regions (pooled df_test).
    Groups are keyed as "XREG_CH_TO_{GROUP_NAME}" and pool all member codes
    into a single test dataset.

    Parameters
    ----------
    region_groups : dict[str, list[str]] or None
        e.g. {"CEU": ["FR", "IT_AT"], "USCA": ["ALA", "CAW"]}
        Codes in groups are excluded from individual auto-discovery unless
        explicitly listed in target_source_codes.
    (all other parameters unchanged from original)
    """
    logging.info("\n" + "=" * 60)
    logging.info("CROSS-REGIONAL LSTM DATASET PREPARATION (CH → EU)")
    logging.info("=" * 60)

    region_groups = region_groups or {}

    # codes already covered by a group
    grouped_codes = {c for codes in region_groups.values() for c in codes}

    # ---- discover individual target codes ----
    df_test_all = res_xreg["df_test"]
    df_test_aug_all = res_xreg["df_test_aug"]

    if target_source_codes is None:
        all_codes = set(df_test_all[source_col].dropna().unique()) - {ch_code}
        target_source_codes = sorted(all_codes - grouped_codes)
        logging.info(
            f"Auto-detected individual target SOURCE_CODEs "
            f"(excluding {ch_code} and grouped codes {grouped_codes}): "
            f"{target_source_codes}"
        )
    else:
        logging.info(f"Using provided target SOURCE_CODEs: {target_source_codes}")

    if region_groups:
        logging.info(f"Region groups: { {k: v for k, v in region_groups.items()} }")

    logging.info(
        f"Total individual targets: {len(target_source_codes)} | "
        f"Total groups: {len(region_groups)}"
    )
    logging.info(f"Cache directory: {cache_dir}")

    # ---- train (CH) cached once ----
    key_train = "XREG_CH_TRAIN"

    logging.info("\n--- CH TRAIN DATASET ---")
    logging.info(f"Cache key: {key_train}")
    logging.info(f"Force recompute train: {force_recompute_train}")

    res_train = {
        "df_train": res_xreg["df_train"],
        "df_train_aug": res_xreg["df_train_aug"],
        "months_head_pad": res_xreg["months_head_pad"],
        "months_tail_pad": res_xreg["months_tail_pad"],
    }

    logging.info(
        f"CH train rows: {len(res_train['df_train'])} | "
        f"Aug rows: {len(res_train['df_train_aug'])}"
    )

    ds_train, train_idx, val_idx, ds_scalers = build_or_load_lstm_train_only(
        cfg=cfg,
        key_train=key_train,
        res_train=res_train,
        MONTHLY_COLS=MONTHLY_COLS,
        STATIC_COLS=STATIC_COLS,
        val_ratio=val_ratio,
        cache_dir=cache_dir,
        force_recompute=force_recompute_train,
        normalize_target=normalize_target,
        expect_target=expect_target,
        strict_nan=strict_nan,
    )

    logging.info(
        f"CH train dataset size: {len(ds_train)} | "
        f"Train split: {len(train_idx)} | Val split: {len(val_idx)}"
    )

    # ---- helper to build one test entry ----
    def _build_test_entry(key_test, df_test_sc, df_test_aug_sc, label):
        logging.info("\n" + "-" * 50)
        logging.info(f"Target: {label}")
        logging.info(f"Cache key (test): {key_test}")
        logging.info(f"Test rows: {len(df_test_sc)} | Aug rows: {len(df_test_aug_sc)}")

        if len(df_test_sc) == 0 or len(df_test_aug_sc) == 0:
            logging.warning(f"Skipping {label}: no usable test rows.")
            return {
                "ds_train": ds_train,
                "ds_test": None,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "note": f"No test rows for {label}",
            }

        res_sc = {
            "df_test": df_test_sc,
            "df_test_aug": df_test_aug_sc,
            "months_head_pad": res_xreg["months_head_pad"],
            "months_tail_pad": res_xreg["months_tail_pad"],
        }

        ds_test = build_or_load_lstm_test_only(
            cfg=cfg,
            key_test=key_test,
            res_test=res_sc,
            MONTHLY_COLS=MONTHLY_COLS,
            STATIC_COLS=STATIC_COLS,
            cache_dir=cache_dir,
            force_recompute=force_recompute_tests,
            normalize_target=normalize_target,
            expect_target=expect_target,
            strict_nan=strict_nan,
        )

        logging.info(f"Test dataset size (sequences): {len(ds_test)}")

        return {
            "ds_train": ds_train,
            "ds_test": ds_test,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "cache_keys": {
                "train": key_train,
                "test": key_test,
            },
        }

    # ---- tests: individual regions ----
    logging.info("\n--- INDIVIDUAL TARGET REGION TEST DATASETS ---")

    outputs = {}
    only_set = set(only_test_keys) if only_test_keys else None

    for sc in target_source_codes:
        key_test = f"XREG_CH_TO_{sc}"

        if only_set is not None:
            if sc not in only_set and key_test not in only_set:
                logging.info(f"Skipping {sc} (not in only_test_keys).")
                continue

        df_test_sc = df_test_all.loc[df_test_all[source_col] == sc].copy()
        df_test_aug_sc = df_test_aug_all.loc[df_test_aug_all[source_col] == sc].copy()

        outputs[key_test] = _build_test_entry(
            key_test=key_test,
            df_test_sc=df_test_sc,
            df_test_aug_sc=df_test_aug_sc,
            label=sc,
        )

    # ---- tests: grouped regions ----
    if region_groups:
        logging.info("\n--- GROUPED TARGET REGION TEST DATASETS ---")

    for group_name, codes in region_groups.items():
        key_test = f"XREG_CH_TO_{group_name}"

        if only_set is not None:
            if group_name not in only_set and key_test not in only_set:
                logging.info(f"Skipping group {group_name} (not in only_test_keys).")
                continue

        mask = df_test_all[source_col].isin(codes)
        mask_aug = df_test_aug_all[source_col].isin(codes)

        df_test_group = df_test_all.loc[mask].copy()
        df_test_aug_group = df_test_aug_all.loc[mask_aug].copy()

        logging.info(
            f"Group '{group_name}': pooled {len(df_test_group)} rows " f"from {codes}."
        )

        outputs[key_test] = _build_test_entry(
            key_test=key_test,
            df_test_sc=df_test_group,
            df_test_aug_sc=df_test_aug_group,
            label=f"{group_name} ({', '.join(codes)})",
        )

    logging.info("\nFinished cross-regional LSTM dataset preparation.")
    logging.info("=" * 60 + "\n")

    return outputs
