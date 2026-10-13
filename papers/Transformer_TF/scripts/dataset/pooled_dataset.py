"""Multi-region (pooled) training dataset of the Transformer grid search
(notebook `2.2 Transformer_grid_search`).

- `load_pool_holdout_splits`: glacier split of each source region into pool (training)
  and holdout glaciers, from `glacier_splits.json`.
- `build_pooled_assets_from_cache`: one pooled training dataset from the pool glaciers of
  all source regions, plus one holdout dataset per region (cached with joblib).
- `report_pooled_assets`: sanity checks of these datasets (sizes, NaNs, leakage).
"""

import os
import json

import joblib
import pandas as pd
import torch
import massbalancemachine as mbm

from papers.Transformer_TF.scripts.dataset.lstm_dataset_helpers import (
    build_combined_LSTM_dataset,
)


def load_pool_holdout_splits(save_path, verbose=True):
    """
    Read the per-region glacier splits from `glacier_splits.json`.

    In that file, the roles of the two glacier sets are reversed relative to their
    names, so 'pool_glaciers' and 'holdout_glaciers' (and the matching fraction and
    Sinkhorn distances) are swapped here.

    Returns {region: {"pool_glaciers", "holdout_glaciers", "actual_pool_frac",
    "sinkhorn_pool_vs_holdout", "sinkhorn_pool_vs_region",
    "sinkhorn_holdout_vs_region", "blur_joint", "best_seed", "best_score"}}.
    """
    with open(save_path, "r") as f:
        output = json.load(f)

    splits = {
        src_region: {
            "pool_glaciers": data["holdout_glaciers"],  # swapped
            "holdout_glaciers": data["pool_glaciers"],  # swapped
            "actual_pool_frac": 1.0 - data["actual_pool_frac"],
            "sinkhorn_pool_vs_holdout": data["sinkhorn_pool_vs_holdout"],
            "sinkhorn_pool_vs_region": data["sinkhorn_holdout_vs_region"],  # swapped
            "sinkhorn_holdout_vs_region": data["sinkhorn_pool_vs_region"],  # swapped
            "blur_joint": data["blur_joint"],
            "best_seed": data["best_seed"],
            "best_score": data["best_score"],
        }
        for src_region, data in output.items()
    }

    if verbose:
        print(f"Loaded splits from: {save_path}")
        for src_region, split in splits.items():
            print(f"\n{src_region}:")
            print(f"  Pool glaciers     : {split['pool_glaciers']}")
            print(f"  Holdout glaciers  : {split['holdout_glaciers']}")
            print(f"  Pool fraction     : {split['actual_pool_frac']:.1%}")
            print(f"  Sk(pool↔holdout)  : {split['sinkhorn_pool_vs_holdout']:.4f}")
    return splits


def build_pooled_assets_from_cache(
    res_xreg_by_source: dict,
    splits: dict,
    source_regions: list,
    monthly_cols: list,
    static_cols: list,
    cfg,
    seed: int = 42,
    val_ratio: float = 0.2,
    cache_dir: str = "logs/LSTM_cache_pooled",
    force_recompute: bool = False,
):
    """
    Build (or load from cache_dir) the pooled multi-region training dataset.

    For each source region, the holdout glaciers (splits[region]['holdout_glaciers'])
    are removed from the training data and stored as a separate holdout dataset. The
    remaining stakes of all regions are pooled into one dataset, split randomly into
    train/validation sequences (val_ratio).

    res_xreg_by_source : {region: monthly-cache entry with 'data_monthly',
        'data_monthly_aug', 'months_head_pad', 'months_tail_pad'}

    Returns
    -------
    ({"ds_train", "train_idx", "val_idx"}, {region: holdout dataset})
    """
    os.makedirs(cache_dir, exist_ok=True)

    # cache key encodes which regions are pooled
    regions_key = "_".join(sorted(source_regions))
    train_p = os.path.join(cache_dir, f"pooled_{regions_key}_train.joblib")
    split_p = os.path.join(cache_dir, f"pooled_{regions_key}_split.joblib")

    # one cache file per holdout region
    holdout_ps = {
        region: os.path.join(cache_dir, f"holdout_{region}.joblib")
        for region in source_regions
    }

    all_cached = (
        os.path.exists(train_p)
        and os.path.exists(split_p)
        and all(os.path.exists(p) for p in holdout_ps.values())
    )

    if all_cached and not force_recompute:
        print(f"Loading pooled assets from cache: {cache_dir}")
        ds_pooled = joblib.load(train_p)
        split = joblib.load(split_p)
        holdout_datasets = {region: joblib.load(p) for region, p in holdout_ps.items()}
        print(
            f"Pooled dataset: {len(ds_pooled)} sequences "
            f"({len(split['train_idx'])} train | {len(split['val_idx'])} val)"
        )
        for region, ds_h in holdout_datasets.items():
            print(f"  {region} holdout: {len(ds_h)} sequences")
        return (
            {
                "ds_train": ds_pooled,
                "train_idx": split["train_idx"],
                "val_idx": split["val_idx"],
            },
            holdout_datasets,
        )

    # --- build fresh ---
    print(f"Building pooled assets for regions: {source_regions}")
    train_dfs_loss, train_dfs_full = [], []
    holdout_datasets = {}
    months_head_pad = months_tail_pad = None

    for region in source_regions:
        df_loss = res_xreg_by_source[region]["data_monthly"]
        df_full = res_xreg_by_source[region]["data_monthly_aug"]
        months_head_pad = res_xreg_by_source[region]["months_head_pad"]
        months_tail_pad = res_xreg_by_source[region]["months_tail_pad"]

        holdout_glaciers = splits[region]["holdout_glaciers"]

        df_loss_train = df_loss[~df_loss["GLACIER"].isin(holdout_glaciers)]
        df_full_train = df_full[~df_full["GLACIER"].isin(holdout_glaciers)]
        df_loss_hold = df_loss[df_loss["GLACIER"].isin(holdout_glaciers)]
        df_full_hold = df_full[df_full["GLACIER"].isin(holdout_glaciers)]

        n_total = df_loss["GLACIER"].nunique()
        n_train = df_loss_train["GLACIER"].nunique()
        n_hold = df_loss_hold["GLACIER"].nunique()
        print(f"{region}: {n_total} glaciers → {n_train} train | {n_hold} holdout")

        train_dfs_loss.append(df_loss_train)
        train_dfs_full.append(df_full_train)

        ds_holdout = build_combined_LSTM_dataset(
            df_loss=df_loss_hold,
            df_full=df_full_hold,
            monthly_cols=monthly_cols,
            static_cols=static_cols,
            months_head_pad=months_head_pad,
            months_tail_pad=months_tail_pad,
            normalize_target=True,
            expect_target=True,
            show_progress=False,
        )
        holdout_datasets[region] = ds_holdout
        joblib.dump(ds_holdout, holdout_ps[region], compress=3)
        print(f"  Saved {region} holdout -> {holdout_ps[region]}")

    # --- pooled train ---
    df_pooled_loss = pd.concat(train_dfs_loss, ignore_index=True)
    df_pooled_full = pd.concat(train_dfs_full, ignore_index=True)

    print(f"\nPooled head pad: {months_head_pad}")
    print(f"Pooled tail pad: {months_tail_pad}")

    mbm.utils.seed_all(seed)

    ds_pooled = build_combined_LSTM_dataset(
        df_loss=df_pooled_loss,
        df_full=df_pooled_full,
        monthly_cols=monthly_cols,
        static_cols=static_cols,
        months_head_pad=months_head_pad,
        months_tail_pad=months_tail_pad,
        normalize_target=True,
        expect_target=True,
        show_progress=True,
    )

    train_idx, val_idx = mbm.data_processing.MBSequenceDataset.split_indices(
        len(ds_pooled), val_ratio=val_ratio, seed=seed
    )

    # --- save ---
    joblib.dump(ds_pooled, train_p, compress=3)
    joblib.dump({"train_idx": train_idx, "val_idx": val_idx}, split_p, compress=3)
    print(f"\nSaved pooled train  -> {train_p}")
    print(f"Saved split indices -> {split_p}")
    print(
        f"Pooled dataset: {len(ds_pooled)} sequences "
        f"({len(train_idx)} train | {len(val_idx)} val)"
    )

    return (
        {"ds_train": ds_pooled, "train_idx": train_idx, "val_idx": val_idx},
        holdout_datasets,
    )


def report_pooled_assets(
    pooled_assets,
    ds_holdout_by_region,
    res_xreg_by_source,
    splits,
    source_regions,
    monthly_cols,
    static_cols,
):
    """
    Print sanity checks of the pooled dataset: sizes, tensor shapes, unfitted
    scalers, NaN/Inf values, coverage of each region's training glaciers, absence
    of holdout glaciers (no leakage), holdout dataset sizes and winter/annual balance.
    """
    ds_pooled = pooled_assets["ds_train"]
    train_idx = pooled_assets["train_idx"]
    val_idx = pooled_assets["val_idx"]

    # ---- 1. Sizes ----
    print("=== Sizes ===")
    print(f"Pooled dataset total sequences : {len(ds_pooled)}")
    print(f"Train split                    : {len(train_idx)}")
    print(f"Val split                      : {len(val_idx)}")
    print(
        f"Sum matches total              : {len(train_idx) + len(val_idx) == len(ds_pooled)}"
    )

    # ---- 2. Tensor shapes ----
    print("\n=== Tensor shapes ===")
    print(
        f"Xm (monthly inputs) : {tuple(ds_pooled.Xm.shape)}  expected (N, T, {len(monthly_cols)})"
    )
    print(
        f"Xs (static inputs)  : {tuple(ds_pooled.Xs.shape)}  expected (N, {len(static_cols)})"
    )
    print(f"mv (valid mask)     : {tuple(ds_pooled.mv.shape)}")
    print(f"y  (targets)        : {tuple(ds_pooled.y.shape)}")

    # ---- 3. Pristine check (no scalers fitted yet) ----
    print("\n=== Scaler state (should all be None) ===")
    print(f"month_mean  : {ds_pooled.month_mean}")
    print(f"static_mean : {ds_pooled.static_mean}")
    print(f"y_mean      : {ds_pooled.y_mean}")

    # ---- 4. NaN / Inf check ----
    print("\n=== NaN / Inf check ===")
    for name, t in [("Xm", ds_pooled.Xm), ("Xs", ds_pooled.Xs), ("y", ds_pooled.y)]:
        n_nan = torch.isnan(t).sum().item()
        n_inf = torch.isinf(t).sum().item()
        status = "OK" if (n_nan == 0 and n_inf == 0) else "ISSUE"
        print(f"  {name}: NaNs={n_nan}, Infs={n_inf}  [{status}]")

    # ---- 5. Region coverage — all training glaciers appear in the pooled keys ----
    print("\n=== Region coverage in keys ===")
    glaciers_in_pooled = {k[0] for k in ds_pooled.keys}
    for region in source_regions:
        df_loss = res_xreg_by_source[region]["data_monthly"]
        train_glaciers = set(df_loss["GLACIER"].unique()) - set(
            splits[region]["holdout_glaciers"]
        )
        overlap = glaciers_in_pooled & train_glaciers
        print(
            f"  {region}: {len(overlap)} / {len(train_glaciers)} train glaciers present in pooled dataset"
        )

    # ---- 6. Holdout check — holdout glaciers must NOT appear in pooled ----
    print("\n=== Holdout isolation check ===")
    for region in source_regions:
        holdout_gls = set(splits[region]["holdout_glaciers"])
        leaked = glaciers_in_pooled & holdout_gls
        status = "OK" if len(leaked) == 0 else f"LEAK: {leaked}"
        print(f"  {region}: {status}")

    # ---- 7. Holdout datasets ----
    print("\n=== Holdout datasets ===")
    for region, ds_h in ds_holdout_by_region.items():
        n_w = ds_h.iw.sum().item()
        n_a = ds_h.ia.sum().item()
        holdout_gls = {k[0] for k in ds_h.keys}
        print(
            f"  {region}: {len(ds_h)} sequences ({n_w} winter | {n_a} annual) | "
            f"{len(holdout_gls)} glaciers"
        )

    # ---- 8. Winter / annual balance in pooled ----
    print("\n=== Winter / Annual balance in pooled dataset ===")
    n_w = ds_pooled.iw.sum().item()
    n_a = ds_pooled.ia.sum().item()
    print(f"  Winter : {n_w} ({100*n_w/len(ds_pooled):.1f}%)")
    print(f"  Annual : {n_a} ({100*n_a/len(ds_pooled):.1f}%)")
