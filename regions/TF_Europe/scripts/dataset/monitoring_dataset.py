import numpy as np
import pandas as pd
import massbalancemachine as mbm

from regions.TF_Europe.scripts.dataset import (
    build_or_load_lstm_train_only,
    build_or_load_lstm_dataset_only,
    build_source_codes_for_dataset,
)


# --------------------------------- Monitoring experiments:
def build_static_tl_assets_source_and_holdout(
    cfg,
    res_xreg,
    target_code: str,  # "ISL"
    source_code: str,  # "CH"
    holdout_glaciers: set,  # fixed glacier IDs
    MONTHLY_COLS,
    STATIC_COLS,
    cache_dir=None,
    force_recompute=False,
    val_ratio=0.2,
    key_train=None,
    key_holdout=None,  # if None -> auto name
    show_progress=True,
):
    """
    Builds (or loads) assets that are constant across all (G,Y,M,seed) experiments:
      - source code (e.g., CH) pretrain dataset + split + scaler donor
      - fixed target holdout dataset (evaluation-only)
    """
    if key_train is None:
        key_train = f"TL_{source_code}_TRAIN"

    if cache_dir is None:
        cache_dir = f"logs/LSTM_cache_TL_{source_code}_ISL_experiment"

    if key_holdout is None:
        key_holdout = f"TL_{source_code}_to_{target_code}_HOLDOUT_FIXED"

    # ---- source code pretrain datasets + scaler donor
    res_train = {
        "df_train": res_xreg["df_train"],
        "df_train_aug": res_xreg["df_train_aug"],
        "months_head_pad": res_xreg["months_head_pad"],
        "months_tail_pad": res_xreg["months_tail_pad"],
    }

    ds_ch, train_idx, val_idx, ds_ch_scalers = build_or_load_lstm_train_only(
        cfg=cfg,
        key_train=key_train,
        res_train=res_train,
        MONTHLY_COLS=MONTHLY_COLS,
        STATIC_COLS=STATIC_COLS,
        val_ratio=val_ratio,
        cache_dir=cache_dir,
        force_recompute=force_recompute,
        show_progress=show_progress,
    )

    ch_source_codes = build_source_codes_for_dataset(
        ds_ch, res_xreg["df_train_aug"], source_col="SOURCE_CODE"
    )

    # ---- fixed holdout df (target region)
    df_target = (
        res_xreg["df_test"]
        .loc[res_xreg["df_test"]["SOURCE_CODE"] == target_code]
        .copy()
    )
    df_target_aug = (
        res_xreg["df_test_aug"]
        .loc[res_xreg["df_test_aug"]["SOURCE_CODE"] == target_code]
        .copy()
    )

    df_hold = df_target[df_target["GLACIER"].isin(holdout_glaciers)].copy()
    df_hold_aug = df_target_aug[df_target_aug["GLACIER"].isin(holdout_glaciers)].copy()

    if len(df_hold) == 0:
        raise ValueError(
            f"{target_code}: fixed holdout is empty. Check holdout_glaciers."
        )

    ds_holdout = build_or_load_lstm_dataset_only(
        cfg=cfg,
        key=key_holdout,
        df_loss=df_hold,
        df_full=df_hold_aug,
        months_head_pad=res_xreg["months_head_pad"],
        months_tail_pad=res_xreg["months_tail_pad"],
        MONTHLY_COLS=MONTHLY_COLS,
        STATIC_COLS=STATIC_COLS,
        cache_dir=cache_dir,
        force_recompute=force_recompute,
        kind="test",
        show_progress=show_progress,
    )

    holdout_source_codes = build_source_codes_for_dataset(
        ds_holdout, df_hold_aug, source_col="SOURCE_CODE"
    )

    static_assets = {
        "ds_pretrain": ds_ch,
        "ds_pretrain_scalers": ds_ch_scalers,
        "pretrain_train_idx": train_idx,
        "pretrain_val_idx": val_idx,
        "pretrain_source_codes": ch_source_codes,
        "ds_test": ds_holdout,
        "test_source_codes": holdout_source_codes,
        "target_code": target_code,
        "cache_keys": {
            "pretrain": key_train,
            "test": key_holdout,
        },
    }
    return static_assets


def _find_contiguous_blocks(years_sorted: np.ndarray, Y: int):
    """
    Return list of contiguous blocks (as arrays) of length Y from sorted unique years.
    Contiguous means consecutive integers.
    """
    years_sorted = np.array(years_sorted, dtype=int)
    if len(years_sorted) < Y:
        return []

    blocks = []
    for i in range(len(years_sorted) - Y + 1):
        w = years_sorted[i : i + Y]
        if np.all(np.diff(w) == 1):
            blocks.append(w)
    return blocks


def _choose_years_for_glacier(
    years_sorted: np.ndarray,
    Y: int,
    rng: np.random.Generator,
    year_pick_method: str,
):
    """
    Choose Y years for one glacier according to:
      - earliest_block: earliest contiguous block if exists, else earliest Y years (gappy allowed)
      - random_block: random contiguous block if exists, else earliest Y years
    """
    years_sorted = np.array(sorted(set(int(y) for y in years_sorted)), dtype=int)
    if len(years_sorted) == 0:
        return years_sorted

    if Y >= len(years_sorted):
        return years_sorted

    blocks = _find_contiguous_blocks(years_sorted, Y)

    if year_pick_method == "earliest_block":
        if blocks:
            return blocks[0]
        return years_sorted[:Y]

    if year_pick_method == "random_block":
        if blocks:
            return blocks[int(rng.integers(0, len(blocks)))]
        return years_sorted[:Y]

    raise ValueError("year_pick_method must be 'earliest_block' or 'random_block'")


def sample_monitoring_subset_from_pool(
    df_pool: pd.DataFrame,
    G: int,
    Y: int,
    seed: int = 0,
    glacier_pick_method: str = "random",  # "random" / "small_first" / "large_first" / "shuffle"
    year_pick_method: str = "earliest_block",  # "earliest_block" / "random_block"
    min_rows_per_glacier: int = 1,
    enforce_full_Y_if_possible: bool = True,
    glacier_col: str = "GLACIER",
    year_col: str = "YEAR",
):
    """
    Sample an LSTM-safe monitoring subset:
      1) pick G glaciers from the pool (optionally preferring glaciers with >= Y years)
      2) per glacier pick a contiguous block of Y years if possible (earliest or random),
         otherwise fall back to earliest Y available years (even if gappy)
      3) keep ALL rows (months) for the chosen glacier-years (no row subsampling)

    Returns
    -------
    df_y : pd.DataFrame
        Subset of df_pool (keeps original indices)
    chosen_set : set
        Set of chosen glacier IDs
    picked_years : dict
        Mapping glacier_id -> list of selected years
    """
    rng = np.random.default_rng(seed)

    # basic row filter (ensure glacier has at least some rows)
    counts = df_pool.groupby(glacier_col).size()
    ok_glaciers = counts[counts >= min_rows_per_glacier].index
    df0 = df_pool[df_pool[glacier_col].isin(ok_glaciers)]

    # compute number of unique years per glacier
    years_per_gl = (
        df0.groupby(glacier_col)[year_col].nunique().sort_values(ascending=False)
    )

    # candidate glaciers (prefer glaciers with >= Y years if possible)
    if enforce_full_Y_if_possible:
        candidates = years_per_gl[years_per_gl >= Y].index.to_numpy()
        if len(candidates) < G:
            candidates = years_per_gl.index.to_numpy()
    else:
        candidates = years_per_gl.index.to_numpy()

    if len(candidates) < G:
        raise ValueError(
            f"G={G} > available pool glaciers ({len(candidates)}) after filtering"
        )

    # choose glaciers
    if glacier_pick_method == "random":
        chosen = rng.choice(candidates, size=G, replace=False)
    elif glacier_pick_method == "small_first":
        chosen = (
            years_per_gl.loc[candidates]
            .sort_values(ascending=True)
            .index[:G]
            .to_numpy()
        )
    elif glacier_pick_method == "large_first":
        chosen = (
            years_per_gl.loc[candidates]
            .sort_values(ascending=False)
            .index[:G]
            .to_numpy()
        )
    elif glacier_pick_method == "shuffle":
        idx = candidates.copy()
        rng.shuffle(idx)
        chosen = idx[:G]
    else:
        raise ValueError(f"Unknown glacier_pick_method='{glacier_pick_method}'")

    chosen_set = set(chosen)
    df_g = df0[df0[glacier_col].isin(chosen_set)].copy()

    # pick years per glacier (contiguous block if possible; fallback to earliest Y years)
    keep_parts = []
    picked_years = {}

    for gid, dfgid in df_g.groupby(glacier_col):
        years_avail = dfgid[year_col].dropna().unique()

        y_keep = _choose_years_for_glacier(
            years_sorted=years_avail,
            Y=Y,
            rng=rng,
            year_pick_method=year_pick_method,
        )
        if len(y_keep) == 0:
            continue

        picked_years[gid] = list(map(int, y_keep))
        keep_parts.append(dfgid[dfgid[year_col].isin(y_keep)])

    if not keep_parts:
        return df_pool.iloc[0:0].copy(), chosen_set, picked_years

    # KEEP ORIGINAL INDICES (do NOT ignore_index=True)
    df_y = pd.concat(keep_parts, axis=0)
    return df_y, chosen_set, picked_years


def make_res_transfer_learning_custom(
    res_xreg: dict,
    target_code: str,
    df_ft: pd.DataFrame,
    holdout_glaciers: set,
    source_col="SOURCE_CODE",
):
    """
    Custom TL slicing:
      - pretrain: source_code (e.g., CH) from res_xreg
      - finetune: provided df_ft (+ its _aug subset)
      - test: fixed holdout glaciers (+ its _aug subset)
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

    # finetune aug: match the same keys as df_ft (GLACIER,YEAR,ID,PERIOD)
    key_cols = ["GLACIER", "YEAR", "ID", "PERIOD"]
    ft_keys = df_ft[key_cols].copy()
    ft_keys["PERIOD"] = ft_keys["PERIOD"].astype(str).str.strip().str.lower()

    df_target_aug2 = df_target_aug.copy()
    df_target_aug2["PERIOD"] = (
        df_target_aug2["PERIOD"].astype(str).str.strip().str.lower()
    )

    df_ft_aug = df_target_aug2.merge(
        ft_keys.drop_duplicates(), on=key_cols, how="inner"
    )

    # holdout = fixed glaciers
    df_hold = df_target[df_target["GLACIER"].isin(holdout_glaciers)].copy()
    df_hold_aug = df_target_aug2[
        df_target_aug2["GLACIER"].isin(holdout_glaciers)
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


def build_budget_assets_finetune_only(
    cfg,
    res_xreg,
    static_assets: dict,
    df_ft: pd.DataFrame,
    exp_key: str,
    MONTHLY_COLS,
    STATIC_COLS,
    cache_dir=None,
    force_recompute=False,
    val_ratio=0.2,
    show_progress=True,
):
    """
    Builds the only thing that varies per experiment: the finetune dataset + split.
    Then combines with static_assets into the final assets[exp_key] dict.
    """
    target_code = static_assets["target_code"]

    # target aug for extracting df_ft_aug
    df_target_aug = (
        res_xreg["df_test_aug"]
        .loc[res_xreg["df_test_aug"]["SOURCE_CODE"] == target_code]
        .copy()
    )
    df_target_aug["PERIOD"] = (
        df_target_aug["PERIOD"].astype(str).str.strip().str.lower()
    )

    # match aug rows to df_ft keys
    key_cols = ["GLACIER", "YEAR", "ID", "PERIOD"]
    ft_keys = df_ft[key_cols].copy()
    ft_keys["PERIOD"] = ft_keys["PERIOD"].astype(str).str.strip().str.lower()

    df_ft_aug = df_target_aug.merge(ft_keys.drop_duplicates(), on=key_cols, how="inner")

    if len(df_ft) == 0 or len(df_ft_aug) == 0:
        raise ValueError(f"{exp_key}: finetune df or aug df is empty.")

    # build finetune dataset (pristine)
    ft_cache_key = f"{exp_key}_FT"
    ds_ft = build_or_load_lstm_dataset_only(
        cfg=cfg,
        key=ft_cache_key,
        df_loss=df_ft,
        df_full=df_ft_aug,
        months_head_pad=res_xreg["months_head_pad"],
        months_tail_pad=res_xreg["months_tail_pad"],
        MONTHLY_COLS=MONTHLY_COLS,
        STATIC_COLS=STATIC_COLS,
        cache_dir=cache_dir,
        force_recompute=force_recompute,
        kind="ft",
        show_progress=show_progress,
    )

    ft_train_idx, ft_val_idx = mbm.data_processing.MBSequenceDataset.split_indices(
        len(ds_ft), val_ratio=val_ratio, seed=cfg.seed
    )

    ft_source_codes = build_source_codes_for_dataset(
        ds_ft, df_ft_aug, source_col="SOURCE_CODE"
    )

    # domain vocab: source code (e.g. CH) + FT + HOLDOUT
    domain_vocab = sorted(
        set(static_assets["pretrain_source_codes"])
        | set(ft_source_codes)
        | set(static_assets["test_source_codes"])
    )

    # assemble final experiment assets (same shape as before)
    assets = {
        exp_key: {
            **static_assets,
            "ds_finetune": ds_ft,
            "finetune_train_idx": ft_train_idx,
            "finetune_val_idx": ft_val_idx,
            "ft_source_codes": ft_source_codes,
            "domain_vocab": domain_vocab,
            "split_name": exp_key,  # optional
            "cache_keys": {
                **static_assets["cache_keys"],
                "finetune": ft_cache_key,
            },
        }
    }
    return assets
