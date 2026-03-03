import logging
import os
import pandas as pd

from regions.TF_Europe.scripts.config_TF_Europe import *
from regions.TF_Europe.scripts.dataset.data_loader import (
    prepare_monthly_dfs_with_padding,
)


def build_xreg_df_ceu_with_ch(dfs: dict) -> pd.DataFrame:
    """
    Concatenate all stake dataframes in `dfs` into one Europe-wide dataframe.

    Expects:
      - Each df has at least columns: GLACIER, YEAR, ID, PERIOD, MONTHS, POINT_BALANCE
      - Central Europe df includes SOURCE_CODE identifying CH/FR/IT_AT etc.

    Returns
    -------
    pd.DataFrame
        Combined dataframe (all rows across all RGI regions).
    """
    frames = []
    for rid, df in dfs.items():
        if df is None or len(df) == 0:
            logging.warning(f"RGI {rid}: empty, skipping in concat.")
            continue
        frames.append(df)

    if not frames:
        raise ValueError("No non-empty dataframes in dfs.")

    d_all = pd.concat(frames, ignore_index=True)
    return d_all


def compute_xreg_test_glaciers(
    df_all: pd.DataFrame,
    target_code: str = "CH",
    source_col: str = "SOURCE_CODE",
    glacier_col: str = "GLACIER",
):
    """
    Train glaciers = all glaciers with SOURCE_CODE == target_code
    Test glaciers  = all glaciers with SOURCE_CODE != target_code

    Returns
    -------
    (train_glaciers, test_glaciers) : (list[str], list[str])
    """
    if source_col not in df_all.columns:
        raise ValueError(
            f"Missing column {source_col}. Needed to separate CH vs others."
        )
    if glacier_col not in df_all.columns:
        raise ValueError(f"Missing column {glacier_col}.")

    target_gl = sorted(
        df_all.loc[df_all[source_col] == target_code, glacier_col].dropna().unique()
    )
    non_target_gl = sorted(
        df_all.loc[df_all[source_col] != target_code, glacier_col].dropna().unique()
    )

    if not target_gl:
        raise ValueError(
            f"No {target_code} glaciers found (SOURCE_CODE=={target_code})."
        )
    if not non_target_gl:
        raise ValueError(
            f"No non-{target_code} glaciers found (SOURCE_CODE!={target_code})."
        )

    logging.info(
        f"Cross-regional split: {target_code} train glaciers={len(target_gl)}, non-{target_code} test glaciers={len(non_target_gl)}"
    )
    return target_gl, non_target_gl


def prepare_monthly_df_xreg_CH_to_EU(
    cfg,
    dfs,
    paths,
    vois_climate,
    vois_topographical,
    run_flag=True,  # True recompute, False load
    region_name="XREG_CH_TO_EU",
    region_id=11,  # arbitrary/int tag used by your pipeline; keep 11 or 0
    csv_subfolder="CrossRegional/CH_to_Europe/csv",
):
    """
    Build ONE monthly-prepped dataset:
      - data = concatenation of all Europe sources
      - train = CH glaciers
      - test  = all non-CH glaciers

    Returns
    -------
    res : dict
        Same output dict as prepare_monthly_dfs_with_padding (df_train/df_test/aug/etc.)
    split_info : dict
        {"train_glaciers": [...], "test_glaciers": [...]}
    """

    # 1) Concatenate all raw stake rows
    df_all = build_xreg_df_ceu_with_ch(dfs)

    # 2) Define test glaciers: all non-CH
    train_glaciers, test_glaciers = compute_xreg_test_glaciers(df_all, target_code="CH")

    # 3) Choose an output folder for this experiment
    paths_ = paths.copy()
    paths_["csv_path"] = os.path.join(cfg.dataPath, path_PMB_WGMS_csv, csv_subfolder)
    os.makedirs(paths_["csv_path"], exist_ok=True)

    logging.info(
        f"Preparing cross-regional monthlies: {region_name} "
        f"(run_flag={run_flag}) | train(CH)={len(train_glaciers)} | test(non-CH)={len(test_glaciers)}"
    )

    res = prepare_monthly_dfs_with_padding(
        cfg=cfg,
        df_region=df_all,
        region_name=region_name,
        region_id=int(region_id),
        paths=paths_,
        test_glaciers=test_glaciers,  # test = all non-CH glaciers
        vois_climate=vois_climate,
        vois_topographical=vois_topographical,
        run_flag=run_flag,
    )

    return res, {"train_glaciers": train_glaciers, "test_glaciers": test_glaciers}


def prepare_monthly_df_xreg_NOR_to_EU(
    cfg,
    dfs,
    paths,
    vois_climate,
    vois_topographical,
    run_flag=True,  # True recompute, False load
    region_name="XREG_NOR_TO_EU",
    region_id=11,  # arbitrary/int tag used by your pipeline; keep 11 or 0
    csv_subfolder="CrossRegional/NOR_to_Europe/csv",
):
    """
    Build ONE monthly-prepped dataset:
      - data = concatenation of all Europe sources
      - train = NOR glaciers
      - test  = all non-NOR glaciers

    Returns
    -------
    res : dict
        Same output dict as prepare_monthly_dfs_with_padding (df_train/df_test/aug/etc.)
    split_info : dict
        {"train_glaciers": [...], "test_glaciers": [...]}
    """

    # 1) Concatenate all raw stake rows
    df_all = build_xreg_df_ceu_with_ch(dfs)

    # 2) Define test glaciers: all non-NOR
    train_glaciers, test_glaciers = compute_xreg_test_glaciers(
        df_all, target_code="NOR"
    )

    # 3) Choose an output folder for this experiment
    paths_ = paths.copy()
    paths_["csv_path"] = os.path.join(cfg.dataPath, path_PMB_WGMS_csv, csv_subfolder)
    os.makedirs(paths_["csv_path"], exist_ok=True)

    logging.info(
        f"Preparing cross-regional monthlies: {region_name} "
        f"(run_flag={run_flag}) | train(NOR)={len(train_glaciers)} | test(non-NOR)={len(test_glaciers)}"
    )

    res = prepare_monthly_dfs_with_padding(
        cfg=cfg,
        df_region=df_all,
        region_name=region_name,
        region_id=int(region_id),
        paths=paths_,
        test_glaciers=test_glaciers,  # test = all non-NOR glaciers
        vois_climate=vois_climate,
        vois_topographical=vois_topographical,
        run_flag=run_flag,
    )

    return res, {"train_glaciers": train_glaciers, "test_glaciers": test_glaciers}
