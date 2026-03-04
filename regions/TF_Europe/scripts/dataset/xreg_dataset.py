import logging
import os
import pandas as pd

from regions.TF_Europe.scripts.config_TF_Europe import *
from regions.TF_Europe.scripts.dataset.data_loader import (
    prepare_monthly_dfs_with_padding,
)


def build_xreg_df_eu(dfs: dict) -> pd.DataFrame:
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
    verbose=True,
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

    if verbose:
        logging.info(
            f"Cross-regional split: {target_code} train glaciers={len(target_gl)}, non-{target_code} test glaciers={len(non_target_gl)}"
        )
    return target_gl, non_target_gl


# def prepare_monthly_df_xreg_CH_to_EU(
#     cfg,
#     dfs,
#     paths,
#     vois_climate,
#     vois_topographical,
#     run_flag=True,  # True recompute, False load
#     region_name="XREG_CH_TO_EU",
#     region_id=11,  # arbitrary/int tag used by your pipeline; keep 11 or 0
#     csv_subfolder="CrossRegional/CH_to_Europe/csv",
# ):
#     """
#     Build ONE monthly-prepped dataset:
#       - data = concatenation of all Europe sources
#       - train = CH glaciers
#       - test  = all non-CH glaciers

#     Returns
#     -------
#     res : dict
#         Same output dict as prepare_monthly_dfs_with_padding (df_train/df_test/aug/etc.)
#     split_info : dict
#         {"train_glaciers": [...], "test_glaciers": [...]}
#     """

#     # 1) Concatenate all raw stake rows
#     df_all = build_xreg_df_eu(dfs)

#     # 2) Define test glaciers: all non-CH
#     train_glaciers, test_glaciers = compute_xreg_test_glaciers(df_all, target_code="CH")

#     # 3) Choose an output folder for this experiment
#     paths_ = paths.copy()
#     paths_["csv_path"] = os.path.join(cfg.dataPath, path_PMB_WGMS_csv, csv_subfolder)
#     os.makedirs(paths_["csv_path"], exist_ok=True)

#     logging.info(
#         f"Preparing cross-regional monthlies: {region_name} "
#         f"(run_flag={run_flag}) | train(CH)={len(train_glaciers)} | test(non-CH)={len(test_glaciers)}"
#     )

#     res = prepare_monthly_dfs_with_padding(
#         cfg=cfg,
#         df_region=df_all,
#         region_name=region_name,
#         region_id=int(region_id),
#         paths=paths_,
#         test_glaciers=test_glaciers,  # test = all non-CH glaciers
#         vois_climate=vois_climate,
#         vois_topographical=vois_topographical,
#         run_flag=run_flag,
#     )

#     return res, {"train_glaciers": train_glaciers, "test_glaciers": test_glaciers}

# def prepare_monthly_df_xreg_NOR_to_EU(
#     cfg,
#     dfs,
#     paths,
#     vois_climate,
#     vois_topographical,
#     run_flag=True,  # True recompute, False load
#     region_name="XREG_NOR_TO_EU",
#     region_id=11,  # arbitrary/int tag used by your pipeline; keep 11 or 0
#     csv_subfolder="CrossRegional/NOR_to_Europe/csv",
# ):
#     """
#     Build ONE monthly-prepped dataset:
#       - data = concatenation of all Europe sources
#       - train = NOR glaciers
#       - test  = all non-NOR glaciers

#     Returns
#     -------
#     res : dict
#         Same output dict as prepare_monthly_dfs_with_padding (df_train/df_test/aug/etc.)
#     split_info : dict
#         {"train_glaciers": [...], "test_glaciers": [...]}
#     """

#     # 1) Concatenate all raw stake rows
#     df_all = build_xreg_df_eu(dfs)

#     # 2) Define test glaciers: all non-NOR
#     train_glaciers, test_glaciers = compute_xreg_test_glaciers(
#         df_all, target_code="NOR"
#     )

#     # 3) Choose an output folder for this experiment
#     paths_ = paths.copy()
#     paths_["csv_path"] = os.path.join(cfg.dataPath, path_PMB_WGMS_csv, csv_subfolder)
#     os.makedirs(paths_["csv_path"], exist_ok=True)

#     logging.info(
#         f"Preparing cross-regional monthlies: {region_name} "
#         f"(run_flag={run_flag}) | train(NOR)={len(train_glaciers)} | test(non-NOR)={len(test_glaciers)}"
#     )

#     res = prepare_monthly_dfs_with_padding(
#         cfg=cfg,
#         df_region=df_all,
#         region_name=region_name,
#         region_id=int(region_id),
#         paths=paths_,
#         test_glaciers=test_glaciers,  # test = all non-NOR glaciers
#         vois_climate=vois_climate,
#         vois_topographical=vois_topographical,
#         run_flag=run_flag,
#     )

#     return res, {"train_glaciers": train_glaciers, "test_glaciers": test_glaciers}


def prepare_monthly_df_xreg_SOURCE_to_EU(
    cfg,
    dfs,
    paths,
    vois_climate,
    vois_topographical,
    source_code: str,  # e.g. "CH", "NOR", "ISL"
    run_flag: bool = True,  # True recompute, False load
    region_name: str | None = None,  # default: f"XREG_{source_code}_TO_EU"
    region_id: int = 11,  # arbitrary/int tag used by your pipeline
    csv_subfolder: (
        str | None
    ) = None,  # default: f"CrossRegional/{source_code}_to_Europe/csv"
):
    """
    Build ONE monthly-prepped dataset for cross-regional experiments:
      - data  = concatenation of all Europe sources (as provided by `dfs`)
      - train = glaciers from `source_code`
      - test  = all non-`source_code` glaciers

    Parameters
    ----------
    source_code:
        Source region code used for training split (e.g. "CH", "NOR", "ISL").
        The test set becomes all glaciers NOT in this code.

    Returns
    -------
    res : dict
        Same output dict as prepare_monthly_dfs_with_padding (df_train/df_test/aug/etc.)
    split_info : dict
        {"train_glaciers": [...], "test_glaciers": [...]}
    """

    source_code = str(source_code).strip().upper()
    if region_name is None:
        region_name = f"XREG_{source_code}_TO_EU"
    if csv_subfolder is None:
        csv_subfolder = f"CrossRegional/{source_code}_to_Europe/csv"

    # 1) Concatenate all raw stake rows
    df_all = build_xreg_df_eu(dfs)

    # 2) Define split: train = source_code, test = non-source_code
    train_glaciers, test_glaciers = compute_xreg_test_glaciers(
        df_all, target_code=source_code
    )

    # 3) Choose an output folder for this experiment
    paths_ = paths.copy()
    paths_["csv_path"] = os.path.join(cfg.dataPath, path_PMB_WGMS_csv, csv_subfolder)
    os.makedirs(paths_["csv_path"], exist_ok=True)

    logging.info(
        f"Preparing cross-regional monthlies: {region_name} (run_flag={run_flag}) | "
        f"train({source_code})={len(train_glaciers)} | test(non-{source_code})={len(test_glaciers)}"
    )

    res = prepare_monthly_dfs_with_padding(
        cfg=cfg,
        df_region=df_all,
        region_name=region_name,
        region_id=int(region_id),
        paths=paths_,
        test_glaciers=test_glaciers,
        vois_climate=vois_climate,
        vois_topographical=vois_topographical,
        run_flag=run_flag,
    )

    return res, {"train_glaciers": train_glaciers, "test_glaciers": test_glaciers}


# ---- Optional thin wrappers for backwards compatibility ----
def prepare_monthly_df_xreg_CH_to_EU(*args, **kwargs):
    return prepare_monthly_df_xreg_SOURCE_to_EU(*args, source_code="CH", **kwargs)


def prepare_monthly_df_xreg_NOR_to_EU(*args, **kwargs):
    return prepare_monthly_df_xreg_SOURCE_to_EU(*args, source_code="NOR", **kwargs)


def prepare_monthly_df_xreg_ISL_to_EU(*args, **kwargs):
    return prepare_monthly_df_xreg_SOURCE_to_EU(*args, source_code="ISL", **kwargs)


def prepare_monthly_df_xreg_pairwise(
    cfg,
    dfs,
    paths,
    vois_climate,
    vois_topographical,
    source_code: str,  # e.g. "CH"
    target_code: str,  # e.g. "ISL"
    run_flag: bool = True,  # True recompute, False load
    region_name: str | None = None,  # default: f"XREG_{source_code}_TO_{target_code}"
    region_id: int = 11,
    csv_subfolder: (
        str | None
    ) = None,  # default: f"CrossRegional/{source_code}_to_{target_code}/csv"
):
    """
    Build ONE monthly-prepped dataset for a directed source→target experiment:
      - data  = concatenation of all Europe sources (as provided by `dfs`)
      - train = glaciers from `source_code`
      - test  = glaciers from `target_code` only

    Returns
    -------
    res : dict
        Output dict from prepare_monthly_dfs_with_padding (df_train/df_test/aug/etc.)
    split_info : dict
        {"train_glaciers": [...], "test_glaciers": [...]}
    """
    source_code = str(source_code).strip().upper()
    target_code = str(target_code).strip().upper()
    if source_code == target_code:
        raise ValueError("source_code and target_code must be different")

    if region_name is None:
        region_name = f"XREG_{source_code}_TO_{target_code}"
    if csv_subfolder is None:
        csv_subfolder = f"CrossRegional/{source_code}_to_{target_code}/csv"

    # 1) Concatenate all raw stake rows
    df_all = build_xreg_df_eu(dfs)

    # 2) Pick glaciers belonging to each region
    # Assumption: compute_xreg_test_glaciers(df_all, target_code=CODE)
    # returns (train_glaciers_for_CODE, test_glaciers_non_CODE)
    train_glaciers_src, _ = compute_xreg_test_glaciers(
        df_all, target_code=source_code, verbose=False
    )
    train_glaciers_tgt, _ = compute_xreg_test_glaciers(
        df_all, target_code=target_code, verbose=False
    )

    # For pairwise, test glaciers are "the target glaciers"
    train_glaciers = train_glaciers_src
    test_glaciers = train_glaciers_tgt

    # filter df_all to only include glaciers in train_glaciers or test_glaciers
    df_sub = df_all[df_all["GLACIER"].isin(train_glaciers + test_glaciers)]

    # 3) Output folder
    paths_ = paths.copy()
    paths_["csv_path"] = os.path.join(cfg.dataPath, path_PMB_WGMS_csv, csv_subfolder)
    os.makedirs(paths_["csv_path"], exist_ok=True)

    logging.info(
        f"Preparing pairwise cross-regional monthlies: {region_name} (run_flag={run_flag}) | "
        f"train({source_code})={len(train_glaciers)} | test({target_code})={len(test_glaciers)}"
    )

    res = prepare_monthly_dfs_with_padding(
        cfg=cfg,
        df_region=df_sub,
        region_name=region_name,
        region_id=int(region_id),
        paths=paths_,
        test_glaciers=test_glaciers,  # test = target glaciers only
        vois_climate=vois_climate,
        vois_topographical=vois_topographical,
        run_flag=run_flag,
    )

    return res, {"train_glaciers": train_glaciers, "test_glaciers": test_glaciers}
