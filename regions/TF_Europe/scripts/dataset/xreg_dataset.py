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


# # ---- Optional thin wrappers for backwards compatibility ----
# def prepare_monthly_df_xreg_CH_to_EU(*args, **kwargs):
#     return prepare_monthly_df_xreg_SOURCE_to_EU(*args, source_code="CH", **kwargs)


# def prepare_monthly_df_xreg_NOR_to_EU(*args, **kwargs):
#     return prepare_monthly_df_xreg_SOURCE_to_EU(*args, source_code="NOR", **kwargs)


# def prepare_monthly_df_xreg_ISL_to_EU(*args, **kwargs):
#     return prepare_monthly_df_xreg_SOURCE_to_EU(*args, source_code="ISL", **kwargs)


import os
import re
import logging
from collections.abc import Iterable


def _parse_region_group(code_or_group):
    """
    Parse a region specification into:
      - label: string used for naming/logging
      - codes: list[str] of normalized region codes

    Accepted inputs
    ---------------
    - "CH"
    - ["FR", "CH", "IT_AT"]
    - ("FR", "CH", "IT_AT")
    - "CEU=[FR; CH; IT_AT]"
    - "CEU=[FR, CH, IT_AT]"

    Returns
    -------
    label : str
        Human-readable label, e.g. "CH" or "CEU"
    codes : list[str]
        Normalized region codes, e.g. ["FR", "CH", "IT_AT"]
    """
    if isinstance(code_or_group, str):
        s = code_or_group.strip()

        # Case 1: group syntax like CEU=[FR; CH; IT_AT]
        m = re.fullmatch(r"([A-Za-z0-9_]+)\s*=\s*\[(.*?)\]", s)
        if m:
            label = m.group(1).strip().upper()
            inside = m.group(2).strip()

            # split on ; or ,
            parts = [p.strip().upper() for p in re.split(r"[;,]", inside) if p.strip()]
            if not parts:
                raise ValueError(f"Empty region group: {code_or_group}")
            return label, parts

        # Case 2: plain single code
        return s.upper(), [s.upper()]

    # Case 3: iterable of codes
    if isinstance(code_or_group, Iterable):
        codes = [str(x).strip().upper() for x in code_or_group if str(x).strip()]
        if not codes:
            raise ValueError("Empty region code iterable")
        label = "_".join(codes)
        return label, codes

    raise TypeError(
        "source_code/target_code must be a string, list, or tuple of region codes"
    )


def _get_glaciers_for_codes(df_all, region_codes, verbose=False):
    """
    Return the union of glaciers belonging to the provided region codes.

    Parameters
    ----------
    df_all : pd.DataFrame
        Full cross-regional dataframe.
    region_codes : list[str]
        Region codes, e.g. ["FR", "CH", "IT_AT"].

    Returns
    -------
    glaciers : list
        Sorted unique glacier IDs belonging to any of the given codes.
    """
    glaciers = set()

    for code in region_codes:
        region_glaciers, _ = compute_xreg_test_glaciers(
            df_all,
            target_code=code,
            verbose=verbose,
        )
        glaciers.update(region_glaciers)

    return sorted(glaciers)


def prepare_monthly_df_xreg_pairwise(
    cfg,
    dfs,
    paths,
    vois_climate,
    vois_topographical,
    source_code,  # e.g. "CH" or "CEU=[FR; CH; IT_AT]" or ["FR", "CH", "IT_AT"]
    target_code,  # e.g. "ISL" or ["NOR", "SJM"]
    run_flag: bool = True,
    region_name: str | None = None,
    region_id: int = 11,
    csv_subfolder: str | None = None,
):
    """
    Build ONE monthly-prepped dataset for a directed source→target experiment,
    allowing source and/or target to be a combination of multiple countries.

    Examples
    --------
    Single region to single region:
        source_code = "CH"
        target_code = "ISL"

    Group to single region:
        source_code = "CEU=[FR; CH; IT_AT]"
        target_code = "ISL"

    Group to group:
        source_code = ["FR", "CH", "IT_AT"]
        target_code = ["NOR", "SJM"]

    Behaviour
    ---------
    - data  = concatenation of all Europe sources (as provided by `dfs`)
    - train = glaciers from all source regions
    - test  = glaciers from all target regions

    Returns
    -------
    res : dict
        Output dict from prepare_monthly_dfs_with_padding
        (df_train/df_test/aug/etc.)
    split_info : dict
        {
            "source_label": ...,
            "target_label": ...,
            "source_codes": [...],
            "target_codes": [...],
            "train_glaciers": [...],
            "test_glaciers": [...]
        }
    """
    source_label, source_codes = _parse_region_group(source_code)
    target_label, target_codes = _parse_region_group(target_code)

    overlap = set(source_codes) & set(target_codes)
    if overlap:
        raise ValueError(
            f"source and target region codes must be disjoint, but overlap on: {sorted(overlap)}"
        )

    if region_name is None:
        region_name = f"XREG_{source_label}_TO_{target_label}"

    if csv_subfolder is None:
        csv_subfolder = f"CrossRegional/{source_label}_to_{target_label}/csv"

    # 1) Concatenate all raw stake rows
    df_all = build_xreg_df_eu(dfs)

    # 2) Pick glaciers belonging to each source/target region set
    train_glaciers = _get_glaciers_for_codes(
        df_all,
        source_codes,
        verbose=False,
    )
    test_glaciers = _get_glaciers_for_codes(
        df_all,
        target_codes,
        verbose=False,
    )

    # Safety check: no glacier should be in both train and test
    overlap_glaciers = set(train_glaciers) & set(test_glaciers)
    if overlap_glaciers:
        raise ValueError(
            f"Train/test glacier overlap detected: {sorted(overlap_glaciers)}"
        )

    # Filter to only relevant glaciers
    df_sub = df_all[df_all["GLACIER"].isin(train_glaciers + test_glaciers)].copy()

    # 3) Output folder
    paths_ = paths.copy()
    paths_["csv_path"] = os.path.join(
        cfg.dataPath,
        path_PMB_WGMS_csv,
        csv_subfolder,
    )
    os.makedirs(paths_["csv_path"], exist_ok=True)

    logging.info(
        f"Preparing pairwise cross-regional monthlies: {region_name} "
        f"(run_flag={run_flag}) | "
        f"train({source_label}={source_codes})={len(train_glaciers)} | "
        f"test({target_label}={target_codes})={len(test_glaciers)}"
    )

    res = prepare_monthly_dfs_with_padding(
        cfg=cfg,
        df_region=df_sub,
        region_name=region_name,
        region_id=int(region_id),
        paths=paths_,
        test_glaciers=test_glaciers,
        vois_climate=vois_climate,
        vois_topographical=vois_topographical,
        run_flag=run_flag,
    )

    return res, {
        "source_label": source_label,
        "target_label": target_label,
        "source_codes": source_codes,
        "target_codes": target_codes,
        "train_glaciers": train_glaciers,
        "test_glaciers": test_glaciers,
    }
