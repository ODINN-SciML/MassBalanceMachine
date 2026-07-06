import logging
import os
import pandas as pd
import re
from collections.abc import Iterable
import pickle
import math

from regions.TF_Europe.scripts.config_TF_Europe import *
from regions.TF_Europe.scripts.dataset.data_loader import (
    prepare_monthly_dfs_with_padding,
    process_or_load_data,
    get_CV_splits,
)

from regions.TF_Europe.scripts.dataset.topoclimatic_distance import *

import massbalancemachine as mbm


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


def build_monthly_cache(
    cfg,
    dfs,
    paths_multi,
    vois_climate,
    vois_topographical,
    region_codes,
    run_flag_by_code,
    region_id=11,
    csv_subfolder="CrossRegional/monthly_cache",
):
    df_all = build_xreg_df_eu(dfs)

    # --- compute shared padding from the full combined aug df ---
    df_all_aug = df_all.copy()
    from_dt = pd.to_datetime(df_all_aug["FROM_DATE"].astype(str), format="%Y%m%d")
    to_dt = pd.to_datetime(df_all_aug["TO_DATE"].astype(str), format="%Y%m%d")
    aug_year = from_dt.dt.year.copy()
    aug_year = aug_year.where(from_dt.dt.year != to_dt.dt.year, aug_year - 1)
    df_all_aug["FROM_DATE"] = (aug_year.astype(str) + "0801").astype(int)

    global_head_pad, global_tail_pad = (
        mbm.data_processing.utils._compute_head_tail_pads_from_df(df_all_aug)
    )
    print(f"Global shared padding — head: {global_head_pad}, tail: {global_tail_pad}")

    # --- per-code processing ---
    cache = {}
    for code in region_codes:
        era5_source = REGION_CODE_TO_ERA5.get(code.upper(), "EU_US_CANADA")
        paths_ = _paths_for_code(code, paths_multi).copy()
        paths_["csv_path"] = os.path.join(
            cfg.dataPath, path_PMB_WGMS_csv, csv_subfolder, era5_source
        )
        os.makedirs(paths_["csv_path"], exist_ok=True)

        run_flag = run_flag_by_code.get(code.upper(), False)

        # get glaciers for this code only
        glaciers = _get_glaciers_for_codes(df_all, [code], verbose=False)
        df_sub = df_all[df_all["GLACIER"].isin(glaciers)].copy()

        logging.info(
            f"[cache] {code} ({era5_source}): {len(glaciers)} glaciers, "
            f"{len(df_sub)} rows, run_flag={run_flag}"
        )

        data_monthly = process_or_load_data(
            run_flag=run_flag,
            df=df_sub,
            paths=paths_,
            cfg=cfg,
            vois_climate=vois_climate,
            vois_topographical=vois_topographical,
            region_name=code,
            region_id=region_id,
            add_pcsr=False,
            output_file=f"{code}_wgms_dataset_monthly.csv",
        )

        # Aug-padded version
        df_sub_aug = df_sub.copy()
        from_dt = pd.to_datetime(df_sub_aug["FROM_DATE"].astype(str), format="%Y%m%d")
        to_dt = pd.to_datetime(df_sub_aug["TO_DATE"].astype(str), format="%Y%m%d")
        aug_year = from_dt.dt.year.copy()
        aug_year = aug_year.where(from_dt.dt.year != to_dt.dt.year, aug_year - 1)
        df_sub_aug["FROM_DATE"] = (aug_year.astype(str) + "0801").astype(int)

        head_pad, tail_pad = mbm.data_processing.utils._compute_head_tail_pads_from_df(
            df_sub_aug
        )

        data_monthly_aug = process_or_load_data(
            run_flag=run_flag,
            df=df_sub_aug,
            paths=paths_,
            cfg=cfg,
            vois_climate=vois_climate,
            vois_topographical=vois_topographical,
            region_name=code,
            region_id=region_id,
            add_pcsr=False,
            output_file=f"{code}_wgms_dataset_monthly_Aug.csv",
        )

        # add SOURCE_CODE
        data_monthly["SOURCE_CODE"] = code.upper()
        data_monthly_aug["SOURCE_CODE"] = code.upper()

        cache[code.upper()] = {
            "data_monthly": data_monthly,
            "data_monthly_aug": data_monthly_aug,
            "months_head_pad": global_head_pad,  # shared
            "months_tail_pad": global_tail_pad,  # shared
        }

    return cache


def prepare_xreg_pairs_from_cache(
    cfg,
    monthly_cache,  # output of build_monthly_cache
    xreg_pairs,  # list of (source_code, [target_codes])
):
    """
    For each (source, targets) pair, assemble train/test splits from
    pre-computed monthly dataframes. No ERA5 processing happens here.
    """
    res_by_source = {}
    split_info_by_source = {}

    for src_code, tgt_codes in xreg_pairs:
        src_code = src_code.upper()
        tgt_codes = [c.upper() for c in tgt_codes]

        print(f"\n{'='*50}")
        print(f"Source: {src_code} → Target: {tgt_codes}")

        # --- assemble monthly data from cache ---
        all_codes = [src_code] + tgt_codes
        missing = [c for c in all_codes if c not in monthly_cache]
        if missing:
            raise KeyError(f"Codes not in monthly cache: {missing}")

        data_monthly = pd.concat(
            [monthly_cache[c]["data_monthly"] for c in all_codes],
            ignore_index=True,
        )
        data_monthly_aug = pd.concat(
            [monthly_cache[c]["data_monthly_aug"] for c in all_codes],
            ignore_index=True,
        )

        # padding from source region (most meaningful for the experiment)
        head_pad = monthly_cache[src_code]["months_head_pad"]
        tail_pad = monthly_cache[src_code]["months_tail_pad"]

        # --- train/test glacier split ---
        train_glaciers = sorted(
            data_monthly[data_monthly["SOURCE_CODE"].str.upper() == src_code][
                "GLACIER"
            ].unique()
        )

        test_glaciers = sorted(
            data_monthly[data_monthly["SOURCE_CODE"].str.upper().isin(tgt_codes)][
                "GLACIER"
            ].unique()
        )

        overlap = set(train_glaciers) & set(test_glaciers)
        if overlap:
            raise ValueError(f"Train/test overlap for {src_code}: {sorted(overlap)}")

        def _make_split(data, test_gl):
            dataloader = mbm.dataloader.DataLoader(
                cfg, data=data, random_seed=cfg.seed, meta_data_columns=cfg.metaData
            )
            _, test_set, train_set = get_CV_splits(
                dataloader,
                test_split_on="GLACIER",
                test_splits=test_gl,
                random_state=cfg.seed,
            )
            df_tr = train_set["df_X"].copy()
            df_tr["y"] = train_set["y"]
            df_te = test_set["df_X"].copy()
            df_te["y"] = test_set["y"]
            return df_tr, df_te

        df_train, df_test = _make_split(data_monthly, test_glaciers)
        df_train_aug, df_test_aug = _make_split(data_monthly_aug, test_glaciers)

        existing = set(data_monthly["GLACIER"].unique())
        missing_test = [g for g in test_glaciers if g not in existing]

        res_by_source[src_code] = {
            "data_monthly": data_monthly,
            "df_train": df_train,
            "df_test": df_test,
            "data_monthly_aug": data_monthly_aug,
            "df_train_aug": df_train_aug,
            "df_test_aug": df_test_aug,
            "train_glaciers": train_glaciers,
            "missing_test_glaciers": missing_test,
            "months_head_pad": head_pad,
            "months_tail_pad": tail_pad,
        }
        split_info_by_source[src_code] = {
            "train_glaciers": train_glaciers,
            "test_glaciers": test_glaciers,
        }

        print(
            f"Train glaciers: {len(train_glaciers)}, Test glaciers: {len(test_glaciers)}"
        )
        print(f"Train rows: {len(df_train)}, Test rows: {len(df_test)}")

    return res_by_source, split_info_by_source


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


def _paths_for_code(code: str, paths_multi: dict) -> dict:
    """
    Resolve the correct ERA5 paths dict for a given region code.

    paths_multi must contain:
        "EU_US_CANADA": { "era5_climate_data": ..., "geopotential_data": ..., "csv_path": ... }
        "HMA":          { "era5_climate_data": ..., "geopotential_data": ..., "csv_path": ... }
    """
    source = REGION_CODE_TO_ERA5.get(code.upper(), "EU_US_CANADA")
    if source not in paths_multi:
        raise KeyError(
            f"No paths entry for ERA5 source '{source}' (code={code}). "
            f"Available: {list(paths_multi)}"
        )
    return paths_multi[source]


def prepare_monthly_df_xreg_pairwise(
    cfg,
    dfs,
    paths_multi,  # dict of {era5_source: paths_dict}  ← replaces `paths`
    vois_climate,
    vois_topographical,
    source_code,
    target_code,
    run_flag: bool = True,
    region_name: str | None = None,
    region_id: int = 11,
    csv_subfolder: str | None = None,
):
    source_label, source_codes = _parse_region_group(source_code)
    target_label, target_codes = _parse_region_group(target_code)

    overlap = set(source_codes) & set(target_codes)
    if overlap:
        raise ValueError(f"source/target overlap: {sorted(overlap)}")

    if region_name is None:
        region_name = f"XREG_{source_label}_TO_{target_label}"
    if csv_subfolder is None:
        csv_subfolder = f"CrossRegional/{source_label}_to_{target_label}/csv"

    df_all = build_xreg_df_eu(dfs)

    train_glaciers = _get_glaciers_for_codes(df_all, source_codes, verbose=False)
    test_glaciers = _get_glaciers_for_codes(df_all, target_codes, verbose=False)

    overlap_glaciers = set(train_glaciers) & set(test_glaciers)
    if overlap_glaciers:
        raise ValueError(f"Train/test glacier overlap: {sorted(overlap_glaciers)}")

    # --- Process each code group with its own ERA5 paths ---
    all_codes = source_codes + target_codes

    # Group codes by ERA5 source so we make one process_or_load_data call per source
    from itertools import groupby

    codes_by_source = {}
    for code in all_codes:
        source = REGION_CODE_TO_ERA5.get(code.upper(), "EU_US_CANADA")
        codes_by_source.setdefault(source, []).append(code)

    monthly_frames = []
    monthly_aug_frames = []
    months_head_pad_aug = months_tail_pad_aug = None

    for era5_source, codes in codes_by_source.items():
        paths_ = _paths_for_code(codes[0], paths_multi).copy()
        paths_["csv_path"] = os.path.join(
            cfg.dataPath,
            path_PMB_WGMS_csv,
            csv_subfolder,
            era5_source,  # separate csv cache per ERA5 source
        )
        os.makedirs(paths_["csv_path"], exist_ok=True)

        # Subset of glaciers belonging to this ERA5 source group
        glaciers_in_group = _get_glaciers_for_codes(df_all, codes, verbose=False)
        df_sub = df_all[df_all["GLACIER"].isin(glaciers_in_group)].copy()

        sub_name = f"{region_name}_{era5_source}"

        logging.info(
            f"  [{era5_source}] codes={codes}, glaciers={len(glaciers_in_group)}, "
            f"rows={len(df_sub)}, run_flag={run_flag}"
        )

        data_monthly = process_or_load_data(
            run_flag=run_flag,
            df=df_sub,
            paths=paths_,
            cfg=cfg,
            vois_climate=vois_climate,
            vois_topographical=vois_topographical,
            region_name=sub_name,
            region_id=region_id,
            add_pcsr=False,
            output_file=f"{sub_name}_wgms_dataset_monthly.csv",
        )
        monthly_frames.append(data_monthly)

        # Aug-padded version
        df_sub_aug = df_sub.copy()
        from_dt = pd.to_datetime(df_sub_aug["FROM_DATE"].astype(str), format="%Y%m%d")
        to_dt = pd.to_datetime(df_sub_aug["TO_DATE"].astype(str), format="%Y%m%d")
        aug_year = from_dt.dt.year.copy()
        aug_year = aug_year.where(from_dt.dt.year != to_dt.dt.year, aug_year - 1)
        df_sub_aug["FROM_DATE"] = (aug_year.astype(str) + "0801").astype(int)
        _head, _tail = mbm.data_processing.utils._compute_head_tail_pads_from_df(
            df_sub_aug
        )
        months_head_pad_aug = _head
        months_tail_pad_aug = _tail

        data_monthly_aug = process_or_load_data(
            run_flag=run_flag,
            df=df_sub_aug,
            paths=paths_,
            cfg=cfg,
            vois_climate=vois_climate,
            vois_topographical=vois_topographical,
            region_name=sub_name,
            region_id=region_id,
            add_pcsr=False,
            output_file=f"{sub_name}_wgms_dataset_monthly_Aug.csv",
        )
        print(data_monthly_aug)
        monthly_aug_frames.append(data_monthly_aug)

    # --- Merge and split ---
    data_monthly = pd.concat(monthly_frames, ignore_index=True)
    data_monthly_aug = pd.concat(monthly_aug_frames, ignore_index=True)

    def _make_split(data, test_gl):
        dataloader = mbm.dataloader.DataLoader(
            cfg, data=data, random_seed=cfg.seed, meta_data_columns=cfg.metaData
        )
        _, test_set, train_set = get_CV_splits(
            dataloader,
            test_split_on="GLACIER",
            test_splits=test_gl,
            random_state=cfg.seed,
        )
        df_tr = train_set["df_X"].copy()
        df_tr["y"] = train_set["y"]
        df_te = test_set["df_X"].copy()
        df_te["y"] = test_set["y"]
        return df_tr, df_te

    df_train, df_test = _make_split(data_monthly, test_glaciers)
    df_train_aug, df_test_aug = _make_split(data_monthly_aug, test_glaciers)

    existing = set(data_monthly["GLACIER"].unique())
    missing_test_glaciers = [g for g in test_glaciers if g not in existing]
    final_train_glaciers = sorted(existing - set(test_glaciers))

    return {
        "data_monthly": data_monthly,
        "df_train": df_train,
        "df_test": df_test,
        "data_monthly_aug": data_monthly_aug,
        "df_train_aug": df_train_aug,
        "df_test_aug": df_test_aug,
        "train_glaciers": final_train_glaciers,
        "missing_test_glaciers": missing_test_glaciers,
        "months_head_pad": months_head_pad_aug,
        "months_tail_pad": months_tail_pad_aug,
    }, {
        "source_label": source_label,
        "target_label": target_label,
        "source_codes": source_codes,
        "target_codes": target_codes,
        "train_glaciers": final_train_glaciers,
        "test_glaciers": test_glaciers,
    }


def prepare_within_region_pairs_from_cache(
    cfg,
    monthly_cache,
    within_codes,
    gl_level_split=None,
    holdout_frac=0.20,
    n_restarts=100,
    min_train_glaciers=3,
    checkpoint_path=None,
):
    gl_level_split = gl_level_split or []

    res_for_scalers = {
        code: {"df_train": monthly_cache[code]["data_monthly"]} for code in within_codes
    }
    scaler_m, scaler_s, _, _ = build_global_scalers_multi_source_simple(
        res_for_scalers,
        monthly_cols=MONTHLY_COLS,
        static_cols=STATIC_COLS,
    )
    _, _, blur_joint, _ = estimate_global_bandwidths_simple(
        res_for_scalers,
        monthly_cols=MONTHLY_COLS,
        static_cols=STATIC_COLS,
        scaler_m=scaler_m,
        scaler_s=scaler_s,
        seed=cfg.seed,
    )
    print(f"Estimated blur_joint={blur_joint:.4f}")

    # ── load checkpoint if it exists ─────────────────────────────────────────
    res_by_code = {}
    split_info_by_code = {}
    failed_codes = []

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        res_by_code = ckpt.get("res_by_code", {})
        split_info_by_code = ckpt.get("split_info_by_code", {})
        failed_codes = ckpt.get("failed_codes", [])
        already_done = set(res_by_code.keys()) | set(failed_codes)
        print(
            f"  Resuming from checkpoint: {len(res_by_code)} done, "
            f"{len(failed_codes)} failed, "
            f"{len(already_done)} skipped"
        )
    else:
        already_done = set()

    for code in within_codes:
        if code in already_done:
            print(f"  Skipping {code} (already in checkpoint)")
            continue

        print(f"\n{'='*50}\nWithin-region: {code}")

        try:
            if code not in monthly_cache:
                raise KeyError(f"Code not in monthly_cache: {code}")

            data_monthly = monthly_cache[code]["data_monthly"].copy()
            data_monthly_aug = monthly_cache[code]["data_monthly_aug"].copy()
            head_pad = monthly_cache[code]["months_head_pad"]
            tail_pad = monthly_cache[code]["months_tail_pad"]

            use_gl_split = code in gl_level_split

            # guard: fall back to ID-split if too few glaciers
            if use_gl_split:
                n_glaciers = data_monthly["GLACIER"].nunique()
                min_total = math.ceil(min_train_glaciers / (1 - holdout_frac))
                if n_glaciers < min_total:
                    print(
                        f"  ⚠ Only {n_glaciers} glacier(s) for {code} "
                        f"(need ≥{min_total}) — falling back to ID-split"
                    )
                    use_gl_split = False

            split_unit = "GLACIER" if use_gl_split else "ID"

            if use_gl_split:
                best_split = split_train_test_sinkhorn(
                    df_region=data_monthly,
                    monthly_cols=MONTHLY_COLS,
                    static_cols=STATIC_COLS,
                    scaler_m=scaler_m,
                    scaler_s=scaler_s,
                    blur_joint=blur_joint,
                    train_frac=1 - holdout_frac,
                    n_restarts=n_restarts,
                    frac_tol=0.05,
                    seed=cfg.seed,
                )
                best_score = best_split["best_score"]
            else:
                print(f"  Using ID-level split")
                best_split = split_pool_holdout_by_id(
                    df_region=data_monthly,
                    id_col="ID",
                    train_frac=1 - holdout_frac,
                    seed=cfg.seed,
                )
                best_score = float("nan")

            test_units = sorted(best_split["test_glaciers"])
            train_units = sorted(best_split["train_glaciers"])

            print(f"  Split on      : {split_unit}")
            print(f"  Train units   : {len(train_units)}")
            print(f"  Test  units   : {len(test_units)}")
            print(f"  Train frac    : {best_split['actual_train_frac']:.1%}")

            def _make_split(data, test_units, split_on):
                data = data.reset_index(drop=True)
                dataloader = mbm.dataloader.DataLoader(
                    cfg,
                    data=data,
                    random_seed=cfg.seed,
                    meta_data_columns=cfg.metaData,
                )
                _, test_set, train_set = get_CV_splits(
                    dataloader,
                    test_split_on=split_on,  # was: split_on=  ← wrong kwarg name
                    test_splits=test_units,  # was: test_glaciers=  ← wrong kwarg name
                    random_state=cfg.seed,
                )
                # Reconstruct full DataFrames (get_CV_splits returns dicts, not raw DFs)
                train_df = train_set["df_X"].copy()
                train_df["POINT_BALANCE"] = train_set["y"]
                test_df = test_set["df_X"].copy()
                test_df["POINT_BALANCE"] = test_set["y"]
                return train_df, test_df

            train_set, test_set = _make_split(data_monthly, test_units, split_unit)
            train_set_aug, test_set_aug = _make_split(
                data_monthly_aug, test_units, split_unit
            )

            res_by_code[code] = {
                "df_train": train_set,
                "df_test": test_set,
                "df_train_aug": train_set_aug,
                "df_test_aug": test_set_aug,
                "months_head_pad": head_pad,  # renamed from "head_pad"
                "months_tail_pad": tail_pad,  # renamed from "tail_pad"
                "scaler_m": scaler_m,
                "scaler_s": scaler_s,
                "blur_joint": blur_joint,
            }
            split_info_by_code[code] = {
                "train_glaciers": train_units,
                "test_glaciers": test_units,
                "actual_train_frac": best_split["actual_train_frac"],
                "best_score": best_score,
                "split_unit": split_unit,
            }

        except Exception as e:
            print(f"  ✗ Failed for {code}: {e}")
            failed_codes.append(code)

        finally:
            if checkpoint_path is not None:
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(
                        {
                            "res_by_code": res_by_code,
                            "split_info_by_code": split_info_by_code,
                            "failed_codes": failed_codes,
                        },
                        f,
                    )

    if failed_codes:
        print(f"\n⚠ Failed regions: {failed_codes}")

    return res_by_code, split_info_by_code


def prepare_within_region_random_splits(
    cfg,
    monthly_cache,
    within_codes,
    gl_level_split=None,
    holdout_frac=0.20,
    min_train_glaciers=3,
    n_random_splits=5,
    checkpoint_path=None,
):
    """
    Same as prepare_within_region_pairs_from_cache, but uses plain random
    train/test splits (no Sinkhorn optimization) as a baseline comparison.

    Returns
    -------
    res_by_code : dict
        {code: [res_dict_split0, res_dict_split1, ...]} — one entry per random seed.
    split_info_by_code : dict
        {code: [split_info_dict_split0, ...]}
    """
    gl_level_split = gl_level_split or []

    res_by_code = {}
    split_info_by_code = {}
    failed_codes = []

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        res_by_code = ckpt.get("res_by_code", {})
        split_info_by_code = ckpt.get("split_info_by_code", {})
        failed_codes = ckpt.get("failed_codes", [])
        already_done = set(res_by_code.keys()) | set(failed_codes)
        print(
            f"  Resuming from checkpoint: {len(res_by_code)} done, "
            f"{len(failed_codes)} failed, "
            f"{len(already_done)} skipped"
        )
    else:
        already_done = set()

    for code in within_codes:
        if code in already_done:
            print(f"  Skipping {code} (already in checkpoint)")
            continue

        print(f"\n{'='*50}\nWithin-region (random): {code}")

        try:
            if code not in monthly_cache:
                raise KeyError(f"Code not in monthly_cache: {code}")

            data_monthly = monthly_cache[code]["data_monthly"].copy()
            data_monthly_aug = monthly_cache[code]["data_monthly_aug"].copy()
            head_pad = monthly_cache[code]["months_head_pad"]
            tail_pad = monthly_cache[code]["months_tail_pad"]

            use_gl_split = code in gl_level_split

            # same fallback guard as the smart-split version, for consistency
            if use_gl_split:
                n_glaciers = data_monthly["GLACIER"].nunique()
                min_total = math.ceil(min_train_glaciers / (1 - holdout_frac))
                if n_glaciers < min_total:
                    print(
                        f"  ⚠ Only {n_glaciers} glacier(s) for {code} "
                        f"(need ≥{min_total}) — falling back to ID-split"
                    )
                    use_gl_split = False

            split_unit = "GLACIER" if use_gl_split else "ID"
            unit_col = "GLACIER" if use_gl_split else "ID"
            all_units = data_monthly[unit_col].unique()

            def _make_split(data, test_units, split_on):
                data = data.reset_index(drop=True)
                dataloader = mbm.dataloader.DataLoader(
                    cfg,
                    data=data,
                    random_seed=cfg.seed,
                    meta_data_columns=cfg.metaData,
                )
                _, test_set, train_set = get_CV_splits(
                    dataloader,
                    test_split_on=split_on,
                    test_splits=test_units,
                    random_state=cfg.seed,
                )
                train_df = train_set["df_X"].copy()
                train_df["POINT_BALANCE"] = train_set["y"]
                test_df = test_set["df_X"].copy()
                test_df["POINT_BALANCE"] = test_set["y"]
                return train_df, test_df

            res_list = []
            split_info_list = []

            for split_idx in range(n_random_splits):
                rng_seed = cfg.seed + split_idx
                rng = np.random.RandomState(rng_seed)

                shuffled = rng.permutation(all_units)
                n_test = max(1, round(len(shuffled) * holdout_frac))
                # ensure at least min_train_glaciers remain in train for GL splits
                n_test = (
                    min(n_test, len(shuffled) - min_train_glaciers)
                    if use_gl_split
                    else n_test
                )
                n_test = max(n_test, 1)

                test_units = sorted(shuffled[:n_test])
                train_units = sorted(shuffled[n_test:])

                actual_train_frac = len(train_units) / len(shuffled)

                print(
                    f"  [random split {split_idx}] seed={rng_seed} | "
                    f"train={len(train_units)} test={len(test_units)} "
                    f"({actual_train_frac:.1%} train)"
                )

                train_set, test_set = _make_split(data_monthly, test_units, split_unit)
                train_set_aug, test_set_aug = _make_split(
                    data_monthly_aug, test_units, split_unit
                )

                res_list.append(
                    {
                        "df_train": train_set,
                        "df_test": test_set,
                        "df_train_aug": train_set_aug,
                        "df_test_aug": test_set_aug,
                        "months_head_pad": head_pad,
                        "months_tail_pad": tail_pad,
                    }
                )
                split_info_list.append(
                    {
                        "train_glaciers": train_units,
                        "test_glaciers": test_units,
                        "actual_train_frac": actual_train_frac,
                        "split_unit": split_unit,
                        "seed": rng_seed,
                    }
                )

            res_by_code[code] = res_list
            split_info_by_code[code] = split_info_list

        except Exception as e:
            print(f"  ✗ Failed for {code}: {e}")
            failed_codes.append(code)

        finally:
            if checkpoint_path is not None:
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(
                        {
                            "res_by_code": res_by_code,
                            "split_info_by_code": split_info_by_code,
                            "failed_codes": failed_codes,
                        },
                        f,
                    )

    if failed_codes:
        print(f"\n⚠ Failed regions: {failed_codes}")

    return res_by_code, split_info_by_code
