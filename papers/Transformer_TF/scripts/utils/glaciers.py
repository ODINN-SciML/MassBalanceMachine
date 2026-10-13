import numpy as np
import pandas as pd
import geopandas as gpd

from papers.Transformer_TF.scripts.dataset import load_stakes_for_rgi_region
from papers.Transformer_TF.scripts.config_TF_Europe import *


def pick_glaciers_by_row_fraction(
    df_test: pd.DataFrame,
    region_code: str,
    target_frac: float,
    source_col: str = "SOURCE_CODE",
    glacier_col: str = "GLACIER",
    seed: int = 42,
    method: str = "greedy_small_first",
    min_rows_per_glacier: int = 1,
):
    """
    Select glaciers whose df_test row counts sum to ~target_frac of total rows for region_code.

    method:
      - "greedy_small_first": sorts glaciers by row count ascending, then accumulates
        (often best for small targets like 5% because it can finely tune)
      - "greedy_large_first": sorts descending, then accumulates (often fine for 50%)
      - "shuffle_then_greedy": shuffle (seeded), then accumulate (stochastic but reproducible)

    Returns
    -------
    selected_glaciers : list[str]
    summary : dict with totals and achieved fraction
    per_glacier_counts : pd.Series of counts (for inspection)
    """
    df_reg = df_test.loc[df_test[source_col] == region_code].copy()
    if df_reg.empty:
        raise ValueError(
            f"No rows in df_test for region '{region_code}' (source_col={source_col})."
        )

    counts = df_reg.groupby(glacier_col).size().sort_values(ascending=False)

    # optional: remove tiny glaciers (if you want)
    counts = counts[counts >= min_rows_per_glacier]
    if counts.empty:
        raise ValueError(
            f"After filtering min_rows_per_glacier={min_rows_per_glacier}, no glaciers remain for {region_code}."
        )

    total_rows = int(counts.sum())
    target_rows = int(round(target_frac * total_rows))

    # order glaciers for greedy
    if method == "greedy_small_first":
        ordered = counts.sort_values(ascending=True)
    elif method == "greedy_large_first":
        ordered = counts.sort_values(ascending=False)
    elif method == "shuffle_then_greedy":
        rng = np.random.default_rng(seed)
        idx = counts.index.to_numpy()
        rng.shuffle(idx)
        ordered = counts.loc[idx]
    else:
        raise ValueError(f"Unknown method='{method}'")

    selected = []
    s = 0

    # greedy accumulate
    for gl, n in ordered.items():
        # if we already hit/exceeded target, decide whether adding this glacier helps or hurts
        if s >= target_rows:
            # check if adding would improve closeness
            cur_err = abs(s - target_rows)
            new_err = abs((s + int(n)) - target_rows)
            if new_err < cur_err:
                selected.append(gl)
                s += int(n)
            break
        else:
            selected.append(gl)
            s += int(n)

    # small local improvement: try swapping one glacier if it improves error (optional, cheap)
    # (helps especially near 50% targets)
    selected_set = set(selected)
    not_selected = [g for g in counts.index if g not in selected_set]

    best_err = abs(s - target_rows)
    best_swap = None

    # limit search for speed (still usually enough)
    cand_sel = selected[: min(len(selected), 40)]
    cand_nsel = not_selected[: min(len(not_selected), 60)]

    sel_counts = counts.loc[cand_sel]
    nsel_counts = counts.loc[cand_nsel]

    for g_out, n_out in sel_counts.items():
        for g_in, n_in in nsel_counts.items():
            s2 = s - int(n_out) + int(n_in)
            err2 = abs(s2 - target_rows)
            if err2 < best_err:
                best_err = err2
                best_swap = (g_out, g_in, s2)

    if best_swap is not None:
        g_out, g_in, s2 = best_swap
        selected = [g for g in selected if g != g_out] + [g_in]
        s = int(s2)

    achieved_frac = s / total_rows if total_rows > 0 else np.nan

    summary = {
        "region": region_code,
        "target_frac": float(target_frac),
        "total_rows_region": total_rows,
        "target_rows": target_rows,
        "selected_rows": int(s),
        "achieved_frac": float(achieved_frac),
        "achieved_pct": float(100 * achieved_frac),
        "n_glaciers_total": int(counts.shape[0]),
        "n_glaciers_selected": int(len(selected)),
        "abs_row_error": int(abs(s - target_rows)),
    }

    return selected, summary, counts


def verify_row_percentage(
    df_test,
    FT_GLACIERS,
    source_col="SOURCE_CODE",
    glacier_col="GLACIER",
    region_groups: dict | None = None,
):
    """
    Verify row percentages for FT glacier splits.

    Parameters
    ----------
    region_groups : dict or None
        Maps group name to list of SOURCE_CODE values, e.g.
        {"CEU": ["FR", "IT_AT", "CH"]}. When a region in FT_GLACIERS
        matches a group key, all member codes are pooled.
    """
    region_groups = region_groups or {}
    results = []

    for region, splits in FT_GLACIERS.items():

        if region in region_groups:
            member_codes = region_groups[region]
            df_reg = df_test[df_test[source_col].isin(member_codes)]
        else:
            df_reg = df_test[df_test[source_col] == region]

        total_rows = len(df_reg)
        if total_rows == 0:
            print(f"{region}: no rows in df_test")
            continue

        for split_name, glacier_list in splits.items():
            df_ft = df_reg[df_reg[glacier_col].isin(glacier_list)]
            ft_rows = len(df_ft)
            pct = 100 * ft_rows / total_rows

            results.append(
                {
                    "region": region,
                    "split": split_name,
                    "rows_total_region": total_rows,
                    "rows_ft": ft_rows,
                    "pct_rows": pct,
                }
            )

            print(
                f"{region} | {split_name}: " f"{ft_rows}/{total_rows} rows = {pct:.2f}%"
            )

    return pd.DataFrame(results)


def build_region_glacier_info(
    cfg,
    *,
    rgi_region_id: str,
    outline_shp_path: str,
    glacier_col="GLACIER",
    lat_col="POINT_LAT",
    lon_col="POINT_LON",
    period_col="PERIOD",
    nmeas_col="Nb. measurements",
    source_col="SOURCE_CODE",
    source_resolution="mode",
    o2region_col="O2Region",
    load_stakes_fn=None,
    verbose=True,
):
    """
    Generic builder for per-glacier info tables (for maps / summaries), for any region.
    No FT/holdout/train-test labeling — just location, measurement counts, source,
    and (if present) O2 sub-region (Central Asia / Alaska) or country override
    (Central Europe: IT/AT split via IT_GLACIERS/AT_GLACIERS).
    """

    if load_stakes_fn is None:
        load_stakes_fn = load_stakes_for_rgi_region

    data_region = load_stakes_fn(cfg, rgi_region_id)

    # --- O2Region split (Central Asia / Alaska only) ---
    if rgi_region_id == "13":
        dfs, glacier_dict = add_o2region_to_dfs({rgi_region_id: data_region}, "13", cfg)
        data_region = dfs[rgi_region_id]
    elif rgi_region_id == "01":
        dfs, glacier_dict = add_o2region_to_dfs(
            {rgi_region_id: data_region}, "01", cfg, deduplicate_glaciers=True
        )
        data_region = dfs[rgi_region_id]

    # --- Country override (Central Europe only: IT / AT) ---
    elif rgi_region_id == "11":
        data_region.loc[data_region[glacier_col].isin(IT_GLACIERS), source_col] = "IT"
        data_region.loc[data_region[glacier_col].isin(AT_GLACIERS), source_col] = "AT"
        if verbose:
            n_it = data_region[source_col].eq("IT").sum()
            n_at = data_region[source_col].eq("AT").sum()
            print(
                f"[{rgi_region_id}] Overrode {source_col}: IT -> {n_it} stake rows, "
                f"AT -> {n_at} stake rows."
            )

    glacier_outline_rgi = gpd.read_file(outline_shp_path)

    if verbose:
        print(
            f"[{rgi_region_id}] stake rows: {len(data_region)} | "
            f"glaciers: {data_region[glacier_col].nunique()}"
        )

    # --- measurement counts ---
    meas_counts = (
        data_region.groupby(glacier_col)
        .size()
        .sort_values(ascending=False)
        .rename(nmeas_col)
        .to_frame()
    )

    # --- mean location ---
    glacier_loc = data_region.groupby(glacier_col)[[lat_col, lon_col]].mean()

    # --- counts per period (winter/annual) ---
    glacier_period = (
        data_region.groupby([glacier_col, period_col])
        .size()
        .unstack()
        .fillna(0)
        .astype(int)
    )

    # --- SOURCE_CODE per glacier ---
    if source_col in data_region.columns:
        gsrc = data_region.groupby(glacier_col)[source_col].apply(
            lambda s: s.dropna().astype(str).unique()
        )

        mixed = gsrc[gsrc.apply(len) > 1]
        if len(mixed) > 0 and verbose:
            print(
                f"Warning: {len(mixed)} glaciers have multiple {source_col} values "
                f"(showing up to 5): {mixed.head(5).to_dict()}"
            )

        if len(mixed) > 0 and source_resolution == "error":
            raise ValueError(
                f"Found glaciers with multiple {source_col} values. "
                f"Set source_resolution to 'first', 'mode', or 'list' to resolve."
            )

        if source_resolution == "list":
            glacier_source = (
                gsrc.apply(lambda arr: list(arr)).rename(source_col).to_frame()
            )
        elif source_resolution == "first":
            glacier_source = (
                gsrc.apply(lambda arr: arr[0] if len(arr) else None)
                .rename(source_col)
                .to_frame()
            )
        elif source_resolution == "mode":

            def _mode(series):
                s = series.dropna().astype(str)
                if len(s) == 0:
                    return None
                return s.value_counts().index[0]

            glacier_source = (
                data_region.groupby(glacier_col)[source_col]
                .apply(_mode)
                .rename(source_col)
                .to_frame()
            )
        else:
            raise ValueError(
                "source_resolution must be one of: 'error','first','mode','list'"
            )
    else:
        glacier_source = None
        if verbose:
            print(
                f"Note: '{source_col}' not found in data_region; skipping SOURCE_CODE merge."
            )

    # --- O2Region per glacier (mode, in case of any inconsistency) ---
    if o2region_col in data_region.columns:

        def _mode(series):
            s = series.dropna().astype(str)
            if len(s) == 0:
                return None
            return s.value_counts().index[0]

        gregion = data_region.groupby(glacier_col)[o2region_col].apply(_mode)

        n_unique_per_glacier = data_region.groupby(glacier_col)[o2region_col].nunique()
        inconsistent = n_unique_per_glacier[n_unique_per_glacier > 1]
        if len(inconsistent) > 0 and verbose:
            print(
                f"Warning: {len(inconsistent)} glaciers have multiple {o2region_col} "
                f"values (using mode): {list(inconsistent.index[:5])}"
            )

        glacier_o2region = gregion.rename(o2region_col).to_frame()
    else:
        glacier_o2region = None
        if verbose:
            print(
                f"Note: '{o2region_col}' not found in data_region; skipping O2Region merge."
            )

    # --- merge base ---
    glacier_info = glacier_loc.join(meas_counts, how="inner").join(
        glacier_period, how="left"
    )
    if glacier_source is not None:
        glacier_info = glacier_info.join(glacier_source, how="left")
    if glacier_o2region is not None:
        glacier_info = glacier_info.join(glacier_o2region, how="left")

    return data_region, glacier_outline_rgi, glacier_info


def build_region_glacier_info_for_splits(
    cfg,
    *,
    rgi_region_id: str,
    outline_shp_path: str,
    ft_glaciers_by_split: dict,
    holdout_glaciers_by_split: dict,  # : explicit holdout per split
    split_names=("5pct", "50pct"),
    ft_label_col="FT/Hold-out glacier",
    ft_label_ft="FT",
    ft_label_holdout="Hold-out",
    ft_label_excluded="Excluded",  # : glaciers in neither set
    glacier_col="GLACIER",
    lat_col="POINT_LAT",
    lon_col="POINT_LON",
    period_col="PERIOD",
    nmeas_col="Nb. measurements",
    source_col="SOURCE_CODE",
    source_resolution="mode",
    load_stakes_fn=None,
    verbose=True,
):
    """
    Generic builder for per-glacier info tables (for maps / summaries), for any region + any splits.

    ft_glaciers_by_split and holdout_glaciers_by_split are both required.
    Glaciers in neither set are labelled ft_label_excluded ("Excluded") —
    these are pool glaciers not sampled for fine-tuning in a given split.
    """

    if load_stakes_fn is None:
        load_stakes_fn = load_stakes_for_rgi_region  # noqa: F821

    data_region = load_stakes_fn(cfg, rgi_region_id)
    glacier_outline_rgi = gpd.read_file(outline_shp_path)

    if verbose:
        print(
            f"[{rgi_region_id}] stake rows: {len(data_region)} | "
            f"glaciers: {data_region[glacier_col].nunique()}"
        )

    # --- measurement counts ---
    meas_counts = (
        data_region.groupby(glacier_col)
        .size()
        .sort_values(ascending=False)
        .rename(nmeas_col)
        .to_frame()
    )

    # --- mean location ---
    glacier_loc = data_region.groupby(glacier_col)[[lat_col, lon_col]].mean()

    # --- counts per period (winter/annual) ---
    glacier_period = (
        data_region.groupby([glacier_col, period_col])
        .size()
        .unstack()
        .fillna(0)
        .astype(int)
    )

    # --- SOURCE_CODE per glacier ---
    if source_col in data_region.columns:
        gsrc = data_region.groupby(glacier_col)[source_col].apply(
            lambda s: s.dropna().astype(str).unique()
        )

        mixed = gsrc[gsrc.apply(len) > 1]
        if len(mixed) > 0 and verbose:
            print(
                f"Warning: {len(mixed)} glaciers have multiple {source_col} values "
                f"(showing up to 5): {mixed.head(5).to_dict()}"
            )

        if len(mixed) > 0 and source_resolution == "error":
            raise ValueError(
                f"Found glaciers with multiple {source_col} values. "
                f"Set source_resolution to 'first', 'mode', or 'list' to resolve."
            )

        if source_resolution == "list":
            glacier_source = (
                gsrc.apply(lambda arr: list(arr)).rename(source_col).to_frame()
            )
        elif source_resolution == "first":
            glacier_source = (
                gsrc.apply(lambda arr: arr[0] if len(arr) else None)
                .rename(source_col)
                .to_frame()
            )
        elif source_resolution == "mode":

            def _mode(series):
                s = series.dropna().astype(str)
                if len(s) == 0:
                    return None
                return s.value_counts().index[0]

            glacier_source = (
                data_region.groupby(glacier_col)[source_col]
                .apply(_mode)
                .rename(source_col)
                .to_frame()
            )
        else:
            raise ValueError(
                "source_resolution must be one of: 'error','first','mode','list'"
            )
    else:
        glacier_source = None
        if verbose:
            print(
                f"Note: '{source_col}' not found in data_region; skipping SOURCE_CODE merge."
            )

    # --- merge base ---
    base = glacier_loc.join(meas_counts, how="inner").join(glacier_period, how="left")
    if glacier_source is not None:
        base = base.join(glacier_source, how="left")

    glacier_info_by_split = {}

    for split in split_names:
        ft_set = set(ft_glaciers_by_split.get(split, []))
        holdout_set = set(holdout_glaciers_by_split.get(split, []))

        # Sanity check
        overlap = ft_set & holdout_set
        if overlap and verbose:
            print(
                f"  Warning split={split}: {len(overlap)} glaciers in both FT and holdout: {overlap}"
            )

        def _label(g):
            if g in ft_set:
                return ft_label_ft
            if g in holdout_set:
                return ft_label_holdout
            return ft_label_excluded

        df = base.copy()
        df[ft_label_col] = df.index.to_series().apply(_label)
        glacier_info_by_split[split] = df

        if verbose:
            n_ft = int((df[ft_label_col] == ft_label_ft).sum())
            n_ho = int((df[ft_label_col] == ft_label_holdout).sum())
            n_excl = int((df[ft_label_col] == ft_label_excluded).sum())
            ft_rows = int(data_region[data_region[glacier_col].isin(ft_set)].shape[0])
            all_rows = int(data_region.shape[0])
            frac = ft_rows / all_rows if all_rows else float("nan")
            print(
                f"  split={split}: FT={n_ft}, Hold-out={n_ho}, Excluded={n_excl} | "
                f"FT rows fraction ~ {frac:.3f}"
            )

    return data_region, glacier_outline_rgi, glacier_info_by_split


def split_monthly_cache_by_glaciers(
    monthly_cache: dict,
    parent_key: str,
    subregions: dict,
    drop_parent: bool = False,
) -> dict:
    """
    Split a parent entry in monthly_cache into subregion entries.

    Parameters
    ----------
    monthly_cache : dict
        The existing monthly cache to update in-place.
    parent_key : str
        Key of the parent entry to split (e.g. 'IT_AT', 'CENTRALASIA', 'ALA').
    subregions : dict
        Mapping of new_key -> filter, where filter is either:
          - a list of glacier names (GLACIER column filter), or
          - a dict {'o2region_col': col, 'values': list} for O2Region filtering.
        Example for IT/AT:
            {'IT': ['CARESER', ...], 'AT': ['VERNAGT F.', ...]}
        Example for CA subregions (after O2Region is already mapped):
            {'CA_12': {'o2region_col': 'O2Region', 'values': ['1', '2']},
             'CA_3':  {'o2region_col': 'O2Region', 'values': ['3']}}
    drop_parent : bool
        If True, remove the parent entry from monthly_cache after splitting.

    Returns
    -------
    dict : updated monthly_cache
    """
    parent = monthly_cache[parent_key]
    dm = parent["data_monthly"]
    dm_aug = parent["data_monthly_aug"]

    for key, filter_spec in subregions.items():
        if isinstance(filter_spec, list):
            mask = dm["GLACIER"].isin(filter_spec)
            mask_aug = dm_aug["GLACIER"].isin(filter_spec)
        else:
            col = filter_spec["o2region_col"]
            values = filter_spec["values"]
            mask = dm[col].isin(values)
            mask_aug = dm_aug[col].isin(values)

        data = dm[mask].copy()
        data_aug = dm_aug[mask_aug].copy()
        data["SOURCE_CODE"] = key
        data_aug["SOURCE_CODE"] = key

        monthly_cache[key] = {
            "data_monthly": data,
            "data_monthly_aug": data_aug,
            "months_head_pad": parent["months_head_pad"],
            "months_tail_pad": parent["months_tail_pad"],
        }
        print(f"  {key}: {len(data)} measurements")

    if drop_parent:
        del monthly_cache[parent_key]

    return monthly_cache


def add_o2region_to_dfs(
    dfs: dict,
    rgi_id: str,
    cfg,
    deduplicate_glaciers: bool = False,
) -> tuple[dict, dict]:
    """
    Load O2Region from RGI shapefile and map onto dfs[rgi_id].
    Optionally deduplicate glaciers that span multiple RGIIds.

    Parameters
    ----------
    dfs : dict
        Stake dataframes keyed by RGI region id.
    rgi_id : str
        RGI region id to process (e.g. '01', '13').
    cfg : config object
        MBM config with dataPath.
    deduplicate_glaciers : bool
        If True, rename glaciers that map to multiple RGIIds
        by appending the RGIId suffix (e.g. EASTFORK_01_00022).

    Returns
    -------
    dfs : updated dict
    glacier_dict : dict mapping glacier name -> {'O2Region': ..., 'RGIId': ...}
    """
    if deduplicate_glaciers:
        deduped = dfs[rgi_id][["GLACIER", "RGIId"]].drop_duplicates().copy()
        multi_mask = deduped.groupby("GLACIER")["RGIId"].transform("count") > 1
        deduped["GLACIER_NEW"] = deduped["GLACIER"]
        deduped.loc[multi_mask, "GLACIER_NEW"] = (
            deduped.loc[multi_mask, "GLACIER"]
            + "_"
            + deduped.loc[multi_mask, "RGIId"].str.extract(r"\.(\d+)$")[0]
        )
        dfs[rgi_id] = dfs[rgi_id].merge(
            deduped[["GLACIER", "RGIId", "GLACIER_NEW"]],
            on=["GLACIER", "RGIId"],
            how="left",
        )
        dfs[rgi_id]["GLACIER"] = dfs[rgi_id]["GLACIER_NEW"].fillna(
            dfs[rgi_id]["GLACIER"]
        )
        dfs[rgi_id] = dfs[rgi_id].drop(columns="GLACIER_NEW")

    rgi_gdf = gpd.read_file(
        cfg.dataPath
        / RGI_V6_ROOT
        / RGI_REGIONS[rgi_id]["folder"]
        / RGI_REGIONS[rgi_id]["file"]
    )[["RGIId", "O2Region"]]

    glacier_dict = (
        dfs[rgi_id][["GLACIER", "RGIId"]]
        .drop_duplicates()
        .merge(rgi_gdf, on="RGIId", how="left")
        .set_index("GLACIER")
        .to_dict("index")
    )

    dfs[rgi_id]["O2Region"] = dfs[rgi_id]["GLACIER"].map(
        {k: v["O2Region"] for k, v in glacier_dict.items()}
    )

    print(dfs[rgi_id].groupby("O2Region").size().rename("n_measurements"))

    return dfs, glacier_dict


def add_o2region_to_monthly_cache(
    monthly_cache, code, glacier_dict, o2region_col="O2Region"
):
    """Add the RGI O2 sub-region of each glacier (from `add_o2region_to_dfs`) to
    the monthly data of one region in `monthly_cache` (both `data_monthly` and
    `data_monthly_aug`), so it can be split with `split_monthly_cache_by_glaciers`."""
    glacier_to_o2 = {k: v["O2Region"] for k, v in glacier_dict.items()}
    for key in ("data_monthly", "data_monthly_aug"):
        monthly_cache[code][key][o2region_col] = monthly_cache[code][key][
            "GLACIER"
        ].map(glacier_to_o2)
    return monthly_cache
