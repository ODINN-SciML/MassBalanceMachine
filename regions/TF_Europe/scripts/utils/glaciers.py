import numpy as np
import pandas as pd
import geopandas as gpd

from regions.TF_Europe.scripts.dataset import load_stakes_for_rgi_region


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
    df_test, FT_GLACIERS, source_col="SOURCE_CODE", glacier_col="GLACIER"
):

    results = []

    for region, splits in FT_GLACIERS.items():

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


def build_region_glacier_info_for_splits(
    cfg,
    *,
    rgi_region_id: str,
    outline_shp_path: str,
    ft_glaciers_by_split: dict,
    split_names=("5pct", "50pct"),
    ft_label_col="FT/Hold-out glacier",
    ft_label_ft="FT",
    ft_label_holdout="Hold-out",
    glacier_col="GLACIER",
    lat_col="POINT_LAT",
    lon_col="POINT_LON",
    period_col="PERIOD",
    nmeas_col="Nb. measurements",
    source_col="SOURCE_CODE",  # NEW
    source_resolution="mode",  # NEW: "error" | "first" | "mode" | "list"
    load_stakes_fn=None,
    verbose=True,
):
    """
    Generic builder for per-glacier info tables (for maps / summaries), for any region + any splits.
    Also carries SOURCE_CODE info into the final per-glacier dataframe.

    Returns
    -------
    data_region : pd.DataFrame
    glacier_outline_rgi : GeoDataFrame
    glacier_info_by_split : dict[str, pd.DataFrame]
        Indexed by GLACIER, with columns:
          [POINT_LAT, POINT_LON, Nb. measurements, (period counts...), SOURCE_CODE, FT/Hold-out glacier]
        SOURCE_CODE handling depends on source_resolution.
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

    # --- SOURCE_CODE per glacier (NEW) ---
    if source_col in data_region.columns:
        gsrc = data_region.groupby(glacier_col)[source_col].apply(
            lambda s: s.dropna().astype(str).unique()
        )

        # detect mixed source glaciers
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
            # mode by frequency in raw rows (more stable than unique list)
            def _mode(series):
                s = series.dropna().astype(str)
                if len(s) == 0:
                    return None
                vc = s.value_counts()
                return vc.index[0]

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

        df = base.copy()
        df[ft_label_col] = df.index.to_series().apply(
            lambda g: ft_label_ft if g in ft_set else ft_label_holdout
        )
        glacier_info_by_split[split] = df

        if verbose:
            n_ft = int((df[ft_label_col] == ft_label_ft).sum())
            n_ho = int((df[ft_label_col] == ft_label_holdout).sum())
            ft_rows = int(data_region[data_region[glacier_col].isin(ft_set)].shape[0])
            all_rows = int(data_region.shape[0])
            frac = (ft_rows / all_rows) if all_rows else float("nan")
            print(
                f"  split={split}: FT glaciers={n_ft}, Hold-out glaciers={n_ho} | "
                f"FT rows fraction ~ {frac:.3f}"
            )

    return data_region, glacier_outline_rgi, glacier_info_by_split
