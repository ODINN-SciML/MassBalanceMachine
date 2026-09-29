"""Selection of geodetic periods fixed by the dates of their DEMs, shared by the sources
that publish a table of such periods (`data_processing.fischer11`,
`data_processing.hagg12`).

Like `utils.years`, it imports nothing from `data_processing`, so that the sources can
use it while being imported by `data_processing.gridded_utils`.
"""

import pandas as pd

from data_processing.utils.years import years_outside

TIE_BREAK_KEYS = {
    "sigma": ["sigma_mwe_per_year"],
    "recent": ["y1", "sigma_mwe_per_year"],
}


def select_dem_periods(
    df: pd.DataFrame,
    max_year: int = None,
    min_year: int = None,
    min_period_years: int = 1,
    glacier_ids_to_keep=None,
    allowed_years=None,
    max_outside_years: int = 0,
    multi_period: bool = True,
    tie_break: str = "sigma",
):
    """Keep the eligible periods of a table of geodetic periods.

    `df` holds one row per period, with the columns RGIId (the glacier id of the
    source), y0 and y1 (years of the two DEMs), n_years, FROM_DATE, TO_DATE and
    sigma_mwe_per_year.

    By default a glacier keeps every eligible period; the periods of a source follow
    each other and never overlap. Without `multi_period` a single period is kept per
    glacier, chosen as GLAMOS chooses its window (see `glamos.select_glamos_windows`):
    **the longest one, ties broken on the lowest sigma**, or on the most recent period
    with `tie_break="recent"`.

    Args:
        max_year: every period must end at or before the DEM of this year.
        min_year: every period must start at or after the DEM of this year. Set it
            to the first year of the climate forcing.
        min_period_years: shortest period to keep.
        glacier_ids_to_keep: glacier ids to restrict the periods to.
        allowed_years: calendar years a period may touch, for instance the years of
            one side of a train/validation split, up to `max_outside_years` years of
            tolerance. A period is fixed by its DEMs and cannot be cut, as for GLAMOS.
        multi_period: keep every eligible period of a glacier rather than one.
        tie_break: how the single period is chosen among the longest ones, "sigma"
            or "recent". Ignored with `multi_period`.

    Returns the rows kept, sorted by glacier and start date.
    """
    assert (
        tie_break in TIE_BREAK_KEYS
    ), f"tie_break must be one of {sorted(TIE_BREAK_KEYS)}, not {tie_break!r}."
    keep = df.n_years >= min_period_years
    if max_year is not None:
        keep &= df.y1 <= max_year
    if min_year is not None:
        keep &= df.y0 >= min_year
    if glacier_ids_to_keep is not None:
        keep &= df.RGIId.isin([str(g) for g in glacier_ids_to_keep])
    if allowed_years is not None:
        allowed = set(allowed_years)
        keep &= pd.Series(
            [
                len(years_outside(r.FROM_DATE, r.TO_DATE, allowed)) <= max_outside_years
                for r in df.itertuples()
            ],
            index=df.index,
        )
    df = df[keep]
    if not multi_period:
        keys = ["n_years"] + TIE_BREAK_KEYS[tie_break]
        df = df.sort_values(
            keys, ascending=[k == "sigma_mwe_per_year" for k in keys]
        ).drop_duplicates("RGIId")
    return df.sort_values(["RGIId", "FROM_DATE"]).reset_index(drop=True)
