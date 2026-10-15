"""Selection of geodetic periods fixed by the dates of their DEMs, shared by the sources
that publish a table of such periods (`data_processing.fischer11`,
`data_processing.hagg12`, `data_processing.belart20`).

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


def first_of_nearest_month(date) -> pd.Timestamp:
    """The first of the month nearest to `date`: the first of its own month up to the
    15th, the first of the next month after it. "1960-08-13" gives 1960-08-01 and
    "1960-08-24" gives 1960-09-01."""
    date = pd.Timestamp(date)
    first = date.replace(day=1)
    return first if date.day <= 15 else first + pd.DateOffset(months=1)


def best_period_chain(periods: pd.DataFrame) -> pd.DataFrame:
    """Rows of the best set of non-overlapping periods of one glacier.

    A source whose periods are nested (1960-1980, 1960-1994 and 1980-1994 on the same
    glacier) cannot hand them all to the loss: they are built on the same DEMs, so the
    same change would count several times with strongly correlated errors. Two periods
    may share a date (one ends on the 1st of September 1980, the next starts on it).

    Among all the sets of non-overlapping periods, the one kept has **the most
    periods**, then the **longest total duration**, then the **lowest sum of sigma^2**,
    as `glamos._best_window_chain` does for the GLAMOS windows, but on the dates
    FROM_DATE and TO_DATE rather than on years.

    The search is the classic weighted interval scheduling dynamic program over the
    periods sorted by end date; a glacier has at most a few dozen periods.
    """
    periods = periods.sort_values(["TO_DATE", "FROM_DATE"])
    start = periods.FROM_DATE.to_numpy()
    end = periods.TO_DATE.to_numpy()
    dur = periods.n_years.to_numpy()
    sigma2 = periods.sigma_mwe_per_year.fillna(float("inf")).to_numpy() ** 2

    # best[i] is the best chain whose last period is i, scored as a tuple compared
    # lexicographically: (number of periods, total duration, -sum of sigma^2)
    best = []
    for i in range(len(periods)):
        score, chain = (1, dur[i], -sigma2[i]), [i]
        for j in range(i):
            if end[j] <= start[i]:
                prev_score, prev_chain = best[j]
                candidate = (
                    prev_score[0] + 1,
                    prev_score[1] + dur[i],
                    prev_score[2] - sigma2[i],
                )
                if candidate > score:
                    score, chain = candidate, prev_chain + [i]
        best.append((score, chain))
    if not best:
        return periods
    _, chain = max(best, key=lambda b: b[0])
    return periods.iloc[chain]
