import os
import pandas as pd

from data_processing.utils.hydro_year import _compute_months_before_after
from data_retrieval.nve import point_mb_path, download_point_mb

# Most months a measurement may reach before or after its hydrological year. Every regular
# measurement of the dataset needs at most 3; the ones needing more carry a date error
# (see `load_points`).
MAX_PADDING_MONTHS = 3
# Shortest annual balance, in days, and the months an annual and a winter balance may end
# in (see `load_points`)
MIN_ANNUAL_DAYS = 250
ANNUAL_END_MONTHS = range(8, 13)
WINTER_END_MONTHS = range(1, 8)
# Stake values copied from one stake to the others of a zone, (Bre-ID, year of the
# winter survey): Ålfotbreen 1982 (3.21 at six stakes over 380 m), Hansebreen 1982
# (3.76 at its four stakes), Ålfotbreen 1965 (3.80 at four stakes over 345 m)
COPIED_WINTER_BALANCES = [(2078, 1982), (2085, 1982), (2078, 1965)]


def raw_data():
    """The NVE point mass balance table, one row per stake and year, downloaded on first
    use."""
    if not os.path.isfile(point_mb_path()):
        download_point_mb()
    return pd.read_csv(point_mb_path(), na_values=["NAN"])


def load_points():
    """The NVE stake measurements in the format of the WGMS point table: one row per
    measurement period, dates as YYYYMMDD strings and YEAR the year of TO_DATE.

    Each stake year has a winter balance (previous minimum to maximum), a summer
    balance (maximum to minimum) and an annual balance (previous minimum to minimum).
    Since the annual balance is the sum of the two others, the summer balance is only
    kept for the years without annual balance, as the WGMS stakes of region 08 only
    have winter and annual balances.

    About twenty measurements reach more than `MAX_PADDING_MONTHS` months before or after
    their hydrological year: winter balances over 15 to 19 months, or annual balances from
    a previous minimum dated in January (1995-01-04 to 1996-09-23), date errors of the
    dataset. They are dropped: one of them is enough for the monthly table to need more
    than 12 padded months, which it cannot hold.

    The other measurements whose dates contradict their period are dropped too: annual
    balances shorter than `MIN_ANNUAL_DAYS` days (1988-09-12 to 1988-10-18) or ending
    outside `ANNUAL_END_MONTHS` (1993-09-29 to 1994-05-15, -2.16 m w.e.), and winter
    balances ending outside `WINTER_END_MONTHS` (1966-11-18 to 1967-09-29, 2011-10-13 to
    2012-11-17). So are the winter balances of `COPIED_WINTER_BALANCES`, one value
    copied over the stakes of a zone, and the summer balance of 1982 of a stake of
    Nigardsbreen at 998 m whose position is that of a stake at 1511 m on the neighbouring
    glacier RGI60-08.01121, where it would be assigned."""
    raw = raw_data()
    prev_min, curr_max, curr_min = (
        pd.to_datetime(raw[c], format="%d.%m.%Y").dt.strftime("%Y%m%d")
        for c in ["prev_yr_min_date", "curr_yr_max_date", "curr_yr_min_date"]
    )
    position = {
        "POINT_LAT": raw.lat,
        "POINT_LON": raw.lon,
        "POINT_ELEVATION": raw.altitude,
        "BREID": raw.breid,
    }
    winter = pd.DataFrame(
        {
            **position,
            "FROM_DATE": prev_min,
            "TO_DATE": curr_max,
            "POINT_BALANCE": raw.bw,
        }
    ).assign(PERIOD="winter")
    annual = pd.DataFrame(
        {
            **position,
            "FROM_DATE": prev_min,
            "TO_DATE": curr_min,
            "POINT_BALANCE": raw.ba,
        }
    ).assign(PERIOD="annual")
    summer = pd.DataFrame(
        {
            **position,
            "FROM_DATE": curr_max,
            "TO_DATE": curr_min,
            "POINT_BALANCE": raw.bs,
        }
    ).assign(PERIOD="summer")[raw.ba.isna()]

    points = pd.concat([winter, annual, summer], ignore_index=True)
    points = points.dropna(subset=["FROM_DATE", "TO_DATE", "POINT_BALANCE"])
    points["YEAR"] = points.TO_DATE.str[:4].astype(int)

    # The periods ending before they start have no hydrological year: they are dropped by
    # `wgms.filter_points`
    dates = points.assign(
        FROM_DATE=pd.to_datetime(points.FROM_DATE),
        TO_DATE=pd.to_datetime(points.TO_DATE),
    )
    forward = dates.TO_DATE >= dates.FROM_DATE
    padding = _compute_months_before_after(dates[forward])
    misdated = pd.Series(False, index=points.index)
    misdated[forward] = (padding.MONTHS_BEFORE > MAX_PADDING_MONTHS) | (
        padding.MONTHS_AFTER > MAX_PADDING_MONTHS
    )
    print(
        f"NVE: {misdated.sum()} measurement(s) reaching more than {MAX_PADDING_MONTHS} "
        "months outside their hydrological year dropped"
    )
    points = points[~misdated.to_numpy(dtype=bool)]
    dates = dates[~misdated.to_numpy(dtype=bool)]

    days = (dates.TO_DATE - dates.FROM_DATE).dt.days
    end_month = dates.TO_DATE.dt.month
    annual, winter = points.PERIOD == "annual", points.PERIOD == "winter"
    wrong_period = (
        annual & ((days < MIN_ANNUAL_DAYS) | ~end_month.isin(ANNUAL_END_MONTHS))
    ) | (winter & ~end_month.isin(WINTER_END_MONTHS))
    copied = winter & pd.Series(
        list(zip(points.BREID, points.YEAR)), index=points.index
    ).isin(COPIED_WINTER_BALANCES)
    wrong_position = (
        (points.BREID == 2297)
        & (points.PERIOD == "summer")
        & (points.YEAR == 1982)
        & (points.POINT_ELEVATION == 998)
    )
    print(
        f"NVE: {wrong_period.sum()} measurement(s) whose dates contradict their period, "
        f"{copied.sum()} winter balance(s) copied over a zone and {wrong_position.sum()} "
        "stake at the position of another dropped"
    )
    points = points[~(wrong_period | copied | wrong_position)]
    return points.drop(columns=["BREID"]).reset_index(drop=True)
