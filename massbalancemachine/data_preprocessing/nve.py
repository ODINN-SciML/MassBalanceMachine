import os
import pandas as pd

from data_retrieval.nve import point_mb_path, download_point_mb


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
    have winter and annual balances."""
    raw = raw_data()
    prev_min, curr_max, curr_min = (
        pd.to_datetime(raw[c], format="%d.%m.%Y").dt.strftime("%Y%m%d")
        for c in ["prev_yr_min_date", "curr_yr_max_date", "curr_yr_min_date"]
    )
    position = {
        "POINT_LAT": raw.lat,
        "POINT_LON": raw.lon,
        "POINT_ELEVATION": raw.altitude,
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
    return points.reset_index(drop=True)
