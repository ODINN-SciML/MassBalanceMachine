"""
GLACIOCLIM stake measurements of the French Alps (https://glacioclim.osug.fr/Donnees-des-Alpes):
Saint-Sorlin, Gébroulaz, Argentière (with Tour Noir), Mer de Glace (with Tacul, Leschaux
and Talèfre) and Sarennes, winter, summer and annual balances, 1957-2022.

The shares downloaded by `data_retrieval.glacioclim` are assembled into one table,
`assembled_path()`, with one row per measurement and the stake positions in WGS84. It is
built on first use; `build_assembled_table()` rebuilds it.
"""

import io
import os
import re
import zipfile

import numpy as np
import pandas as pd
from pyproj import Transformer

from data_processing.utils.hydro_year import _compute_months_before_after
from data_retrieval.glacioclim import (
    SARENNES_SHARE,
    SHARES,
    download_all,
    glacioclim_path,
    share_file,
)

# Most months a measurement may reach before or after its hydrological year (see
# `load_points`)
MAX_PADDING_MONTHS = 3
# Erroneous measurements, (stake id as built by `stake_id`, period, TO_DATE), found by
# the checks run on the WGMS stakes (profile outliers, seasonal closure, window length,
# copies, co-located stakes, positions). "Neighbours" are the stakes of the same glacier,
# period and dates within 75 m of elevation. A residual from the neighbours that recurs at
# the same site in other years (avalanche deposits at the margins of the Argentière
# profiles, Saint-Sorlin stake 29, Gébroulaz stake 14...) is taken as real and kept.
# Since the summer balance of a stake is its annual minus its winter balance (they close
# within 0.05 m w.e. at all but one of the 2,538 stake-years with the three), an error
# made at the spring survey spoils both seasonal balances: both are dropped, and the
# annual balance is kept when it agrees with its neighbours.
ERRONEOUS_MEASUREMENTS = [
    # Value inconsistent with the neighbours (residual of 2.5 to 84 m w.e., 3 to 25 times
    # their spread) and not recurring at the same site in other years
    (
        "saint_sorlin/ablation/ablation/1972/58",
        "annual",
        "19820924",
    ),  # -0.04, nb. -2.61
    (
        "saint_sorlin/ablation/ablation/2012/10",
        "annual",
        "20151009",
    ),  # +0.02, nb. -3.75
    (
        "saint_sorlin/ablation/ablation/2012/10",
        "summer",
        "20151009",
    ),  # -0.82, nb. -4.62
    (
        "saint_sorlin/ablation/ablation/2015/14",
        "summer",
        "20170926",
    ),  # -88.41, nb. -4.33
    ("argentiere/ablation/profil5/1981/9", "annual", "19820909"),  # +0.72, nb. -2.63
    ("argentiere/ablation/profil5/1981/3", "annual", "19830905"),  # -4.95, nb. -1.98
    (
        "mer_de_glace/langue/Langue/2009/4",
        "winter",
        "20110502",
    ),  # +5.88 at 1963 m, nb. -0.28
    ("mer_de_glace/langue/Langue/2019/4", "winter", "20200520"),  # -3.87, nb. -1.49
    # Spring survey off: negative winter balances from September to April at 2750-2770
    # m (neighbours +1.1 to +1.3), or a winter of +4.49 at 2747 m (nb. +1.17), with their
    # summer balances; the annual balances agree with the neighbours
    ("saint_sorlin/ablation/ablation/2003/10", "winter", "20040422"),  # -1.74
    ("saint_sorlin/ablation/ablation/2003/10", "summer", "20041005"),  # -1.55
    ("saint_sorlin/ablation/ablation/2019/3", "winter", "20220411"),  # +4.49
    ("saint_sorlin/ablation/ablation/2019/3", "summer", "20220923"),  # -9.08
    ("gebroulaz/ablation/ablation/2004/12", "winter", "20060426"),  # -1.09
    ("gebroulaz/ablation/ablation/2004/12", "summer", "20060921"),  # -1.36
    ("mer_de_glace/langue/Langue/2008/5", "winter", "20090506"),  # -2.56, nb. -0.45
    ("mer_de_glace/langue/Langue/2008/5", "summer", "20091026"),  # -6.25
    # Autumn survey off: winter balance in line with the neighbours, annual and summer
    # both about 2 m w.e. away from theirs
    (
        "saint_sorlin/ablation/ablation/2012/30",
        "annual",
        "20181012",
    ),  # -2.48, nb. -0.73
    (
        "saint_sorlin/ablation/ablation/2012/30",
        "summer",
        "20181012",
    ),  # -4.70, nb. -2.49
    ("gebroulaz/ablation/ablation/2016/10", "annual", "20170929"),  # -0.31, nb. -2.50
    ("gebroulaz/ablation/ablation/2016/10", "summer", "20170929"),  # -1.41, nb. -3.46
    # Annual balance differing from the sum of its seasonal balances (-0.72), the value of
    # stake 11
    ("saint_sorlin/ablation/ablation/2019/1", "annual", "20201008"),  # -1.62
    # Two stakes at the same position (an old stake and its replacement, < 10 m apart)
    # over the same dates, disagreeing by more than 0.5 m w.e.: the one away from the
    # neighbours is dropped
    (
        "saint_sorlin/ablation/ablation/1990/18",
        "annual",
        "19940901",
    ),  # +0.17 / -2.29, nb. -1.66
    (
        "argentiere/ablation/profil5/1984/8",
        "annual",
        "19870909",
    ),  # +0.72 / -1.35, nb. -1.57
    (
        "argentiere/ablation/profil5/1984/7",
        "annual",
        "19870909",
    ),  # -0.22 / -1.53, nb. -1.57
    (
        "argentiere/ablation/profil5/1984/8",
        "annual",
        "19880919",
    ),  # -2.07 / -2.61, nb. -2.77
    (
        "argentiere/ablation/profil7/1998/18",
        "annual",
        "19990908",
    ),  # -0.15 / -0.85, nb. -0.93
    (
        "argentiere/ablation/profil7/1994/23",
        "annual",
        "20001004",
    ),  # -0.27 / -1.94, nb. -1.82
    (
        "mer_de_glace/tacul/Tacul/1991/12",
        "annual",
        "19930911",
    ),  # -2.79 / -3.73, nb. -4.36
    (
        "mer_de_glace/talefre/Talefre/2005/2",
        "annual",
        "20080930",
    ),  # -2.25 / -3.02, nb. -3.05
    # Accumulation zone of the Mer de Glace, 2017: annual balances of stakes labelled
    # 2016 at the positions and dates of stakes 2017/4 and 2017/5, of opposite sign to
    # theirs (+0.67, +0.76), which close with their own winter and summer balances
    ("mer_de_glace/accumulation/acum/2016/4", "annual", "20170907"),  # -0.53
    ("mer_de_glace/accumulation/acum/2016/5", "annual", "20170907"),  # -0.61
    # Leschaux, stake 2017/3: position of stake 2017/1 (2595 m instead of about 2420 m)
    ("mer_de_glace/leschaux/Leschaux/2017/3", "summer", "20191017"),
    ("mer_de_glace/leschaux/Leschaux/2017/3", "winter", "20200520"),
    # Argentière profile 4, summer 2018, stake 2017/3: ends on 11 December, while the
    # same campaign ends on 23 October for the other stakes
    ("argentiere/ablation/profil4/2017/3", "summer", "20181211"),
    # Sarennes 1996: annual balance 0.00 at the five stakes (2858-3072 m), a placeholder;
    # the summer balances, equal to minus the winter ones, were derived from it
    *[
        (f"sarennes/all/sarennes/-/{stake}", period, "19960914")
        for stake in range(1, 6)
        for period in ["annual", "summer"]
    ],
]

# Coordinate systems. The label of the coordinate columns of the CSV files (x_lambert3 or
# x_lambert2e) cannot be trusted: the winter balances of Saint-Sorlin in the accumulation
# zone in 2018-2020 are labelled lambert2e but given in Lambert III like the rest of the
# glacier. The system is therefore chosen per glacier:
# - Saint-Sorlin, Gébroulaz and Sarennes: Lambert III (zone), y around 3.2e5-3.4e5;
# - Argentière and Mer de Glace: Lambert II étendu, almost always written without its
#   leading 2,000,000 of northing, i.e. in Lambert II (zone), which differs from it only
#   by its false northing (200,000 instead of 2,200,000). The few northings above 1e6 are
#   full Lambert II étendu.
LAMBERT3 = "EPSG:27563"
LAMBERT2 = "EPSG:27562"
LAMBERT2E = "EPSG:27572"
LAMBERT3_GLACIERS = ["saint_sorlin", "gebroulaz", "sarennes"]

COLUMNS = [
    "GLACIER",
    "ZONE",
    "PERIOD",
    "PROFILE",
    "STAKE_YEAR_SETUP",
    "STAKE_NUMBER",
    "FROM_DATE",
    "TO_DATE",
    "X",
    "Y",
    "CRS_LABEL",
    "EPSG",
    "ALTITUDE",
    "POINT_LAT",
    "POINT_LON",
    "BALANCE",
    "SHARE",
    "FILE",
]


def assembled_path():
    return os.path.join(glacioclim_path(), "glacioclim_points.csv")


def stake_id(table):
    """Identifier of the stakes of the assembled table:
    glacier/zone/profile/setup year/number, with '-' for an unknown setup year
    (Sarennes)."""
    setup = table.STAKE_YEAR_SETUP.astype("Int64").astype(str).replace("<NA>", "-")
    return (
        table.GLACIER
        + "/"
        + table.ZONE
        + "/"
        + table.PROFILE
        + "/"
        + setup
        + "/"
        + table.STAKE_NUMBER
    )


def _yyyymmdd(year, month, day):
    return pd.to_datetime(
        pd.DataFrame({"year": year, "month": month, "day": day})
    ).dt.strftime("%Y%m%d")


def _parse_share(token, glacier, zone, period):
    """The measurements of a zip share: one CSV per year, with the columns profile_name,
    stake_year_setup, stake_number, day/month/year_start, day/month/year_end,
    x_<crs>, y_<crs>, altitude and <period>_smb (m w.e.)."""
    tables = []
    with zipfile.ZipFile(share_file(token)) as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.endswith(".csv"):
                continue
            df = pd.read_csv(io.BytesIO(z.read(info)), encoding="latin-1")
            (x_col,) = [c for c in df.columns if c.startswith("x_")]
            label = x_col[2:]
            assert (
                f"{period}_smb" in df.columns
            ), f"{info.filename} of share {token} has no {period}_smb column: {list(df.columns)}"
            tables.append(
                pd.DataFrame(
                    {
                        "PROFILE": df.profile_name.astype(str),
                        "STAKE_YEAR_SETUP": df.stake_year_setup,
                        "STAKE_NUMBER": df.stake_number.astype(str),
                        "FROM_DATE": _yyyymmdd(
                            df.year_start, df.month_start, df.day_start
                        ),
                        "TO_DATE": _yyyymmdd(df.year_end, df.month_end, df.day_end),
                        "X": df[x_col],
                        "Y": df[f"y_{label}"],
                        "CRS_LABEL": label,
                        "ALTITUDE": df.altitude,
                        "BALANCE": df[f"{period}_smb"],
                        "FILE": info.filename,
                    }
                )
            )
    return pd.concat(tables, ignore_index=True).assign(
        GLACIER=glacier, ZONE=zone, PERIOD=period, SHARE=token
    )


def _parse_sarennes_date(value):
    """Dates of the Sarennes workbook, written 'D. M.YYYY' with optional spaces."""
    if isinstance(value, (pd.Timestamp, np.datetime64)) or hasattr(value, "year"):
        return pd.Timestamp(value).strftime("%Y%m%d")
    match = re.fullmatch(r"\s*(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{4})\s*", str(value))
    assert match, f"Unreadable date in the Sarennes workbook: {value!r}"
    day, month, year = (int(g) for g in match.groups())
    return pd.Timestamp(year=year, month=month, day=day).strftime("%Y%m%d")


def _parse_sarennes():
    """The measurements of the Sarennes workbook. Each sheet is a hydrological year; each
    stake (column Balise) has three dated rows, autumn, spring and next autumn: the
    first holds the position (X, Y in Lambert III, Z) and the row below it the winter
    balance, the second the annual balance and the row below it the summer balance. A
    stake out of the glacier has no balance."""
    sheets = pd.read_excel(
        share_file(SARENNES_SHARE), sheet_name=None, header=None, engine="xlrd"
    )
    rows = []
    for sheet, df in sheets.items():
        assert list(df.iloc[1, :5]) == ["hydrologique", "Balise", "Date", "X", "Y"], (
            f"Unexpected header in sheet {sheet} of the Sarennes workbook: "
            f"{list(df.iloc[1])}"
        )
        body = df.iloc[3:].reset_index(drop=True)
        number = lambda i, j: pd.to_numeric(body.at[i, j], errors="coerce")
        stakes = pd.to_numeric(body[1], errors="coerce")
        for stake, idx in body.index[stakes.notna()].groupby(stakes.dropna()).items():
            assert len(idx) == 3, f"Sarennes {sheet}, stake {stake}: {len(idx)} dates"
            autumn, spring, next_autumn = idx
            balances = {
                "winter": (number(autumn + 1, 6), autumn, spring),
                "summer": (number(spring + 1, 7), spring, next_autumn),
                "annual": (number(spring, 8), autumn, next_autumn),
            }
            for period, (balance, start, end) in balances.items():
                if np.isnan(balance):
                    continue
                rows.append(
                    {
                        "PROFILE": "sarennes",
                        "STAKE_YEAR_SETUP": np.nan,
                        "STAKE_NUMBER": str(int(stake)),
                        "FROM_DATE": _parse_sarennes_date(body.at[start, 2]),
                        "TO_DATE": _parse_sarennes_date(body.at[end, 2]),
                        "X": number(autumn, 3),
                        "Y": number(autumn, 4),
                        "CRS_LABEL": "",
                        "ALTITUDE": number(autumn, 5),
                        "BALANCE": balance,
                        "PERIOD": period,
                        "FILE": f"sheet {sheet}",
                    }
                )
    points = pd.DataFrame(rows).assign(
        GLACIER="sarennes", ZONE="all", SHARE=SARENNES_SHARE
    )
    assert points[["X", "Y"]].notna().all().all(), "Sarennes stake without position"
    # The sheet of a hydrological year ends in that year
    assert (points.TO_DATE.str[:4] == points.FILE.str[-4:]).all()
    return points


def _epsg(points):
    """Coordinate system of each point, see `LAMBERT3_GLACIERS`."""
    lambert3 = points.GLACIER.isin(LAMBERT3_GLACIERS)
    assert (
        points.Y[lambert3] < 1e6
    ).all(), "Lambert III northing above 1e6, unknown coordinate system"
    return np.where(lambert3, LAMBERT3, np.where(points.Y < 1e6, LAMBERT2, LAMBERT2E))


def build_assembled_table():
    """Parse every downloaded share (downloading the missing ones), project the stake
    positions to WGS84 and write the measurements to `assembled_path()`."""
    download_all()
    points = pd.concat(
        [_parse_share(*share) for share in SHARES] + [_parse_sarennes()],
        ignore_index=True,
    )
    points["EPSG"] = _epsg(points)
    points["POINT_LAT"], points["POINT_LON"] = np.nan, np.nan
    for epsg, group in points.groupby("EPSG"):
        transformer = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(group.X.to_numpy(), group.Y.to_numpy())
        points.loc[group.index, "POINT_LON"] = lon
        points.loc[group.index, "POINT_LAT"] = lat
    points = points[COLUMNS]
    points.to_csv(assembled_path(), index=False)
    print(f"{len(points)} GLACIOCLIM measurements written to {assembled_path()}")
    return points


def raw_data():
    """The assembled GLACIOCLIM table, one row per measurement, built on first use."""
    if not os.path.isfile(assembled_path()):
        build_assembled_table()
    return pd.read_csv(
        assembled_path(),
        dtype={"FROM_DATE": str, "TO_DATE": str, "STAKE_NUMBER": str, "CRS_LABEL": str},
        keep_default_na=False,
        na_values=[""],
    )


def load_points():
    """The GLACIOCLIM stake measurements in the format of the WGMS point table: one row
    per measurement period (winter, summer and annual all kept), dates as YYYYMMDD
    strings and YEAR the year of TO_DATE.

    An elevation of 0 is a missing elevation: it is set to NaN, and the measurement is
    dropped by `wgms.filter_points`, as is the annual balance of the Langue of the Mer
    de Glace in 2008 at stake 9, dated from 2009-09-03 to 2008-09-30. The measurements
    of `ERRONEOUS_MEASUREMENTS` are dropped. The balances of exactly 0 are kept apart
    from Sarennes 1996: they agree with their neighbours (stakes near the equilibrium
    line, winters ending in June on the Argentière profiles or on the Langue). Identical
    copies of a measurement (two in the dataset) are dropped by
    `stakes.drop_identical_measurements`. The measurements reaching more than
    `MAX_PADDING_MONTHS` months before or after their hydrological year, as the monthly
    table cannot hold them.

    Stakes off the RGI outlines (early Saint-Sorlin stakes below the 2003 front, Sarennes
    1994-2005) get no RGIId and are dropped by `wgms.filter_points`."""
    raw = raw_data()
    erroneous = pd.Series(list(zip(stake_id(raw), raw.PERIOD, raw.TO_DATE))).isin(
        ERRONEOUS_MEASUREMENTS
    )
    assert (
        erroneous.sum()
        == len(set(ERRONEOUS_MEASUREMENTS))
        == len(ERRONEOUS_MEASUREMENTS)
    ), "Erroneous GLACIOCLIM measurements not found: the dataset has changed"
    print(f"GLACIOCLIM: {erroneous.sum()} erroneous measurement(s) dropped")
    raw = raw[~erroneous.to_numpy()].reset_index(drop=True)
    points = pd.DataFrame(
        {
            "POINT_LAT": raw.POINT_LAT,
            "POINT_LON": raw.POINT_LON,
            "POINT_ELEVATION": raw.ALTITUDE.where(raw.ALTITUDE > 0),
            "POINT_BALANCE": raw.BALANCE,
            "PERIOD": raw.PERIOD,
            "FROM_DATE": raw.FROM_DATE,
            "TO_DATE": raw.TO_DATE,
            "GLACIER": raw.GLACIER,
        }
    )
    points["YEAR"] = points.TO_DATE.str[:4].astype(int)
    print(
        f"GLACIOCLIM: {points.POINT_ELEVATION.isna().sum()} measurement(s) without elevation"
    )

    # The periods ending before they start have no hydrological year: they are dropped by
    # `wgms.filter_points`
    dates = points.assign(
        FROM_DATE=pd.to_datetime(points.FROM_DATE),
        TO_DATE=pd.to_datetime(points.TO_DATE),
    )
    forward = dates.TO_DATE >= dates.FROM_DATE
    padding = _compute_months_before_after(dates[forward])
    misdated = pd.Series(False, index=points.index)
    misdated[forward] = (
        (padding.MONTHS_BEFORE > MAX_PADDING_MONTHS)
        | (padding.MONTHS_AFTER > MAX_PADDING_MONTHS)
    ).to_numpy(dtype=bool)
    print(
        f"GLACIOCLIM: {misdated.sum()} measurement(s) reaching more than "
        f"{MAX_PADDING_MONTHS} months outside their hydrological year dropped"
    )
    points = points[~misdated.to_numpy(dtype=bool)]
    return points.reset_index(drop=True)
