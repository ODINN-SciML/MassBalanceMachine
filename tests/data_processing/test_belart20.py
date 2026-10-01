"""Tests of the Belart20 source: the periods of Table S1 of Belart et al. (2020), their
dates, units and uncertainty, the chain of non-overlapping periods, and the crosswalk
to RGI 6.2.

The cross-check against WGMS reads the Fluctuations of Glaciers database from
`.data/WGMS`, and the outline tests read the RGI 6.2 outlines from OGGM's local copy;
each is skipped when its data is missing.
"""

import os

import pandas as pd
import pytest

from data_processing.belart20 import (
    BELART20_GLACIERS,
    BELART20_PERIODS,
    CI95_TO_SIGMA,
    belart20_periods,
    geodetic_target_Belart20,
    load_belart20_outlines,
    table_RGI62_to_Belart20,
)
from data_processing.product_utils import get_data_path
from data_processing.utils.periods import first_of_nearest_month


def test_every_glacier_of_the_paper_has_periods():
    periods = belart20_periods()
    assert len(periods) == 96
    assert set(periods.RGIId) == set(BELART20_GLACIERS)


def test_dates_are_rounded_to_the_nearest_first_of_a_month():
    assert first_of_nearest_month("1960-08-13") == pd.Timestamp("1960-08-01")
    assert first_of_nearest_month("1960-08-15") == pd.Timestamp("1960-08-01")
    assert first_of_nearest_month("1960-08-24") == pd.Timestamp("1960-09-01")
    assert first_of_nearest_month("2016-12-20") == pd.Timestamp("2017-01-01")
    periods = belart20_periods()
    assert (periods.FROM_DATE.dt.day == 1).all() and (periods.TO_DATE.dt.day == 1).all()
    assert (periods.FROM_DATE < periods.TO_DATE).all()


def test_rate_follows_the_years_of_the_surveys():
    # Mýrdalsjökull, 5 August 1999 - 5 October 2004: the published rate is the change
    # over 5 years, spread here over the 62 months from August 1999 to October 2004
    row = belart20_periods().set_index(["RGIId", "y0", "y1"]).loc[("MYR", 1999, 2004)]
    assert row.FROM_DATE == pd.Timestamp("1999-08-01")
    assert row.TO_DATE == pd.Timestamp("2004-10-01")
    assert row.n_years == pytest.approx(62 / 12)
    assert row.cumulative_mwe == pytest.approx(-2.64 * 5)
    assert row.mwe_per_year == pytest.approx(-2.64 * 5 / (62 / 12))
    assert row.sigma_mwe_per_year == pytest.approx(0.21 * CI95_TO_SIGMA * 5 / (62 / 12))


def test_chain_keeps_non_overlapping_periods():
    target = geodetic_target_Belart20()
    for _, df in target.groupby("RGIId"):
        df = df.sort_values("FROM_DATE")
        assert (df.TO_DATE.to_numpy()[:-1] <= df.FROM_DATE.to_numpy()[1:]).all()
    # Tindfjallajökull: 1945-1960, then the most periods without overlap
    tin = target[target.RGIId == "TIN"]
    assert list(zip(tin.y0, tin.y1)) == [
        (1945, 1960),
        (1960, 1978),
        (1978, 1990),
        (1990, 1994),
        (1994, 2004),
        (2004, 2011),
        (2011, 2017),
    ]


def test_single_period_and_filters():
    target = geodetic_target_Belart20(min_year=1950, max_year=2017, multi_period=False)
    assert target.RGIId.is_unique
    assert target.y0.min() >= 1950 and target.y1.max() <= 2017
    # Hrútfell: 1960-1995 is the longest period starting after 1950
    assert tuple(target.set_index("RGIId").loc["HRU", ["y0", "y1"]]) == (1960, 1995)


WGMS_CHANGE = os.path.join(
    get_data_path(), "WGMS", "DOI-WGMS-FoG-2026-02-10", "data", "change.csv"
)
# glacier code: WGMS id of the glacier
WGMS_IDS = {
    "BAR": 26288,
    "DRA": 6831,
    "EIR": 26289,
    "EYJ": 3353,
    "HOF": 26290,
    "HRU": 26291,
    "MYR": 6832,
    "ORA": 26292,
    "SNA": 26294,
    "SNJ": 26293,
    "THR": 3125,
    "TIN": 26295,
    "TOR": 26296,
    "TUN": 26297,
}
# Known errors of the WGMS copy, see the docstring of `data_processing.belart20`
WGMS_ERRORS = {("HOF", 1967, 1990), ("EIR", 1960, 1978), ("EIR", 1960, 1995)}


@pytest.mark.skipif(not os.path.exists(WGMS_CHANGE), reason="WGMS FoG not available")
def test_table_matches_wgms():
    wgms = pd.read_csv(WGMS_CHANGE)
    wgms = wgms[wgms.references.fillna("").str.contains(r"Belart et al\. \(2020\)")]
    wgms = wgms.assign(
        y0=pd.to_datetime(wgms.begin_date).dt.year,
        y1=pd.to_datetime(wgms.end_date).dt.year,
    )
    wgms["mwe_per_year"] = wgms.elevation_change * 0.85 / (wgms.y1 - wgms.y0)
    assert len(wgms) == len(BELART20_PERIODS)
    for r in belart20_periods().itertuples():
        if (r.RGIId, r.y0, r.y1) in WGMS_ERRORS:
            continue
        match = wgms[
            (wgms.glacier_id == WGMS_IDS[r.RGIId])
            & (wgms.y0 == r.y0)
            & (wgms.y1 == r.y1)
        ]
        assert len(match) == 1, (r.RGIId, r.y0, r.y1)
        assert match.mwe_per_year.iloc[0] == pytest.approx(
            r.mwe_per_year_published, abs=0.011
        ), (r.RGIId, r.y0, r.y1)


def _rgi_available():
    try:
        from data_processing.glacier_utils import get_region_shape_file

        return os.path.exists(get_region_shape_file("06"))
    except Exception:
        return False


@pytest.mark.skipif(
    not _rgi_available(), reason="RGI 6.2 region 06 not available locally"
)
def test_crosswalk_maps_every_rgi_outline_once():
    table = table_RGI62_to_Belart20()
    assert table.RGIId.is_unique
    expected = {
        r: g for g, (_, rgi_ids, _) in BELART20_GLACIERS.items() for r in rgi_ids
    }
    assert dict(zip(table.RGIId, table.custom_id)) == expected
    assert (table.groupby("custom_id").frac_custom.sum().round(6) == 1).all()
    assert table_RGI62_to_Belart20(region_id=11).empty


@pytest.mark.skipif(
    not _rgi_available(), reason="RGI 6.2 region 06 not available locally"
)
def test_outlines_have_the_area_of_their_epoch():
    # The outlines date from 1999-2004: their area should be close to the mean area
    # of the period of the paper around 2000. Barkárdals- and Tungnahryggsjökull and
    # Eiríksjökull are delineated more tightly in RGI 6.2 than in the paper.
    outlines = load_belart20_outlines().set_index("BELART20_ID")
    periods = belart20_periods()
    tolerance = {"BAR": 0.15, "EIR": 0.15}
    for code, area in outlines.area_rgi.items():
        p = periods[periods.RGIId == code]
        mid = p.survey_from + (p.survey_to - p.survey_from) / 2
        around_2000 = p.area_mean_km2.iloc[
            (mid - pd.Timestamp("2001-01-01")).abs().argmin()
        ]
        assert area == pytest.approx(around_2000, rel=tolerance.get(code, 0.1)), code
