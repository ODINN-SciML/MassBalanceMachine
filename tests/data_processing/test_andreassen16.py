"""Tests of the Andreassen16 source: the periods of Table 4 of Andreassen et al. (2016),
their dates, units and uncertainty, and the crosswalk to RGI 6.2.

The cross-check against WGMS reads the Fluctuations of Glaciers database from
`.data/WGMS`, and the outline tests read the RGI 6.2 outlines from OGGM's local copy;
each is skipped when its data is missing.
"""

import os

import pandas as pd
import pytest

from data_processing.andreassen16 import (
    ANDREASSEN16_GLACIERS,
    andreassen16_periods,
    geodetic_target_Andreassen16,
    load_andreassen16_outlines,
    table_RGI62_to_Andreassen16,
)
from data_processing.product_utils import data_path


def test_every_glacier_of_the_paper_has_periods():
    periods = andreassen16_periods()
    assert len(periods) == 21
    assert set(periods.RGIId) == set(ANDREASSEN16_GLACIERS)


def test_periods_of_a_glacier_follow_each_other():
    for _, df in andreassen16_periods().groupby("RGIId"):
        df = df.sort_values("FROM_DATE")
        assert (df.TO_DATE.to_numpy()[:-1] == df.FROM_DATE.to_numpy()[1:]).all()


def test_periods_cover_whole_mass_balance_years():
    periods = andreassen16_periods()
    assert (periods.FROM_DATE.dt.strftime("%m-%d") == "10-01").all()
    assert (periods.TO_DATE.dt.strftime("%m-%d") == "10-01").all()
    assert (periods.survey_from.dt.year <= periods.y0).all()
    assert (periods.survey_to.dt.year == periods.y1).all()
    # Engabreen: surveyed in 1968, compared from the first glaciological year on
    row = andreassen16_periods().set_index(["RGIId", "y0"]).loc[("ENG", 1969)]
    assert row.survey_from == pd.Timestamp("1968-08-25")
    assert row.FROM_DATE == pd.Timestamp("1969-10-01")
    assert row.TO_DATE == pd.Timestamp("2001-10-01")
    assert row.n_years == 32
    assert row.cumulative_mwe == pytest.approx(-0.03 * 32)


def test_sigma_adds_the_errors_in_quadrature():
    # Sect. 3.4.3: the geodetic balance of Nigardsbreen is known to 0.16 m w.e. per
    # year for 1964-1984 and to 0.08 for 1984-2013
    nig = andreassen16_periods().set_index(["RGIId", "y0"])
    assert nig.loc[("NIG", 1964)].sigma_mwe_per_year == pytest.approx(0.16, abs=0.005)
    assert nig.loc[("NIG", 1984)].sigma_mwe_per_year == pytest.approx(0.08, abs=0.005)


def test_calving_glaciers_are_left_out_by_default():
    assert "AUS" not in set(geodetic_target_Andreassen16().RGIId)
    assert len(geodetic_target_Andreassen16()) == 20
    assert len(geodetic_target_Andreassen16(exclude_calving=False)) == 21


def test_single_period_and_filters():
    target = geodetic_target_Andreassen16(
        min_year=1970, max_year=2010, multi_period=False
    )
    assert target.RGIId.is_unique
    assert target.y0.min() >= 1970 and target.y1.max() <= 2010
    # Nigardsbreen has no period inside 1970-2010
    assert "NIG" not in set(target.RGIId)
    # Hellstugubreen: 1980-1997 is the longest period starting after 1970
    assert tuple(target.set_index("RGIId").loc["HEL", ["y0", "y1"]]) == (1980, 1997)


WGMS_CHANGE = os.path.join(
    data_path, "WGMS", "DOI-WGMS-FoG-2026-02-10", "data", "change.csv"
)
# glacier code: WGMS id of the glacier
WGMS_IDS = {
    "ALF": 317,
    "HAN": 322,
    "NIG": 290,
    "AUS": 321,
    "REM": 2296,
    "STO": 302,
    "HEL": 300,
    "GRA": 299,
    "ENG": 298,
    "LAN": 323,
}


@pytest.mark.skipif(not os.path.exists(WGMS_CHANGE), reason="WGMS FoG not available")
def test_survey_dates_and_areas_match_wgms():
    # WGMS holds every period between the survey dates of Table 2, with its mean area
    wgms = pd.read_csv(WGMS_CHANGE)
    for r in andreassen16_periods().itertuples():
        match = wgms[
            (wgms.glacier_id == WGMS_IDS[r.RGIId])
            & (pd.to_datetime(wgms.begin_date) == r.survey_from)
            & (pd.to_datetime(wgms.end_date) == r.survey_to)
        ]
        assert len(match) == 1, (r.RGIId, r.y0, r.y1)
        assert match.area.iloc[0] / 1e6 == pytest.approx(r.area_mean_km2, abs=0.001)


def _rgi_available():
    try:
        from data_processing.glacier_utils import get_region_shape_file

        return os.path.exists(get_region_shape_file("08"))
    except Exception:
        return False


@pytest.mark.skipif(
    not _rgi_available(), reason="RGI 6.2 region 08 not available locally"
)
def test_crosswalk_maps_one_rgi_outline_per_glacier():
    table = table_RGI62_to_Andreassen16()
    expected = {rgi_id: g for g, (_, rgi_id, _, _) in ANDREASSEN16_GLACIERS.items()}
    assert dict(zip(table.RGIId, table.custom_id)) == expected
    assert table_RGI62_to_Andreassen16(region_id=6).empty


@pytest.mark.skipif(
    not _rgi_available(), reason="RGI 6.2 region 08 not available locally"
)
def test_outlines_have_the_area_of_their_epoch():
    # The outlines date from 1999-2006: their area should be close to the mean area of
    # the period of the paper spanning that time. Nigardsbreen is the hydrological
    # basin in the paper, larger than the glacier of RGI 6.2.
    outlines = load_andreassen16_outlines().set_index("ANDREASSEN16_ID")
    assert outlines.o2_region.to_dict() == {
        "ALF": "02",
        "AUS": "02",
        "ENG": "01",
        "GRA": "03",
        "HAN": "02",
        "HEL": "03",
        "LAN": "01",
        "NIG": "02",
        "REM": "02",
        "STO": "03",
    }
    periods = andreassen16_periods()
    for code, area in outlines.area_rgi.items():
        p = periods[periods.RGIId == code]
        mid = p.survey_from + (p.survey_to - p.survey_from) / 2
        around_2003 = p.area_mean_km2.iloc[
            (mid - pd.Timestamp("2003-01-01")).abs().argmin()
        ]
        if code == "NIG":
            assert area < 0.9 * around_2003
        else:
            assert area == pytest.approx(around_2003, rel=0.1), code
