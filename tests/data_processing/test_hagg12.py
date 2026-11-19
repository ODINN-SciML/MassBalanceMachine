"""Tests of the Hagg12 source: the periods of Tables 4-7 of Hagg et al. (2012), their
units and uncertainty, and the crosswalk to RGI 6.2.

Only the crosswalk reads the RGI 6.2 outlines, from OGGM's local copy; it is skipped
when that copy is missing.
"""

import os

import pandas as pd
import pytest

from data_processing.hagg12 import (
    HAGG12_GLACIERS,
    ICE_DENSITY_FACTOR,
    geodetic_target_Hagg12,
    hagg12_periods,
    period_sigma_mwe_per_year,
    survey_method,
    table_RGI62_to_Hagg12,
)


def test_periods_of_a_glacier_follow_each_other():
    for _, df in hagg12_periods().groupby("RGIId"):
        df = df.sort_values("y0")
        assert (df.y0.to_numpy()[1:] == df.y1.to_numpy()[:-1]).all()


def test_rates_are_converted_to_metres_at_850():
    row = hagg12_periods().set_index(["RGIId", "y0"]).loc[("HTF", 1970)]
    assert ICE_DENSITY_FACTOR == pytest.approx(850 / 900)
    assert row.mwe_per_year == pytest.approx(0.40 * 850 / 900)
    assert row.cumulative_mwe == pytest.approx(row.mwe_per_year * 11)
    assert row.FROM_DATE == pd.Timestamp("1970-10-01")
    assert row.TO_DATE == pd.Timestamp("1981-10-01")


def test_survey_methods():
    assert survey_method(1889) == "map"
    assert survey_method(1897) == "map"
    assert survey_method(1892) == "photogrammetry"
    assert survey_method(1924) == "photogrammetry"
    assert survey_method(2006) == "photogrammetry"
    assert survey_method(2009) == "ground"


def test_uncertainty_reproduces_the_paper():
    # Sect. 3.2.1, in metres of elevation per year, converted here at 850 kg m-3
    to_metres = 1000 / 850
    assert period_sigma_mwe_per_year(35, 1889, 1924) * to_metres == pytest.approx(
        0.17, abs=0.005
    )
    assert period_sigma_mwe_per_year(62, 1897, 1959) * to_metres == pytest.approx(
        0.10, abs=0.005
    )
    assert period_sigma_mwe_per_year(10, 1959, 1969) * to_metres == pytest.approx(0.2)


def test_area_referenced_by_a_rate():
    row = hagg12_periods().set_index(["RGIId", "y0"]).loc[("SSF", 1979)]
    assert row.area_mean_km2 == pytest.approx((31.4 + 12.3) / 2 / 100)
    # Table 2 has no area of Blaueis in 1999
    assert pd.isna(
        hagg12_periods().set_index(["RGIId", "y0"]).loc[("BLA", 1989)].area_mean_km2
    )


def test_target_filters_and_single_period():
    target = geodetic_target_Hagg12(min_year=1950, max_year=2000, min_period_years=10)
    assert target.y0.min() >= 1950 and target.y1.max() <= 2000
    assert target.n_years.min() >= 10
    single = geodetic_target_Hagg12(
        min_year=1950, max_year=2000, min_period_years=10, multi_period=False
    )
    assert single.RGIId.is_unique
    # Nördlicher Schneeferner: 1979-1990 is the only 11-year period
    assert tuple(single.set_index("RGIId").loc["NSF", ["y0", "y1"]]) == (1979, 1990)


def test_periods_stay_inside_the_years_they_are_given():
    target = geodetic_target_Hagg12(
        glacier_ids_to_keep=["HTF"],
        allowed_years=range(1970, 1982),
        max_outside_years=0,
    )
    assert list(zip(target.y0, target.y1)) == [(1970, 1981)]


def _rgi_available():
    try:
        from data_processing.glacier_utils import get_region_shape_file

        return os.path.exists(get_region_shape_file("11"))
    except Exception:
        return False


@pytest.mark.skipif(
    not _rgi_available(), reason="RGI 6.2 region 11 not available locally"
)
def test_crosswalk_maps_every_rgi_outline_once():
    table = table_RGI62_to_Hagg12()
    assert table.RGIId.is_unique
    expected = {r: g for g, (_, rgi_ids, _) in HAGG12_GLACIERS.items() for r in rgi_ids}
    assert dict(zip(table.RGIId, table.custom_id)) == expected
    assert (table.groupby("custom_id").frac_custom.sum().round(6) == 1).all()
    assert table_RGI62_to_Hagg12(region_id=12).empty
