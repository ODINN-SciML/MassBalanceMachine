"""Tests of the Fischer11 source: the periods of Table 3 of Fischer (2011), their
uncertainty and the inventory each one is gridded on.

None of them needs the Austrian inventories or a download.
"""

import pandas as pd
import pytest

from data_processing import fischer11

from data_processing.fischer11 import (
    FISCHER11_TOTALS,
    fischer11_periods,
    geodetic_target_Fischer11,
    outline_epoch_of_period,
    period_sigma_mwe_per_year,
)


def test_periods_add_up_to_the_totals_of_table_3():
    periods = fischer11_periods()
    sums = periods.groupby("RGIId").cumulative_mwe.sum()
    for glacier_id, total in FISCHER11_TOTALS.items():
        assert sums[glacier_id] == pytest.approx(total, abs=1e-9)


def test_periods_of_a_glacier_follow_each_other():
    for _, df in fischer11_periods().groupby("RGIId"):
        df = df.sort_values("y0")
        assert (df.y0.to_numpy()[1:] == df.y1.to_numpy()[:-1]).all()


def test_a_period_covers_the_hydrological_years_between_its_dems():
    row = fischer11_periods().set_index(["RGIId", "y0"]).loc[("2133", 1969)]
    assert row.FROM_DATE == pd.Timestamp("1969-10-01")
    assert row.TO_DATE == pd.Timestamp("1997-10-01")
    assert row.n_years == 28
    assert row.mwe_per_year == pytest.approx(-2.6 / 28)


def test_uncertainty_reproduces_the_budget_of_the_paper():
    assert period_sigma_mwe_per_year(10, "TP", "TP") == pytest.approx(2.7)
    assert period_sigma_mwe_per_year(10, "AL", "AL") == pytest.approx(0.702)


def test_an_unknown_dem_method_is_refused():
    with pytest.raises(AssertionError):
        period_sigma_mwe_per_year(10, "TP", "radar")


def test_a_period_is_gridded_on_the_inventory_of_its_first_dem():
    # Hintereisferner, surveyed for GI 2 in 1997
    assert outline_epoch_of_period(1969, 1997) == 1969
    assert outline_epoch_of_period(1997, 1997) == 1998
    # no inventory at the first DEM: GI 1
    assert outline_epoch_of_period(1953, 1997) == 1969
    assert outline_epoch_of_period(1991, 1997) == 1969
    # Jamtalferner, surveyed for GI 2 in 2002: 1996 has no inventory
    assert outline_epoch_of_period(1996, 2002) == 1969
    assert outline_epoch_of_period(2002, 2002) == 1998


def test_every_period_ending_before_2000_is_on_gi1():
    periods = fischer11_periods()
    assert (periods[periods.y1 <= 2000].outline_epoch == 1969).all()
    on_gi2 = periods[periods.outline_epoch == 1998]
    assert set(zip(on_gi2.RGIId, on_gi2.y0)) == {
        ("2125", 1997),
        ("2129", 1997),
        ("2133", 1997),
        ("13019", 2002),
    }


def test_target_keeps_one_inventory_at_a_time():
    assert (geodetic_target_Fischer11().outline_epoch == 1969).all()
    assert (geodetic_target_Fischer11(epoch=1998).outline_epoch == 1998).all()
    assert len(geodetic_target_Fischer11(epoch=None)) == len(fischer11_periods())


def test_target_filters():
    target = geodetic_target_Fischer11(max_year=1997, min_year=1960)
    assert target.y1.max() <= 1997 and target.y0.min() >= 1960
    assert set(geodetic_target_Fischer11(glacier_ids_to_keep=[5097]).RGIId) == {"5097"}
    assert geodetic_target_Fischer11(min_period_years=10).n_years.min() >= 10


def test_periods_stay_inside_the_years_they_are_given():
    # 1969-10 to 1979-10 touches the calendar years 1969 to 1979
    target = geodetic_target_Fischer11(
        glacier_ids_to_keep=["2125"], allowed_years=range(1969, 1980)
    )
    assert list(zip(target.y0, target.y1)) == [(1969, 1979)]
    target = geodetic_target_Fischer11(
        glacier_ids_to_keep=["2125"],
        allowed_years=range(1970, 1980),
        max_outside_years=1,
    )
    assert list(zip(target.y0, target.y1)) == [(1969, 1979)]


def test_single_period_is_the_longest_then_the_lowest_sigma():
    target = geodetic_target_Fischer11(multi_period=False, max_year=2000)
    assert target.RGIId.is_unique
    assert dict(zip(target.RGIId, zip(target.y0, target.y1))) == {
        "2125": (1979, 1991),
        "2129": (1971, 1997),
        "2133": (1969, 1997),
        "5097": (1969, 1998),
    }


def test_ties_on_length_go_to_the_lowest_sigma_or_the_most_recent(monkeypatch):
    # Table 3 has no two periods of one glacier with the same length: add one, ten
    # years of laser scans after the 1969-1979 photogrammetric period
    monkeypatch.setattr(
        fischer11,
        "FISCHER11_PERIODS",
        [("2125", 1969, 1979, 2.7, "AP", "AP"), ("2125", 1979, 1989, -9.0, "AL", "AL")],
    )
    by_sigma = geodetic_target_Fischer11(multi_period=False)
    assert list(zip(by_sigma.y0, by_sigma.y1)) == [(1979, 1989)]

    monkeypatch.setattr(
        fischer11,
        "FISCHER11_PERIODS",
        [("2125", 1969, 1979, 2.7, "AL", "AL"), ("2125", 1979, 1989, -9.0, "TP", "TP")],
    )
    by_sigma = geodetic_target_Fischer11(multi_period=False)
    assert list(zip(by_sigma.y0, by_sigma.y1)) == [(1969, 1979)]
    recent = geodetic_target_Fischer11(multi_period=False, tie_break="recent")
    assert list(zip(recent.y0, recent.y1)) == [(1979, 1989)]


def test_an_unknown_tie_break_is_refused():
    with pytest.raises(AssertionError):
        geodetic_target_Fischer11(multi_period=False, tie_break="longest")
