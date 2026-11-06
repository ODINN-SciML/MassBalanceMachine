"""Tests of the NVE stake measurements: the misdated ones are dropped, so that the monthly
table can be padded. Skipped when the NVE dataset has not been downloaded."""

import os

import pandas as pd
import pytest

from data_retrieval.nve import point_mb_path


@pytest.mark.skipif(
    not os.path.exists(point_mb_path()), reason="NVE point dataset not downloaded"
)
def test_points_fit_their_hydrological_year():
    from data_preprocessing.nve import MAX_PADDING_MONTHS, load_points
    from data_processing.utils.hydro_year import _compute_months_before_after

    points = load_points()
    dates = points.assign(
        FROM_DATE=pd.to_datetime(points.FROM_DATE),
        TO_DATE=pd.to_datetime(points.TO_DATE),
    )
    padding = _compute_months_before_after(dates[dates.TO_DATE >= dates.FROM_DATE])
    assert padding.MONTHS_BEFORE.max() <= MAX_PADDING_MONTHS
    assert padding.MONTHS_AFTER.max() <= MAX_PADDING_MONTHS
    # The winter balance of Langfjordjøkelen from 7 January 2013 to 9 May 2014 is gone
    assert not ((points.FROM_DATE == "20130107") & (points.TO_DATE == "20140509")).any()


@pytest.mark.skipif(
    not os.path.exists(point_mb_path()), reason="NVE point dataset not downloaded"
)
def test_inconsistent_points_are_dropped():
    from data_preprocessing.nve import load_points

    points = load_points()
    # Annual balances of 36 days (1988-09-12 to 1988-10-18) or ending in May
    assert not ((points.FROM_DATE == "19880912") & (points.TO_DATE == "19881018")).any()
    assert not ((points.FROM_DATE == "19930929") & (points.TO_DATE == "19940515")).any()
    # Winter balance ending in September 1967
    assert not ((points.FROM_DATE == "19661118") & (points.TO_DATE == "19670929")).any()
    # Ålfotbreen and Hansebreen winter 1982, one value copied over the stakes
    assert not ((points.PERIOD == "winter") & (points.FROM_DATE == "19811010")).any()
    # Nigardsbreen 998 m in summer 1982, at the position of a stake of RGI60-08.01121
    assert not ((points.POINT_ELEVATION == 998) & (points.TO_DATE == "19821024")).any()
