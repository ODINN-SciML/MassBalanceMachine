"""Tests of the GLACIOCLIM stake measurements: positions projected onto the right glaciers
whatever the label of their coordinate system, erroneous measurements dropped, and periods
fitting the monthly table. Skipped when the GLACIOCLIM shares have not been assembled."""

import os

import pandas as pd
import pytest

from data_preprocessing.glacioclim import assembled_path

pytestmark = pytest.mark.skipif(
    not os.path.exists(assembled_path()),
    reason="GLACIOCLIM stake table not assembled",
)

# RGI60-11 glaciers of the stakes of each GLACIOCLIM glacier (Argentière with Tour Noir,
# Mer de Glace with Leschaux and Talèfre)
EXPECTED_RGI = {
    "argentiere": {"RGI60-11.03638", "RGI60-11.03475"},
    "gebroulaz": {"RGI60-11.03671"},
    "mer_de_glace": {"RGI60-11.03643", "RGI60-11.03642", "RGI60-11.03466"},
    "saint_sorlin": {"RGI60-11.03674"},
    "sarennes": {"RGI60-11.03756"},
}


def test_stakes_lie_on_their_glacier():
    from data_preprocessing.glacioclim import load_points
    from data_processing.utils import get_rgi

    points = get_rgi(data=load_points(), region=11)
    assert points.POINT_LAT.between(45.0, 46.1).all()
    assert points.POINT_LON.between(6.0, 7.1).all()
    for glacier, rgi_ids in EXPECTED_RGI.items():
        found = set(points[points.GLACIER == glacier].RGIId.dropna())
        assert found == rgi_ids, (glacier, found)
    # Only the stakes below the RGI front of the earliest years fall off the outlines
    assert points.RGIId.isna().mean() < 0.02


def test_mislabelled_coordinates_are_lambert3():
    from data_preprocessing.glacioclim import raw_data, LAMBERT3

    raw = raw_data()
    # Winter 2018-2020 of the accumulation zone of Saint-Sorlin, labelled lambert2e
    mislabelled = (raw.GLACIER == "saint_sorlin") & (raw.CRS_LABEL == "lambert2e")
    assert mislabelled.sum() > 0
    assert (raw[mislabelled].EPSG == LAMBERT3).all()
    assert raw[mislabelled].POINT_LAT.between(45.15, 45.18).all()


def test_erroneous_measurements_are_dropped():
    from data_preprocessing.glacioclim import (
        ERRONEOUS_MEASUREMENTS,
        load_points,
        raw_data,
        stake_id,
    )

    raw = raw_data()
    keys = pd.Series(list(zip(stake_id(raw), raw.PERIOD, raw.TO_DATE)))
    # Each listed measurement is one row of the assembled table
    assert keys.isin(ERRONEOUS_MEASUREMENTS).sum() == len(ERRONEOUS_MEASUREMENTS)
    assert len(load_points()) == len(raw) - len(ERRONEOUS_MEASUREMENTS)


def test_points_are_clean():
    from data_preprocessing.glacioclim import load_points, MAX_PADDING_MONTHS
    from data_processing.utils.hydro_year import _compute_months_before_after

    points = load_points()
    assert set(points.PERIOD) == {"annual", "winter", "summer"}
    assert set(points.GLACIER) == set(EXPECTED_RGI)
    assert not (points.POINT_ELEVATION <= 0).any()
    # Saint-Sorlin summer 2017 at stake 14, -88.41 m w.e.
    assert points.POINT_BALANCE.min() > -15
    # Mer de Glace, Langue, winter 2011 at stake 2009/4: +5.88 m w.e. at 1963 m
    assert not ((points.PERIOD == "winter") & (points.POINT_BALANCE > 5.5)).any()
    # Sarennes 1996, annual balance 0.00 at every stake
    assert not (
        (points.GLACIER == "sarennes")
        & (points.TO_DATE == "19960914")
        & (points.PERIOD == "annual")
    ).any()

    dates = points.assign(
        FROM_DATE=pd.to_datetime(points.FROM_DATE),
        TO_DATE=pd.to_datetime(points.TO_DATE),
    )
    padding = _compute_months_before_after(dates[dates.TO_DATE >= dates.FROM_DATE])
    assert padding.MONTHS_BEFORE.max() <= MAX_PADDING_MONTHS
    assert padding.MONTHS_AFTER.max() <= MAX_PADDING_MONTHS
