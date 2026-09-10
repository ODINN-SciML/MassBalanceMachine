"""Tests of the support for several geodetic windows per glacier: the choice of the
GLAMOS windows, the weights that turn the monthly predictions into the rate of every
window, and the geodetic loss built on them.

None of them needs downloaded data: the GLAMOS table and the grid metadata are
synthetic.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from data_processing.glamos import select_glamos_windows
from data_processing.gridded_utils import (
    MONTH_TO_ID,
    _period_key,
    geodetic_window_weights,
)
from training.training import timeWindowGeodeticLoss


def _glamos_table(windows):
    """A GLAMOS-like table from (sgi_id, y0, y1, sigma) tuples."""
    df = pd.DataFrame(windows, columns=["SGI-ID", "y0", "y1", "sigma"])
    df["dur"] = df.y1 - df.y0
    df["covered"] = 100.0
    return df


def _chain(selected, sgi_id):
    rows = selected.loc[[sgi_id]]
    return list(zip(rows.y0, rows.y1))


def test_single_period_keeps_the_longest_window():
    glamos = _glamos_table(
        [
            ("A", 1956, 1999, 0.3),
            ("A", 1956, 1967, 0.1),
            ("A", 1967, 1999, 0.1),
            ("B", 1960, 1990, 0.2),
            ("B", 1960, 1990, 0.1),
        ]
    )
    selected = select_glamos_windows(glamos, max_year=2000, min_window_years=10)
    assert selected.index.is_unique
    assert _chain(selected, "A") == [(1956, 1999)]
    # ties on the duration are broken on sigma
    assert selected.loc["B"].sigma == 0.1


def test_multi_period_chains_non_overlapping_windows():
    glamos = _glamos_table(
        [
            # nested and overlapping windows around a chain of three
            ("A", 1956, 1999, 0.1),
            ("A", 1956, 1982, 0.1),
            ("A", 1956, 1967, 0.2),
            ("A", 1967, 1982, 0.2),
            ("A", 1967, 1988, 0.2),
            ("A", 1982, 1999, 0.2),
            ("A", 1988, 1999, 0.2),
            # only one window: nothing to chain
            ("B", 1960, 1990, 0.2),
        ]
    )
    selected = select_glamos_windows(
        glamos, max_year=2000, min_window_years=10, multi_period=True
    )
    # three windows, and among the chains of three the ones covering 1956-1999;
    # (1967, 1982) + (1982, 1999) and (1967, 1988) + (1988, 1999) tie on the
    # duration and on sigma, so both are acceptable
    chain_a = _chain(selected, "A")
    assert len(chain_a) == 3
    assert chain_a[0] == (1956, 1967) and chain_a[-1][1] == 1999
    assert all(prev[1] <= nxt[0] for prev, nxt in zip(chain_a, chain_a[1:]))
    assert _chain(selected, "B") == [(1960, 1990)]


def test_multi_period_prefers_coverage_then_sigma():
    glamos = _glamos_table(
        [
            # two chains of two windows: the second one covers more years
            ("A", 1960, 1972, 0.1),
            ("A", 1972, 1985, 0.1),
            ("A", 1972, 1995, 0.3),
            # two chains of equal count and coverage: the lower sigma wins
            ("B", 1960, 1975, 0.1),
            ("B", 1975, 1990, 0.4),
            ("B", 1960, 1970, 0.1),
            ("B", 1970, 1990, 0.1),
        ]
    )
    selected = select_glamos_windows(
        glamos, max_year=2000, min_window_years=10, multi_period=True
    )
    assert _chain(selected, "A") == [(1960, 1972), (1972, 1995)]
    assert _chain(selected, "B") == [(1960, 1970), (1970, 1990)]


def test_multi_period_windows_respect_the_eligibility_criteria():
    glamos = _glamos_table(
        [
            ("A", 1940, 1956, 0.1),  # starts before min_year
            ("A", 1956, 1964, 0.1),  # shorter than min_window_years
            ("A", 1964, 1980, 0.1),
            ("A", 1980, 1995, 0.1),
            ("A", 1995, 2010, 0.1),  # ends after max_year
        ]
    )
    selected = select_glamos_windows(
        glamos, max_year=2000, min_window_years=10, min_year=1950, multi_period=True
    )
    assert _chain(selected, "A") == [(1964, 1980), (1980, 1995)]


def _monthly_metadata(years):
    rows = [{"YEAR": y, "MONTHS": m} for y in years for m in MONTH_TO_ID]
    meta = pd.DataFrame(rows)
    meta["GLWD_M_ID_int"] = np.arange(len(meta))
    return meta


def test_window_weights():
    meta = _monthly_metadata([2000, 2001, 2002])
    windows = [
        (np.datetime64("2000-01-01"), np.datetime64("2001-01-01")),
        (np.datetime64("2001-01-01"), np.datetime64("2003-01-01")),
    ]
    weights = geodetic_window_weights(meta, windows)
    assert weights.shape == (2, 36)
    # the product with a constant monthly value gives 12 times that value
    np.testing.assert_allclose(weights.sum(axis=1), 12)
    np.testing.assert_allclose(weights[0, :12], 1)
    np.testing.assert_allclose(weights[0, 12:], 0)
    np.testing.assert_allclose(weights[1, :12], 0)
    np.testing.assert_allclose(weights[1, 12:], 0.5)
    # windows given as calendar years, as for Hugonnet21
    np.testing.assert_allclose(geodetic_window_weights(meta, [(2000, 2003)]), 1 / 3)


def test_window_weights_refuse_a_window_the_grid_does_not_cover():
    meta = _monthly_metadata([2000, 2001])
    with pytest.raises(AssertionError):
        geodetic_window_weights(meta, [(2000, 2003)])


def test_period_key():
    one = [(np.datetime64("1956-01-01"), np.datetime64("1999-01-01"))]
    # unchanged, so that the grids assembled before several windows were supported
    # are still found
    assert _period_key(one) == "1956-01-01_1999-01-01"
    assert _period_key(range(2000, 2003)) == "2000_2001_2002"

    many = [
        (np.datetime64(f"{1950 + 3 * i}-01-01"), np.datetime64(f"{1963 + 3 * i}-01-01"))
        for i in range(14)
    ]
    key = _period_key(many)
    assert key.startswith("1950-01-01_2002-01-01_14w_")
    assert len(key) < 255
    assert key != _period_key(many[:-1])


def test_loss_with_one_window_matches_a_mean_over_the_window():
    n_months = 240
    pred = torch.randn(n_months)
    target, sigma = torch.tensor([-0.7]), torch.tensor([0.15])
    rate = pred.sum() * 12 / n_months
    weights = torch.full((1, n_months), 12 / n_months)
    loss, ypred = timeWindowGeodeticLoss(
        pred, target, sigma, weights, [(2000, 2020)], "quad"
    )
    torch.testing.assert_close(ypred, rate.view(1))
    torch.testing.assert_close(loss, ((rate - target) / sigma) ** 2)
    loss, _ = timeWindowGeodeticLoss(
        pred, target, sigma, weights, [(2000, 2020)], "linear"
    )
    torch.testing.assert_close(loss, (rate - target) ** 2 / sigma)


def test_loss_has_one_term_per_window():
    meta = _monthly_metadata([2000, 2001, 2002])
    windows = [(2000, 2001), (2001, 2003)]
    weights = torch.from_numpy(geodetic_window_weights(meta, windows))
    # a monthly value of 1 in the first year and 2 afterwards
    pred = torch.cat([torch.ones(12), 2 * torch.ones(24)])
    target, sigma = torch.tensor([10.0, 20.0]), torch.tensor([1.0, 2.0])
    loss, ypred = timeWindowGeodeticLoss(pred, target, sigma, weights, windows, "quad")
    torch.testing.assert_close(ypred, torch.tensor([12.0, 24.0]))
    torch.testing.assert_close(loss, torch.tensor([4.0, 4.0]))
