"""Area weighting of the geodetic loss term (`training.areaWeightGeo`)."""

import numpy as np
import pytest
import torch

from training.training import compute_geo_loss, geodeticAreaWeights


class _Identity:
    """A model predicting its single input feature."""

    def forward(self, x):
        return x


class _Loader:
    def __init__(self, areas):
        self.areas = areas

    def geodetic_areas(self, g):
        return np.asarray(self.areas[g], dtype=float)


def test_weights_follow_the_area_and_average_to_one():
    weights = geodeticAreaWeights(_Loader({"A": [30.0, 28.0], "B": [2.0]}), ["A", "B"])
    np.testing.assert_allclose(weights["A"].numpy(), [1.5, 1.4])
    np.testing.assert_allclose(weights["B"].numpy(), [0.1])
    assert np.concatenate(
        [w.numpy() for w in weights.values()]
    ).mean() == pytest.approx(1)


def _geo_loss(windowWeights):
    # Two months, one point each, and two windows of one month each
    geoGrid = torch.tensor([[-0.5], [0.3]])
    meta = {
        "GLWD_M_ID_int": torch.tensor([0, 1]),
        "nunique_glwd_m_ids": 2,
        "geo_window_weights": torch.eye(2),
    }
    loss, _ = compute_geo_loss(
        _Identity(),
        geoGrid,
        None,
        torch.tensor([-0.2, 0.0]),
        torch.tensor([0.1, 0.2]),
        [("2000-01-01", "2000-02-01"), ("2000-02-01", "2000-03-01")],
        meta,
        "quad",
        windowWeights=windowWeights,
    )
    return float(loss)


def test_loss_term_of_every_window_is_weighted():
    first, second = _geo_loss(torch.tensor([1.0, 0.0])), _geo_loss(
        torch.tensor([0.0, 1.0])
    )
    assert first != second
    # Without weights every window counts the same
    assert _geo_loss(None) == pytest.approx(first + second)
    assert _geo_loss(torch.tensor([1.5, 0.5])) == pytest.approx(
        1.5 * first + 0.5 * second
    )
