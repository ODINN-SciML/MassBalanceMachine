"""Scores of the glacier-wide geodetic figure: unweighted, weighted by the inverse
variance of the observations and, given the areas, area-weighted."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from plots import glacierWideScores, predVSTruthGlacierWide


def test_area_weighted_scores():
    target = np.array([0.0, 0.0])
    pred = np.array([0.1, 0.4])
    s = glacierWideScores(target, pred, err=np.array([0.1, 0.1]), area=[3.0, 1.0])
    assert s["bias"] == pytest.approx(0.25)
    # The large glacier counts three times as much as the small one
    assert s["bias_area"] == pytest.approx((3 * 0.1 + 0.4) / 4)
    assert s["rmse_area"] == pytest.approx(np.sqrt((3 * 0.01 + 0.16) / 4))
    assert "bias_area" not in glacierWideScores(target, pred, np.array([0.1, 0.1]))


@pytest.mark.parametrize("dense", [False, True])
def test_figure_writes_the_area_weighted_scores(dense):
    # Two windows for the first glacier, one for the second
    geoTarget = {"A": np.array([-0.5, -0.3]), "B": np.array([-0.2])}
    geoPred = {"A": np.array([-0.4, -0.1]), "B": np.array([-0.2])}
    geoErr = {"A": np.array([0.1, 0.2]), "B": np.array([0.1])}
    geoArea = {"A": np.array([10.0, 9.0]), "B": np.array([1.0])}
    fig = predVSTruthGlacierWide(
        geoTarget, geoPred, geoErr, geoArea=geoArea, dense=dense
    )
    texts = [t.get_text() for t in fig.axes[0].texts]
    plt.close(fig)
    (box,) = [t for t in texts if "glaciers" in t]
    assert "2 glaciers, 3 values" in box
    assert "bias $= +0.10$" in box
    # (10 * 0.1 + 9 * 0.2 + 0) / 20 = 0.14
    assert "area-weighted: bias $= +0.14$" in box
