"""Normalization bounds of a model set in its config (`model.normalization`)."""

import pytest

import massbalancemachine as mbm
from cli.common import setNormalizationBounds
from config import DEFAULT_BNDS


def test_listed_bounds_replace_the_global_ones_and_the_others_are_kept():
    cfg = mbm.Config()
    setNormalizationBounds(
        cfg,
        {"model": {"normalization": {"POINT_LAT": [63, 67], "POINT_LON": [-25, -13]}}},
    )
    assert cfg.bnds["POINT_LAT"] == (63.0, 67.0)
    assert cfg.bnds["POINT_LON"] == (-25.0, -13.0)
    for k in set(DEFAULT_BNDS) - {"POINT_LAT", "POINT_LON"}:
        assert cfg.bnds[k] == DEFAULT_BNDS[k]
    # The global bounds are not modified
    assert DEFAULT_BNDS["POINT_LAT"] == (-90, 90)


@pytest.mark.parametrize("normalization", [None, {}])
def test_without_normalization_the_global_bounds_are_used(normalization):
    cfg = mbm.Config()
    params = {
        "model": {} if normalization is None else {"normalization": normalization}
    }
    setNormalizationBounds(cfg, params)
    assert cfg.bnds == DEFAULT_BNDS


@pytest.mark.parametrize("bounds", [[67, 63], [63, 63], [63]])
def test_invalid_bounds_are_refused(bounds):
    with pytest.raises(AssertionError):
        setNormalizationBounds(
            mbm.Config(), {"model": {"normalization": {"POINT_LAT": bounds}}}
        )
