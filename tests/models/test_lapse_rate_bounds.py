"""Range of the temperature lapse rate of the TIlike model (`model.grad_T.min/max`)."""

import pytest
import torch

from models.TorchNeuralNetworkRegressor import TILikeModel

INPUTS = [
    "ELEVATION_DIFFERENCE",
    "t2m",
    "tp_sum",
    "POINT_LAT",
    "POINT_LON",
    "svf",
    "ssrd",
]
BOUNDS = {
    "ELEVATION_DIFFERENCE": (0, 1000),
    "t2m": (-20, 15),
    "tp_sum": (0, 1),
    "POINT_LAT": (63, 67),
    "POINT_LON": (-25, -13),
    "svf": (0, 1),
    "ssrd": (-10e6, 10e6),
}


def _model(grad_T_bounds=None, branch="sw_contrib"):
    params = {
        "inputs": INPUTS,
        "grad_T": {"inputs": ["t2m"], "layers": [4], **(grad_T_bounds or {})},
        "bias_T": {"inputs": ["POINT_LAT", "POINT_LON"], "layers": [4]},
        "cor_acc": {"inputs": ["t2m"], "layers": [4]},
        "cor_abl": {"inputs": ["t2m"], "layers": [4]},
    }
    if branch == "sw_contrib":
        params["sw_contrib"] = {"inputs": ["ELEVATION_DIFFERENCE"], "layers": [4]}
    else:
        params["cor_dir"] = {"inputs": ["t2m"], "layers": [4]}
        params["cor_terrain"] = {"inputs": ["t2m"], "layers": [4]}
    return TILikeModel(params, BOUNDS)


def _extremes(model):
    """Lapse rate for a grad_T output of -inf and +inf."""
    last = model.grad_T[-1]
    torch.nn.init.zeros_(last.weight)
    with torch.no_grad():
        values = []
        for b in (-50.0, 50.0):
            last.bias.fill_(b)
            values.append(float(model._lapse_rate(torch.zeros((1, 1)))[0]))
    return values


@pytest.mark.parametrize("branch", ["sw_contrib", "cor_dir"])
def test_default_range(branch):
    assert _extremes(_model(branch=branch)) == pytest.approx([-2.0, -15.0])


def test_default_formula_is_unchanged():
    model = _model()
    x = torch.rand((10, 1))
    with torch.no_grad():
        former = -2 - 13 * torch.sigmoid(model.grad_T(x)[:, 0])
        assert torch.allclose(model._lapse_rate(x), former)


def test_range_from_the_config():
    assert _extremes(_model({"min": -7.5, "max": -5.5})) == pytest.approx([-5.5, -7.5])


def test_invalid_range_is_refused():
    with pytest.raises(AssertionError):
        _model({"min": -5.0, "max": -7.0})
