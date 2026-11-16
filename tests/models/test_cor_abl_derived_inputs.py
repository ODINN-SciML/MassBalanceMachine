"""Inputs of cor_abl derived by the TIlike model from its own inputs
(`DERIVED_COR_ABL_INPUTS`): wind speed from u10 and v10."""

import math

import pytest
import torch

from models.TorchNeuralNetworkRegressor import TILikeModel

INPUTS = ["ELEVATION_DIFFERENCE", "t2m", "tp_sum", "str", "u10", "v10"]
BOUNDS = {
    "ELEVATION_DIFFERENCE": (0, 1000),
    "t2m": (-20, 15),
    "tp_sum": (0, 1),
    "str": (-1e7, 1e7),
    "u10": (-10, 10),
    "v10": (-10, 10),
}


def _model(cor_abl_inputs, inputs=INPUTS, bounds=BOUNDS):
    params = {
        "inputs": inputs,
        "grad_T": {"inputs": ["t2m"], "layers": [4]},
        "bias_T": {"inputs": ["ELEVATION_DIFFERENCE"], "layers": [4]},
        "sw_contrib": {"inputs": ["ELEVATION_DIFFERENCE"], "layers": [4]},
        "cor_acc": {"inputs": ["t2m"], "layers": [4]},
        "cor_abl": {"inputs": cor_abl_inputs, "layers": [4]},
    }
    torch.manual_seed(0)
    return TILikeModel(params, dict(bounds))


def _norm(value, column):
    lower, upper = BOUNDS[column]
    return (value - lower) / (upper - lower)


def test_wind_speed_from_u10_and_v10():
    model = _model(["str", "wind_speed"])
    assert model.cor_abl[0].in_features == 2
    x = torch.rand((2, len(INPUTS)))
    x[0, INPUTS.index("u10")] = _norm(0.6, "u10")
    x[0, INPUTS.index("v10")] = _norm(-0.8, "v10")
    x[1, INPUTS.index("u10")] = _norm(0.0, "u10")
    x[1, INPUTS.index("v10")] = _norm(0.0, "v10")
    inp = model._cor_abl_inputs(x)
    # str first, as read from the inputs, then the wind speed normalized on [0, 2]
    assert torch.allclose(inp[:, 0], x[:, INPUTS.index("str")])
    assert inp[:, 1].tolist() == pytest.approx([0.5, 0.0], abs=1e-6)
    # The forward pass and the intermediate variables use it
    with torch.no_grad():
        model(x)
        assert model.get_cor_T(x)[9].shape == (2,)


def test_wind_speed_bounds_from_the_normalization():
    model = _model(["wind_speed"], bounds=BOUNDS | {"wind_speed": (0.0, 5.0)})
    x = torch.zeros((1, len(INPUTS)))
    x[0, INPUTS.index("u10")] = _norm(3.0, "u10")
    x[0, INPUTS.index("v10")] = _norm(4.0, "v10")
    assert model._cor_abl_inputs(x)[0, 0].item() == pytest.approx(1.0, abs=1e-6)


def test_inputs_without_derived_ones_are_unchanged():
    model = _model(["v10", "str"])
    x = torch.rand((3, len(INPUTS)))
    expected = x[:, sorted([INPUTS.index("v10"), INPUTS.index("str")])]
    assert torch.equal(model._cor_abl_inputs(x), expected)


def test_wind_speed_needs_u10_and_v10():
    inputs = [c for c in INPUTS if c != "v10"]
    with pytest.raises(AssertionError, match="needs \\['v10'\\]"):
        _model(["wind_speed"], inputs=inputs)
