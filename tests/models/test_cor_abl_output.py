"""Output of the ablation factor of the TIlike model (`model.cor_abl.output/min/max`)."""

import pytest
import torch
import torch.nn.functional as F

from models.TorchNeuralNetworkRegressor import TILikeModel

INPUTS = [
    "ELEVATION_DIFFERENCE",
    "t2m",
    "tp_sum",
    "POINT_LAT",
    "POINT_LON",
]
BOUNDS = {
    "ELEVATION_DIFFERENCE": (0, 1000),
    "t2m": (-20, 15),
    "tp_sum": (0, 1),
    "POINT_LAT": (63, 67),
    "POINT_LON": (-25, -13),
}


def _model(cor_abl_options=None):
    params = {
        "inputs": INPUTS,
        "grad_T": {"inputs": ["t2m"], "layers": [4]},
        "bias_T": {"inputs": ["POINT_LAT", "POINT_LON"], "layers": [4]},
        "sw_contrib": {"inputs": ["ELEVATION_DIFFERENCE"], "layers": [4]},
        "cor_acc": {"inputs": ["t2m"], "layers": [4]},
        "cor_abl": {"inputs": ["t2m"], "layers": [4], **(cor_abl_options or {})},
    }
    return TILikeModel(params, BOUNDS)


def _extremes(model):
    """Ablation factor for a raw cor_abl output of -inf and +inf."""
    last = model.cor_abl[-1]
    torch.nn.init.zeros_(last.weight)
    with torch.no_grad():
        values = []
        for b in (-50.0, 50.0):
            last.bias.fill_(b)
            values.append(float(model._ablation_factor(torch.zeros((1, 1)))[0]))
    return values


def test_default_is_softplus():
    model = _model()
    assert model.cor_abl_output == "softplus"
    x = torch.rand((10, 1))
    with torch.no_grad():
        assert torch.allclose(
            model._ablation_factor(x), F.softplus(model.cor_abl(x).view(-1))
        )
    assert _extremes(model) == pytest.approx([0.0, 50.0], abs=1e-6)


def test_state_dict_has_the_layers_of_the_softplus_network():
    # The output activation has no parameters, so checkpoints load either way
    keys = set(_model().state_dict())
    assert {k for k in keys if k.startswith("cor_abl.")} == {
        "cor_abl.0.weight",
        "cor_abl.0.bias",
        "cor_abl.2.weight",
        "cor_abl.2.bias",
    }


def test_sigmoid_bounds_from_the_config():
    model = _model({"output": "sigmoid", "min": 19.0, "max": 62.0})
    assert _extremes(model) == pytest.approx([19.0, 62.0])


def test_forward_uses_the_bounded_factor():
    model = _model({"output": "sigmoid", "min": 19.0, "max": 62.0})
    x = torch.rand((20, len(INPUTS)))
    with torch.no_grad():
        cor_abl = model.get_cor_T(x)[9]
    assert ((cor_abl >= 19.0) & (cor_abl <= 62.0)).all()


@pytest.mark.parametrize(
    "options",
    [
        {"output": "relu"},
        {"output": "sigmoid"},
        {"output": "sigmoid", "min": 19.0},
        {"output": "sigmoid", "min": 62.0, "max": 19.0},
        {"output": "sigmoid", "min": -1.0, "max": 19.0},
        {"min": 19.0, "max": 62.0},
    ],
)
def test_invalid_options_are_refused(options):
    with pytest.raises(AssertionError):
        _model(options)
