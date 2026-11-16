"""Hidden-layer activation of the precipitation correction networks of the TIlike model
(`model.bias_cor.activation` and `model.bias_cor_elev.activation`)."""

import pytest
import torch.nn as nn

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


def _model(bias_cor_options=None, bias_cor_elev_options=None):
    params = {
        "inputs": INPUTS,
        "grad_T": {"inputs": ["t2m"], "layers": [4]},
        "bias_T": {"inputs": ["POINT_LAT", "POINT_LON"], "layers": [4]},
        "sw_contrib": {"inputs": ["ELEVATION_DIFFERENCE"], "layers": [4]},
        "cor_acc": {"inputs": ["t2m"], "layers": [4]},
        "cor_abl": {"inputs": ["t2m"], "layers": [4]},
        "bias_cor": {
            "inputs": ["POINT_LAT", "POINT_LON"],
            "layers": [4, 4],
            "p_min": 0.1,
            "p_max": 2.0,
            **(bias_cor_options or {}),
        },
        "bias_cor_elev": {
            "inputs": ["ELEVATION_DIFFERENCE"],
            "layers": [4, 4],
            **(bias_cor_elev_options or {}),
        },
    }
    return TILikeModel(params, BOUNDS)


def _activations(net):
    return {type(m) for m in net if not isinstance(m, nn.Linear)}


def test_default_is_relu():
    model = _model()
    assert _activations(model.bias_cor) == {nn.ReLU}
    assert _activations(model.bias_cor_elev) == {nn.ReLU}


@pytest.mark.parametrize(
    "bias_cor, bias_cor_elev",
    [("leaky_relu", "relu"), ("relu", "leaky_relu"), ("leaky_relu", "leaky_relu")],
)
def test_each_network_has_its_own_activation(bias_cor, bias_cor_elev):
    model = _model({"activation": bias_cor}, {"activation": bias_cor_elev})
    expected = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}
    assert _activations(model.bias_cor) == {expected[bias_cor]}
    assert _activations(model.bias_cor_elev) == {expected[bias_cor_elev]}
    # The other sub-networks keep ReLU
    assert _activations(model.bias_T) == {nn.ReLU}


def test_unknown_activation_fails():
    with pytest.raises(AssertionError, match="bias_cor_elev.activation"):
        _model(bias_cor_elev_options={"activation": "gelu"})
