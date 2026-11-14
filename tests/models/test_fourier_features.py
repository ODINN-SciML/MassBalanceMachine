"""Fourier features of the inputs of the temperature bias and precipitation correction
networks of the TIlike model (`model.bias_T.fourier_features` and
`model.bias_cor.fourier_features`)."""

import pandas as pd
import pytest
import torch

from models.TorchNeuralNetworkRegressor import FourierFeatures, TILikeModel

INPUTS = ["ELEVATION_DIFFERENCE", "t2m", "tp_sum", "POINT_LAT", "POINT_LON"]
BOUNDS = {
    "ELEVATION_DIFFERENCE": (0, 1000),
    "t2m": (-20, 15),
    "tp_sum": (0, 1),
    "POINT_LAT": (59.5, 70.5),
    "POINT_LON": (5.0, 22.0),
}
FF = {"n_frequencies": 8, "sigma": 10.0}


def _model(bias_T_options=None, bias_cor_options=None):
    params = {
        "inputs": INPUTS,
        "grad_T": {"inputs": ["t2m"], "layers": [4]},
        "bias_T": {
            "inputs": ["POINT_LAT", "POINT_LON"],
            "layers": [4, 4],
            **(bias_T_options or {}),
        },
        "sw_contrib": {"inputs": ["ELEVATION_DIFFERENCE"], "layers": [4]},
        "cor_acc": {"inputs": ["t2m"], "layers": [4]},
        "cor_abl": {"inputs": ["t2m"], "layers": [4]},
        "bias_cor": {
            "inputs": ["POINT_LAT", "POINT_LON"],
            "layers": [4, 4],
            **(bias_cor_options or {}),
        },
    }
    return TILikeModel(params, BOUNDS)


def test_disabled_by_default():
    model = _model()
    assert not any(isinstance(m, FourierFeatures) for m in model.modules())
    # Same layout as before the option: older checkpoints still load
    assert "bias_T.0.weight" in model.state_dict()
    assert "bias_cor.0.weight" in model.state_dict()


@pytest.mark.parametrize("include_input, n_features", [(True, 2 + 16), (False, 16)])
def test_each_network_has_its_own_features(include_input, n_features):
    model = _model({"fourier_features": {**FF, "include_input": include_input}})
    assert isinstance(model.bias_T[0], FourierFeatures)
    assert model.bias_T[1].in_features == n_features
    # bias_cor keeps its inputs
    assert not isinstance(model.bias_cor[0], FourierFeatures)
    assert model.bias_cor[0].in_features == 2

    model = _model(bias_cor_options={"fourier_features": FF})
    assert isinstance(model.bias_cor[0], FourierFeatures)
    assert not isinstance(model.bias_T[0], FourierFeatures)


def test_frequencies_are_saved_and_reproducible():
    a = _model({"fourier_features": {**FF, "seed": 3}})
    b = _model({"fourier_features": {**FF, "seed": 3}})
    c = _model({"fourier_features": {**FF, "seed": 4}})
    assert torch.equal(a.bias_T[0].B, b.bias_T[0].B)
    assert not torch.equal(a.bias_T[0].B, c.bias_T[0].B)
    assert "bias_T.0.B" in a.state_dict()
    c.load_state_dict(a.state_dict())
    assert torch.equal(c.bias_T[0].B, a.bias_T[0].B)


def test_predictions_from_df():
    model = _model({"fourier_features": FF}, {"fourier_features": FF})
    df = pd.DataFrame({"POINT_LAT": [69.7, 70.1], "POINT_LON": [20.15, 21.7]})
    bias = model.predict_temp_bias_from_df(df)
    p_cor = model.predict_bias_cor_from_df(df)
    assert bias.shape == (2,) and p_cor.shape == (2,)
    assert torch.all(bias.abs() <= model.b_max)


@pytest.mark.parametrize(
    "options, match",
    [
        ({"n_frequencies": 8}, "needs sigma"),
        ({**FF, "scale": 1.0}, "Unknown bias_T.fourier_features"),
        ({"n_frequencies": 0, "sigma": 1.0}, "n_frequencies > 0"),
    ],
)
def test_invalid_options_fail(options, match):
    with pytest.raises(AssertionError, match=match):
        _model({"fourier_features": options})
