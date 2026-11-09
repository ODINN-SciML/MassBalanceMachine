"""Shortwave radiation contribution R_sw of the TIlike model (`model.sw_contrib`):
scaling by the downward shortwave radiation, temperature gate and, with a surface
type, its ratio of ice to snow (`sfc_type.sw_ratio`)."""

import pytest
import torch
import torch.nn.functional as F

from data_processing.Dataset import Normalizer
from models.TorchNeuralNetworkRegressor import (
    SSRD_TO_MELT,
    TILikeModel,
    TILikeSfcModel,
)

INPUTS = ["ELEVATION_DIFFERENCE", "t2m", "tp_sum", "ssrd", "slope"]
BOUNDS = {
    "ELEVATION_DIFFERENCE": (0, 1000),
    "t2m": (-20, 15),
    "tp_sum": (0, 1),
    "ssrd": (0, 3e7),
    "slope": (0, 90),
}


def _params(sw_contrib=None, sfc_type=None):
    params = {
        "inputs": INPUTS,
        "grad_T": {"inputs": ["t2m"], "layers": [4]},
        "bias_T": {"inputs": ["ELEVATION_DIFFERENCE"], "layers": [4]},
        "sw_contrib": {"inputs": ["slope"], "layers": [4], **(sw_contrib or {})},
        "cor_acc": {"inputs": ["t2m"], "layers": [4]},
        "cor_abl": {"inputs": ["t2m"], "layers": [4]},
    }
    if sfc_type is not None:
        params["sfc_type"] = sfc_type
    return params


def _model(sw_contrib=None, sfc_type=None, inputs=None):
    params = _params(sw_contrib, sfc_type)
    if inputs is not None:
        params["inputs"] = inputs
    torch.manual_seed(0)
    cls = TILikeModel if sfc_type is None else TILikeSfcModel
    return cls(params, BOUNDS)


def _one_sequence(n):
    return {
        "perm": torch.arange(n),
        "start": torch.zeros(n, dtype=torch.int64),
        "spinup": None,
    }


def _fraction(model, x):
    """The output of the shortwave network, max * sigmoid."""
    return model.R_sw_max * F.sigmoid(
        model.R_sw_contrib(x[:, model.ind_inp_sw_contrib]).view(-1)
    )


def test_default_contribution_is_unchanged():
    model = _model({"max": 0.5})
    x = torch.rand((10, len(INPUTS)))
    T = torch.randn(10)
    with torch.no_grad():
        assert torch.allclose(model._shortwave(x, T), _fraction(model, x))
    assert "sw_gate_threshold" not in model.scalar_parameters()


def test_contribution_scaled_by_the_downward_shortwave_radiation():
    model = _model({"max": 0.2, "scale_by_ssrd": True})
    x = torch.rand((10, len(INPUTS)))
    ssrd = Normalizer._unorm(x[:, INPUTS.index("ssrd")], *BOUNDS["ssrd"])
    with torch.no_grad():
        R_sw = model._shortwave(x, torch.zeros(10))
    assert torch.allclose(R_sw, _fraction(model, x) * ssrd * SSRD_TO_MELT)
    # 2e7 J m-2 per day (a sunny summer month) could melt about 1.8 m w.e.
    assert 2e7 * SSRD_TO_MELT == pytest.approx(1.82, abs=0.01)
    with pytest.raises(AssertionError, match="ssrd in the model inputs"):
        _model(
            {"scale_by_ssrd": True},
            inputs=["ELEVATION_DIFFERENCE", "t2m", "tp_sum", "slope"],
        )


def test_temperature_gate_with_its_own_threshold():
    gate = {"threshold": {"value": -2.0}, "width": 0.5}
    model = _model({"gate": gate})
    x = torch.rand((3, len(INPUTS)))
    T = torch.tensor([-10.0, -2.0, 5.0])
    with torch.no_grad():
        ratio = model._shortwave(x, T) / _fraction(model, x)
    assert ratio[0] < 1e-6
    assert ratio[1] == pytest.approx(0.5)
    assert ratio[2] == pytest.approx(1.0, abs=1e-6)
    assert model.scalar_parameters()["sw_gate_threshold"] == pytest.approx(-2.0)
    # A fixed threshold is not in the checkpoints
    assert not any("sw_gate" in k for k in model.state_dict())
    with pytest.raises(AssertionError, match="width must be positive"):
        _model({"gate": {"width": 0.0}})
    with pytest.raises(AssertionError, match="Unknown sw_contrib.gate"):
        _model({"gate": {"slope": 1.0}})


def test_trainable_gate_threshold_within_its_bounds():
    gate = {"threshold": {"value": 0.0, "trainable": True, "min": -5.0, "max": 1.0}}
    model = _model({"gate": gate})
    assert model.sw_gate_threshold().item() == pytest.approx(0.0, abs=1e-6)
    x = torch.rand((12, len(INPUTS)))
    model(x).sum().backward()
    assert model.sw_gate_threshold_raw.grad is not None
    assert "sw_gate_threshold_raw" in model.state_dict()
    with torch.no_grad():
        model.sw_gate_threshold_raw.fill_(-50.0)
    assert model.sw_gate_threshold().item() == pytest.approx(-5.0)


def test_surface_type_on_the_shortwave_contribution_only():
    model = _model(sfc_type={"ice_ratio": {"value": 1.0}, "sw_ratio": {"value": 2.5}})
    assert model.sw_ratio().item() == pytest.approx(2.5)
    assert not any("sw_ratio" in k for k in model.state_dict())
    PDD = torch.full((4,), 50.0)
    melt = model.beta2 * 50.0
    R_sw = torch.full((4,), 0.1)
    ones = torch.ones(4)
    seq = _one_sequence(4)
    with torch.no_grad():
        # No snow: the surface is ice from the start (swe < 0)
        bare = model._mass_balance_sfc(torch.zeros(4), PDD, ones, ones, R_sw, seq)
        # Deep snow all along
        snowy = torch.tensor([20.0 / model.beta1, 0.0, 0.0, 0.0])
        snow = model._mass_balance_sfc(snowy, PDD, ones, ones, R_sw, seq)
        _, f_snow = model._snow_cover(model.beta1 * snowy, melt * ones, R_sw, seq)
    # The temperature melt does not depend on the surface, the shortwave does
    assert (-bare[1:]).tolist() == pytest.approx([melt + 2.5 * 0.1] * 3, rel=1e-2)
    assert (-snow[1:]).tolist() == pytest.approx([melt + 0.1] * 3, rel=1e-2)
    expected = model.beta1 * snowy - melt - R_sw * (f_snow + (1 - f_snow) * 2.5)
    assert torch.allclose(snow, expected)


def test_snow_cover_gives_the_effective_shortwave_contribution():
    model = _model(sfc_type={"ice_ratio": {"value": 1.0}, "sw_ratio": {"value": 2.0}})
    x = torch.rand((6, len(INPUTS)))
    with torch.no_grad():
        swe, f_snow, cor_abl_eff, R_sw_eff = model.snow_cover(x, _one_sequence(6))
        _, _, _, cor_abl, R_sw = model._terms(x)
    assert torch.allclose(cor_abl_eff, cor_abl)
    assert torch.allclose(R_sw_eff, R_sw * (f_snow + (1 - f_snow) * 2.0))


def test_sw_ratio_defaults_to_one_and_can_be_learned():
    model = _model(sfc_type={})
    assert model.sw_ratio().item() == pytest.approx(1.0)
    params = model.scalar_parameters()
    assert params["sw_ratio"] == pytest.approx(1.0) and "sw_ratio_raw" not in params
    learned = _model(
        sfc_type={"sw_ratio": {"value": 2.5, "trainable": True, "min": 1.5, "max": 3.5}}
    )
    assert learned.sw_ratio().item() == pytest.approx(2.5)
    x = torch.rand((12, len(INPUTS)))
    learned(x, _one_sequence(12)).sum().backward()
    assert learned.sw_ratio_raw.grad is not None
    assert "sw_ratio_raw" in learned.scalar_parameters()
    with pytest.raises(AssertionError, match="min < value < max"):
        _model(sfc_type={"sw_ratio": {"value": 5.0, "trainable": True}})
