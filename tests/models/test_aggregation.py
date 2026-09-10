"""Tests of the aggregations turning point-wise predictions into per-stake or
glacier-wide values."""

import numpy as np
import torch

from models.TorchNeuralNetworkRegressor import aggrPredict, aggrPredictGlwd

PRED = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0])
ID_AGGR = np.array([0, 0, 0, 1, 1])


def test_sum():
    out = torch.zeros(2)
    torch.testing.assert_close(
        aggrPredict(PRED, ID_AGGR, out=out), torch.tensor([6.0, 30.0])
    )


def test_mean_does_not_count_the_initial_value():
    # With the initial zeros counted as samples these would be 6 / 4 and 30 / 3
    expected = torch.tensor([2.0, 15.0])
    torch.testing.assert_close(
        aggrPredict(PRED, ID_AGGR, reduce="mean", out=torch.zeros(2)), expected
    )
    torch.testing.assert_close(
        aggrPredictGlwd(PRED, ID_AGGR, out=torch.zeros(2)), expected
    )
    torch.testing.assert_close(
        aggrPredictGlwd(PRED, torch.from_numpy(ID_AGGR)), expected
    )


def test_mean_gradient():
    pred = PRED.clone().requires_grad_()
    aggrPredictGlwd(pred, ID_AGGR, out=torch.zeros(2)).sum().backward()
    torch.testing.assert_close(pred.grad, torch.tensor([1 / 3, 1 / 3, 1 / 3, 0.5, 0.5]))
