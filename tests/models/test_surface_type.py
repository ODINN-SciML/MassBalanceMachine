"""Surface type (snow or ice) of the ablation of the TIlike model (`model.sfc_type`)
and the time order of its inputs (`dataloader.surface_type`)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from data_processing.get_climate_data import _create_date_range, _create_month_range
from data_processing.gswp3_w5e5 import calendar_months as gswp3_calendar_months
from data_processing.transform_to_monthly import SPINUP_COLUMN, transform_to_monthly
from data_processing.utils.hydro_year import (
    build_head_tail_pads_from_monthly_df,
    spinup_tokens,
    split_padded_tokens,
)
from dataloader.surface_type import (
    GRID_PADDED_MONTHS,
    calendar_months,
    grid_sequences,
    hydro_year_sequences,
    stake_sequences,
)
from models.TorchNeuralNetworkRegressor import (
    TILikeModel,
    TILikeSfcModel,
    createModel,
    forward_rows,
    needs_time_order,
)

INPUTS = ["ELEVATION_DIFFERENCE", "t2m", "tp_sum"]
BOUNDS = {
    "ELEVATION_DIFFERENCE": (0, 1000),
    "t2m": (-20, 15),
    "tp_sum": (0, 1),
}
HYDRO_MONTHS = ["oct", "nov", "dec", "jan", "feb", "mar", "apr"]
SUMMER_MONTHS = ["may", "jun", "jul", "aug", "sep"]


def test_calendar_months_of_stakes_and_grids():
    years = [2000] * 6
    cal, num = calendar_months(years, ["sep_", "oct", "dec", "jan", "sep", "oct_"])
    assert cal.tolist() == [1999, 1999, 1999, 2000, 2000, 2000]
    assert num.tolist() == [9, 10, 12, 1, 9, 10]
    # Grids are in calendar years
    cal, _ = calendar_months([2000] * 3, ["jan", "sep", "dec_"], GRID_PADDED_MONTHS)
    assert cal.tolist() == [2000, 2000, 2000]


def test_padded_months_are_told_apart_by_their_run():
    # 'jan_' after the hydrological year (Iceland), 'mar_' before it (RGI 13)
    head, tail = split_padded_tokens(["sep_", "oct_", "nov_", "dec_", "jan_", "oct"])
    assert head == ["oct_", "nov_", "dec_", "jan_"] and tail == ["sep_"]
    tail_months = ["mar_", "apr_", "may_", "jun_", "jul_", "aug_", "sep_"]
    head, tail = split_padded_tokens(tail_months + ["oct_"])
    assert head == ["oct_"] and tail == tail_months
    with pytest.raises(ValueError, match="cannot be told apart"):
        split_padded_tokens([m + "_" for m in HYDRO_MONTHS + SUMMER_MONTHS])
    # Both month mappers agree, also on a subset of the table with gaps
    months = ["mar_", "sep_", "oct", "jul", "oct_", "jan_", "_oct"]
    cal, num = calendar_months([2015] * len(months), months, ["oct_", "jan_"])
    assert cal.tolist() == [2014, 2014, 2014, 2015, 2015, 2016, 2013]
    assert num.tolist() == [3, 9, 10, 7, 10, 1, 10]
    df = pd.DataFrame({"YEAR": 2015, "MONTHS": ["mar_", "sep_", "oct", "jul"]})
    cal_g, num_g = gswp3_calendar_months(df, grid=False)
    assert cal_g.tolist() == cal[:4].tolist() and num_g.tolist() == num[:4].tolist()
    # The sequence of the hydrological year 2016 starts at 'oct_' and holds 'jan_'
    head = ["oct_", "nov_", "dec_", "jan_"]
    seq = hydro_year_sequences([0] * 3, [2015] * 3, ["oct_", "jan_", "sep"], head)
    perm = seq["perm"].tolist()
    # In time order: sep (2015), then oct_ (2015) and jan_ (2016) in one sequence
    assert perm == [2, 0, 1] and seq["start"].tolist() == [0, 1, 1]


def test_sequences_of_an_empty_table():
    sfc = stake_sequences(pd.DataFrame({"ID": [], "YEAR": [], "MONTHS": []}))
    assert len(sfc["perm"]) == 0 and len(sfc["start"]) == 0


def test_sequences_split_points_and_hydrological_years():
    # Two points over a calendar year of a grid, shuffled
    months = ["jan", "sep", "oct_", "dec_"]
    df = pd.DataFrame(
        {
            "POINT_LAT": [1.0] * 4 + [2.0] * 4,
            "POINT_LON": [0.0] * 8,
            "YEAR": [2000] * 8,
            "MONTHS": months * 2,
        }
    ).sample(frac=1, random_state=0)
    sfc = grid_sequences(df)
    ordered = df.iloc[sfc["perm"].numpy()]
    start = sfc["start"].numpy()
    # Within a point, jan-sep then oct_-dec_ (the next hydrological year)
    assert ordered.MONTHS.tolist() == months * 2
    assert start.tolist() == [0, 0, 2, 2, 4, 4, 6, 6]
    assert sfc["spinup"] is None


def test_sequences_reject_repeated_months():
    with pytest.raises(AssertionError, match="repeat the month"):
        hydro_year_sequences([0, 0], [2000, 2000], ["jan", "jan"])


TAIL_PAD = ["aug_", "sep_"]
SPINUP_TOKENS = spinup_tokens(TAIL_PAD)
MONTH_LIST = SPINUP_TOKENS + TAIL_PAD + HYDRO_MONTHS + SUMMER_MONTHS


def _wide_stakes():
    """A winter, a summer and an annual measurement (from August, padded before the
    hydrological year) at three locations, as the table of the climate features: one
    column per variable and month token, here of the spin-up too."""
    rows = []
    for period, start, end, lat in [
        ("winter", "20001001", "20010430", 1.0),
        ("summer", "20010501", "20010930", 5.0),
        ("annual", "20000801", "20010930", 7.0),
    ]:
        row = {
            "FROM_DATE": start,
            "TO_DATE": end,
            "YEAR": 2001,
            "RGIId": "G",
            "PERIOD": period,
            "POINT_LAT": lat,
            "POINT_LON": 0.0,
            "POINT_BALANCE": {"winter": 1.5, "summer": -2.0, "annual": -1.0}[period],
            "POINT_ELEVATION": 3000.0,
            "ALTITUDE_CLIMATE": 2500.0,
            "ELEVATION_DIFFERENCE": 500.0,
            "aspect": lat,
        }
        # The climate of each location differs
        for i, m in enumerate(MONTH_LIST):
            row[f"t2m_{m}"] = 100 * lat + i
        rows.append(row)
    return pd.DataFrame(rows)


def _monthly(spinup, df=None):
    return transform_to_monthly(
        _wide_stakes() if df is None else df,
        ["RGIId", "ID", "N_MONTHS", "MONTHS", "PERIOD"],
        ["t2m"],
        ["aspect"],
        None,
        "csv",
        spinup_month_list=MONTH_LIST if spinup else None,
    )


def test_calendar_months_of_spinup_tokens():
    # '_m' is month m of the hydrological year YEAR - 1
    assert SPINUP_TOKENS == ["_" + m for m in HYDRO_MONTHS + ["may", "jun", "jul"]]
    cal, num = calendar_months([2001] * 4, ["_oct", "_dec", "_jul", "aug_"])
    assert cal.tolist() == [1999, 1999, 2000, 2000]
    assert num.tolist() == [10, 12, 7, 8]
    df = pd.DataFrame({"YEAR": 2001, "MONTHS": ["_oct", "_jul", "aug_", "oct"]})
    df = pd.concat([df, pd.DataFrame({"YEAR": 2001, "MONTHS": MONTH_LIST})])
    cal, num = gswp3_calendar_months(df, grid=False)
    assert cal[:4].tolist() == [1999, 2000, 2000, 2000]
    assert num[:4].tolist() == [10, 7, 8, 10]
    # The pads of the table ignore the spin-up tokens
    head, tail = build_head_tail_pads_from_monthly_df(df)
    assert head == [] and tail == TAIL_PAD
    # The climate is read from the 1st of October of the year before
    start, end, back = _create_month_range(SPINUP_TOKENS + TAIL_PAD, [])
    dates = _create_date_range(2001, start, end, back)
    assert dates[0] == pd.Timestamp("1999-10-01") and len(dates) == len(MONTH_LIST)


def test_spinup_rows_are_the_months_since_october_at_the_stake():
    plain = _monthly(spinup=False)
    df = _monthly(spinup=True)
    spin = df[df[SPINUP_COLUMN]]
    summer = spin[spin.PERIOD == "summer"]
    annual = spin[spin.PERIOD == "annual"]
    # The winter measurement starts on the 1st of October: no spin-up
    assert set(spin.PERIOD) == {"summer", "annual"}
    # The summer measurement gets October to April
    assert summer.MONTHS.tolist() == HYDRO_MONTHS
    # The annual one, from August 2000, gets October 1999 to July 2000
    assert annual.MONTHS.tolist() == SPINUP_TOKENS
    # At its own location: its climate and topography, not those of another stake
    first_core = MONTH_LIST.index("oct")
    assert summer.t2m.tolist() == [500.0 + first_core + i for i in range(7)]
    assert annual.t2m.tolist() == [700.0 + i for i in range(len(SPINUP_TOKENS))]
    assert (summer.aspect == 5.0).all() and (summer.POINT_LAT == 5.0).all()
    # Its metadata, N_MONTHS counting its own months only
    assert (summer.POINT_BALANCE == -2.0).all()
    assert (summer.N_MONTHS == len(SUMMER_MONTHS)).all()
    assert (annual.N_MONTHS == 14).all()
    # The other rows are those of the table without spin-up, and come first
    pd.testing.assert_frame_equal(
        df.iloc[: len(plain)][plain.columns],
        plain,
        check_dtype=False,
    )
    # Every sequence starts on a 1st of October: the annual measurement has two,
    # October 1999 - September 2000 and October 2000 - September 2001
    sfc = stake_sequences(df)
    ordered = df.iloc[sfc["perm"].numpy()].reset_index(drop=True)
    first = ordered.iloc[sfc["start"].numpy()].reset_index(drop=True)
    assert set(first.MONTHS) == {"oct", "_oct"}
    is_annual = ordered.PERIOD == "annual"
    assert (first.MONTHS[is_annual & ordered.MONTHS.isin(TAIL_PAD)] == "_oct").all()
    assert (first.MONTHS[is_annual & ordered.MONTHS.isin(HYDRO_MONTHS)] == "oct").all()


def test_spinup_before_the_climate_data_is_dropped():
    wide = _wide_stakes()
    wide.loc[wide.PERIOD == "annual", "t2m__oct"] = np.nan
    df = _monthly(spinup=True, df=wide)
    spin = df[df[SPINUP_COLUMN]]
    assert set(spin.PERIOD) == {"summer"}
    assert len(df[~df[SPINUP_COLUMN]]) == len(_monthly(spinup=False))


def _params():
    return {
        "inputs": INPUTS,
        "grad_T": {"inputs": ["t2m"], "layers": [4]},
        "bias_T": {"inputs": ["ELEVATION_DIFFERENCE"], "layers": [4]},
        "sw_contrib": {"inputs": ["ELEVATION_DIFFERENCE"], "layers": [4]},
        "cor_acc": {"inputs": ["t2m"], "layers": [4]},
        "cor_abl": {"inputs": ["t2m"], "layers": [4]},
    }


def _model(sfc_type=None):
    params = _params()
    if sfc_type is not None:
        params["sfc_type"] = sfc_type
    torch.manual_seed(0)
    return (TILikeModel if sfc_type is None else TILikeSfcModel)(params, BOUNDS)


def _one_sequence(n):
    return {
        "perm": torch.arange(n),
        "start": torch.zeros(n, dtype=torch.int64),
        "spinup": None,
    }


def test_the_model_knows_whether_it_needs_the_time_order():
    plain, sfc = _model(), _model({})
    assert not needs_time_order(plain) and needs_time_order(sfc)
    assert not any(k.startswith("ice_ratio") for k in plain.state_dict())
    # createModel picks the class from the parameters
    cfg = SimpleNamespace(bnds=BOUNDS, featureColumns=INPUTS)
    params = {"type": "TIlike", **_params()}
    assert type(createModel(cfg, params)) is TILikeModel
    assert type(createModel(cfg, {**params, "sfc_type": None})) is TILikeSfcModel
    with pytest.raises(AssertionError, match="TILikeSfcModel"):
        TILikeModel({**_params(), "sfc_type": {}}, BOUNDS)
    # forward_rows gives the time order to the models that take it only
    x = torch.rand((12, len(INPUTS)))
    with torch.no_grad():
        assert torch.equal(forward_rows(plain, x, _one_sequence(12)), plain(x))
        assert torch.equal(
            forward_rows(sfc, x, _one_sequence(12)), sfc(x, _one_sequence(12))
        )


def test_surface_type_with_equal_melt_factors_is_the_plain_model():
    plain, sfc = _model(), _model({"ice_ratio": {"value": 1.0}})
    x = torch.rand((12, len(INPUTS)))
    with torch.no_grad():
        assert torch.allclose(plain(x), sfc(x, _one_sequence(12)))


def test_sfc_type_needs_the_time_order():
    model = _model({})
    x = torch.rand((3, len(INPUTS)))
    with pytest.raises(TypeError):
        model(x)
    with pytest.raises(ValueError, match="time order"):
        forward_rows(model, x, None)


def test_snow_cover_is_the_cumulative_balance_of_the_sequence():
    model = _model({"snow_scale": 0.2})
    n = 6
    acc = torch.tensor([0.5, 0.3, 0.0, 0.0, 0.0, 0.0])
    melt = torch.tensor([0.0, 0.0, 0.4, 0.6, 0.8, 0.8])
    # Two sequences of three months, given in reverse order
    perm = torch.tensor([5, 4, 3, 2, 1, 0])
    start = torch.tensor([0, 0, 0, 3, 3, 3])
    swe, f_snow = model._snow_cover(
        acc, melt, 0.0, {"perm": perm, "start": start, "spinup": None}
    )
    # Sequence 1 is rows 5, 4, 3 and sequence 2 rows 2, 1, 0, in time order
    delta = acc - melt
    expected = torch.empty(n)
    for rows in ([5, 4, 3], [2, 1, 0]):
        before = 0.0
        for r in rows:
            expected[r] = before + 0.5 * delta[r]
            before += delta[r]
    assert torch.allclose(swe, expected)
    assert torch.allclose(f_snow, torch.sigmoid(expected / 0.2))


def test_ice_melts_faster_and_spinup_rows_are_zero():
    model = _model({"ice_ratio": {"value": 2.0}})
    assert model.ice_ratio().item() == pytest.approx(2.0)
    assert model.scalar_parameters()["ice_ratio"] == pytest.approx(2.0)
    # 2 m w.e. of snow in the first month, then 0.49 m w.e. of melt per month at the
    # snow melt factor
    P_solid = torch.tensor([2.0 / model.beta1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    PDD = torch.tensor([0.0] + [100.0] * 6)
    cor_acc = torch.ones(7)
    cor_abl = torch.ones(7)
    seq = _one_sequence(7)
    with torch.no_grad():
        MB = model._mass_balance_sfc(P_solid, PDD, cor_acc, cor_abl, 0.0, seq)
    melt_snow = model.beta2 * 100.0
    assert melt_snow == pytest.approx(0.49)
    # Snow while the 2 m w.e. last (months 1-3), a transition, then ice
    assert -MB[1:4] == pytest.approx([melt_snow] * 3, rel=1e-2)
    assert melt_snow < -MB[4] < -MB[5] < 2.0 * melt_snow
    assert -MB[6] == pytest.approx(2.0 * melt_snow, rel=1e-2)
    # The spin-up rows build the snow cover without counting in the balance
    seq["spinup"] = torch.tensor([True, True] + [False] * 5)
    with torch.no_grad():
        MB_spin = model._mass_balance_sfc(P_solid, PDD, cor_acc, cor_abl, 0.0, seq)
    assert (MB_spin[:2] == 0).all() and torch.equal(MB_spin[2:], MB[2:])


def test_ice_ratio_is_fixed_by_default():
    model = _model({})
    assert model.ice_ratio().item() == pytest.approx(1 / 0.7)
    # Neither optimized nor in the checkpoints: always the value of the params
    assert "ice_ratio_value" not in model.state_dict()
    other = _model({"ice_ratio": {"value": 2.0}})
    other.load_state_dict(model.state_dict())
    assert other.ice_ratio().item() == pytest.approx(2.0)
    assert not any("ice_ratio" in name for name, _ in model.named_parameters())
    assert "ice_ratio_raw" not in model.scalar_parameters()
    with pytest.raises(AssertionError, match="only apply with trainable"):
        _model({"ice_ratio": {"value": 2.0, "min": 1.0}})


def test_trainable_ice_ratio_gets_gradients_within_its_bounds():
    model = _model(
        {"ice_ratio": {"value": 1.5, "trainable": True, "min": 1.0, "max": 2.5}}
    )
    assert model.ice_ratio().item() == pytest.approx(1.5)
    x = torch.rand((12, len(INPUTS)))
    model(x, _one_sequence(12)).sum().backward()
    assert model.ice_ratio_raw.grad is not None
    with torch.no_grad():
        model.ice_ratio_raw.fill_(50.0)
    assert model.ice_ratio().item() == pytest.approx(2.5)
