"""Spin-up months of the surface type in the geodetic grids (`spinup=True`): the months
between the 1st of October preceding every window and its start. Synthetic grids in a
temporary data tree: no data needed."""

import os

import numpy as np
import pandas as pd
import pytest

import data_processing.product_utils as product_utils
import data_processing.gridded_utils as gridded_utils
from data_processing.gridded_utils import (
    MONTH_TO_ID,
    SPINUP_COLUMN,
    _spinup_lower_bound,
    geodetic_input_Hugonnet21,
    geodetic_input_NVE,
    geodetic_window_weights,
    spinup_years,
    spinup_years_calendar,
)
from data_processing.product_utils import glacier_id_to_folders
from dataloader.surface_type import grid_sequences


@pytest.fixture(autouse=True)
def _no_pcsr(monkeypatch):
    """The synthetic grids have no DEM to compute the potential clear-sky radiation
    from: its value is irrelevant here."""
    monkeypatch.setattr(
        gridded_utils, "pcsr_for_rows", lambda df, folder: np.zeros(len(df), np.float32)
    )


NVE_ID = "NVE-2636"
RGI_ID = "RGI60-11.01238"
N_POINTS = {"NVE_GAO": 2, "NVE": 3, "Hugonnet21": 2}


def _write_grids(root, glacier, tree, years):
    folder = os.path.join(root, "grids", tree, *glacier_id_to_folders(glacier))
    os.makedirs(folder, exist_ok=True)
    n = N_POINTS[tree]
    for year in years:
        pd.DataFrame(
            {
                "ID": np.tile(np.arange(n), 12),
                "MONTHS": np.repeat(list(MONTH_TO_ID), n),
                "YEAR": year,
                # The points of the two outlines share their coordinates
                "POINT_LAT": np.tile(np.arange(n, dtype=float), 12),
                "POINT_LON": 0.0,
                "aspect": 0.0,
                "slope": 0.0,
            }
        ).to_parquet(os.path.join(folder, f"{year}.parquet"))


@pytest.fixture
def data_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(product_utils, "_data_path", str(tmp_path))
    _write_grids(tmp_path, NVE_ID, "NVE_GAO", [1984, 1985])
    _write_grids(tmp_path, NVE_ID, "NVE", [1985, 1986])
    _write_grids(tmp_path, RGI_ID, "Hugonnet21", [1999, 2000, 2001])
    return tmp_path


def _with_ints(df):
    return df.assign(GLWD_M_ID_int=pd.factorize(df.GLWD_M_ID)[0])


def test_spinup_bounds():
    month = lambda y, m: y * 12 + m - 1
    assert _spinup_lower_bound(month(2000, 1)) == month(1999, 10)
    assert _spinup_lower_bound(month(2000, 9)) == month(1999, 10)
    assert _spinup_lower_bound(month(2000, 10)) == month(2000, 10)
    assert _spinup_lower_bound(month(2000, 12)) == month(2000, 10)
    jan, oct_ = pd.Timestamp("2000-01-01"), pd.Timestamp("1999-10-01")
    assert spinup_years([(jan, pd.Timestamp("2010-01-01"))]) == [1999]
    assert spinup_years([(oct_, pd.Timestamp("2010-01-01"))]) == []
    assert spinup_years([(2000, 2020)]) == [1999]
    assert spinup_years_calendar(range(2000, 2020)) == [1999]
    assert spinup_years_calendar([1980, 1981, 2000]) == [1979, 1999]


def test_hugonnet_grid_gets_october_to_december_before(data_tree):
    plain = geodetic_input_Hugonnet21(RGI_ID, years=[2000, 2001])
    df = geodetic_input_Hugonnet21(RGI_ID, years=[2000, 2001], spinup=True)
    spin = df[df[SPINUP_COLUMN]]
    assert set(spin.YEAR) == {1999}
    assert sorted(spin.MONTHS.unique()) == ["dec_", "nov_", "oct_"]
    assert len(df) == len(plain) + 3 * N_POINTS["Hugonnet21"]
    assert SPINUP_COLUMN not in plain.columns
    # The spin-up months are in no window, the others are weighted as before
    w = geodetic_window_weights(_with_ints(df), [(2000, 2002)])
    w_plain = geodetic_window_weights(_with_ints(plain), [(2000, 2002)])
    assert w.sum() == pytest.approx(w_plain.sum())
    # Every point chains October 1999 to September 2000
    sfc = grid_sequences(df)
    ordered = df.iloc[sfc["perm"].numpy()].reset_index(drop=True)
    first = ordered.iloc[sfc["start"].numpy()].reset_index(drop=True)
    jan = (ordered.YEAR == 2000) & (ordered.MONTHS == "jan")
    assert ((first.YEAR[jan] == 1999) & (first.MONTHS[jan] == "oct_")).all()
    assert sfc["spinup"].sum() == len(spin)


def test_spinup_of_a_window_on_other_outlines(data_tree):
    # The GAO window runs to the end of 1985, the RGI one starts in 1986: its spin-up,
    # October to December 1985, is read from the RGI tree although these months are
    # in the GAO window
    gao = (pd.Timestamp("1984-10-01"), pd.Timestamp("1986-01-01"))
    rgi = (pd.Timestamp("1986-01-01"), pd.Timestamp("1986-10-01"))
    plain = geodetic_input_NVE(NVE_ID, [gao, rgi])
    df = geodetic_input_NVE(NVE_ID, [gao, rgi], spinup=True)
    spin = df[df[SPINUP_COLUMN]]
    assert set(spin.GRID_TREE) == {"NVE"} and set(spin.YEAR) == {1985}
    assert sorted(spin.MONTHS.unique()) == ["dec_", "nov_", "oct_"]
    assert len(spin) == 3 * N_POINTS["NVE"]
    assert len(df) == len(plain) + len(spin)
    # Their glacier-wide months are their own, and in no window
    assert not set(spin.GLWD_M_ID) & set(df.GLWD_M_ID[~df[SPINUP_COLUMN]])
    w = geodetic_window_weights(_with_ints(df), [gao, rgi])
    w_plain = geodetic_window_weights(_with_ints(plain), [gao, rgi])
    np.testing.assert_allclose(w.sum(axis=1), w_plain.sum(axis=1))
    # The points of the two outlines are not chained, although they share coordinates
    sfc = grid_sequences(df)
    ordered = df.iloc[sfc["perm"].numpy()].reset_index(drop=True)
    first = ordered.iloc[sfc["start"].numpy()].reset_index(drop=True)
    assert (ordered.GRID_TREE == first.GRID_TREE).all()
    jan86 = (ordered.YEAR == 1986) & (ordered.MONTHS == "jan")
    assert ((first.YEAR[jan86] == 1985) & (first.MONTHS[jan86] == "oct_")).all()
