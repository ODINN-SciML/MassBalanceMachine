"""The NVE windows before 1985 are read from the grids on the GAO outlines, the later
ones from those on RGI 6.2, even when two windows meet inside one year. Synthetic
grids in a temporary data tree: no data needed."""

import os

import numpy as np
import pandas as pd
import pytest

import data_processing.product_utils as product_utils
from data_processing.gridded_utils import (
    MONTH_TO_ID,
    _location_trees,
    geodetic_input_NVE,
    grid_tree,
)
from data_processing.product_utils import glacier_id_to_folders

GLACIER = "NVE-2636"
# Points of the glacier on each outline
N_POINTS = {"NVE_GAO": 2, "NVE": 3}


def _write_grids(root, tree, years):
    folder = os.path.join(root, "grids", tree, *glacier_id_to_folders(GLACIER))
    os.makedirs(folder, exist_ok=True)
    n = N_POINTS[tree]
    for year in years:
        pd.DataFrame(
            {
                "ID": np.tile(np.arange(n), 12),
                "MONTHS": np.repeat(list(MONTH_TO_ID), n),
                "YEAR": year,
                "TREE": tree,
                "aspect": 0.0,
                "slope": 0.0,
            }
        ).to_parquet(os.path.join(folder, f"{year}.parquet"))


@pytest.fixture
def data_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(product_utils, "_data_path", str(tmp_path))
    _write_grids(tmp_path, "NVE_GAO", [1984, 1985])
    _write_grids(tmp_path, "NVE", [1985, 1986])
    return tmp_path


def test_trees_of_the_nve_windows():
    assert grid_tree("NVE", 1984) == "NVE_GAO"
    assert grid_tree("NVE", 1985) == "NVE"
    assert grid_tree("Belart20", 1960) == "Belart20"
    assert _location_trees("NVE") == ["NVE", "NVE_GAO"]


def test_windows_meeting_inside_a_year_read_both_trees(data_tree):
    gao = (pd.Timestamp("1984-10-01"), pd.Timestamp("1985-10-01"))
    rgi = (pd.Timestamp("1985-10-01"), pd.Timestamp("1986-10-01"))
    df = geodetic_input_NVE(GLACIER, [gao, rgi])
    per_month = df.groupby(["YEAR", "MONTHS"]).TREE.agg(["unique", "size"])
    month_index = (
        per_month.index.get_level_values(0) * 12
        + per_month.index.get_level_values(1).map(MONTH_TO_ID)
        - 1
    )
    in_gao = month_index < 1985 * 12 + 9
    assert len(per_month) == 24
    assert all(list(u) == ["NVE_GAO"] for u in per_month["unique"][in_gao])
    assert all(list(u) == ["NVE"] for u in per_month["unique"][~in_gao])
    assert (per_month["size"][in_gao] == N_POINTS["NVE_GAO"]).all()
    assert (per_month["size"][~in_gao] == N_POINTS["NVE"]).all()
    # One glacier-month per month, and one ID per point and year of one tree
    assert df.GLWD_M_ID.nunique() == 24
    assert df.ID.nunique() == 2 + (2 + 3) + 3


def test_windows_on_different_outlines_cannot_share_months(data_tree):
    gao = (pd.Timestamp("1984-10-01"), pd.Timestamp("1985-10-01"))
    rgi = (pd.Timestamp("1985-07-01"), pd.Timestamp("1986-07-01"))
    with pytest.raises(AssertionError):
        geodetic_input_NVE(GLACIER, [gao, rgi])
