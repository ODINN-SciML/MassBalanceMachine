import pandas as pd
import pytest
import pyproj
import xarray as xr
from types import SimpleNamespace
from pathlib import Path

import massbalancemachine.aws.features as aws_features
import massbalancemachine as mbm


@pytest.mark.integration
def test_build_features_lom0154_retrieves_svf():
    data_dir = Path(__file__).parent / "data"

    cfg = mbm.Config()
    result = aws_features.build_features("LOM0154", cfg, data_dir=data_dir)

    assert result["svf"].notna().all()
    assert result["svf"].between(0, 1).all()


def test_create_aws_grid_adds_monthly_grid_fields(monkeypatch):
    monthly = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2007-01-01", "2007-02-01"]),
            "P": [3.0, 6.0],
            "POINT_LAT": [46.0, 46.0],
            "POINT_LON": [13.0, 13.0],
            "POINT_ELEVATION": [1000, 1000],
            "RGIId": ["RGI_TEST", "RGI_TEST"],
            "aspect": [10.0, 10.0],
            "slope": [20.0, 20.0],
        }
    )
    monkeypatch.setattr(aws_features, "build_features", lambda aws_code, cfg: monthly)

    cfg = mbm.Config()
    result = aws_features.create_aws_grid("TEST0001", cfg)

    assert result["POINT_ID"].tolist() == [1, 1]
    assert result["N_MONTHS"].tolist() == [1, 1]
    assert result["YEAR"].tolist() == [2007, 2007]
    # assert result["MONTHS"].tolist() == [1, 2]
    assert result["FROM_DATE"].tolist() == ["20070101", "20070201"]
    assert result["TO_DATE"].tolist() == ["20070131", "20070228"]
    assert result["PERIOD"].tolist() == ["annual", "annual"]


def test_interpolate_aws_svf_returns_nan_outside_grid():
    monthly = pd.DataFrame({"POINT_LON": [2.0], "POINT_LAT": [0.5]})
    ds = xr.Dataset(coords={"x": [0.0, 1.0], "y": [0.0, 1.0]})
    svf = xr.Dataset(
        {"svf": (("y", "x"), [[0.2, 0.4], [0.6, 0.8]])},
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    )
    gdir = SimpleNamespace(grid=SimpleNamespace(proj=pyproj.Proj("epsg:4326")))

    result = aws_features._interpolate_aws_svf(monthly, ds, gdir, svf)

    assert result["svf"].isna().all()


def test_interpolate_topography_uses_netcdf_domain_not_glacier_outline():
    monthly = pd.DataFrame({"POINT_LON": [0.5], "POINT_LAT": [0.5]})
    ds = xr.Dataset(
        {
            "aspect": (("y", "x"), [[10.0, 20.0], [30.0, 40.0]]),
            "slope": (("y", "x"), [[5.0, 6.0], [7.0, 8.0]]),
        },
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    )
    glacier_indices = (pd.array([0, 0, 1, 1]), pd.array([0, 1, 0, 1]))
    gdir = SimpleNamespace(grid=SimpleNamespace(proj=pyproj.Proj("epsg:4326")))

    result = aws_features._interpolate_glacier_topography(
        monthly, ds, glacier_indices, gdir
    )

    assert result[["aspect", "slope"]].notna().all().all()


if __name__ == "__main__":
    test_build_features_lom0154_retrieves_svf()
    with pytest.MonkeyPatch.context() as monkeypatch:
        test_create_aws_grid_adds_monthly_grid_fields(monkeypatch)
    test_interpolate_aws_svf_returns_nan_outside_grid()
    test_interpolate_topography_uses_netcdf_domain_not_glacier_outline()
