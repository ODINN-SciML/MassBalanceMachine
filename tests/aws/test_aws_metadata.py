import pandas as pd
import pytest
import geopandas as gpd
import pyproj
import xarray as xr
from pathlib import Path
from tempfile import TemporaryDirectory
from shapely.geometry import box
from types import SimpleNamespace

from massbalancemachine.aws.metadata import (
    check_aws_glacier_proximity,
    parse_aws_metadata,
)
from massbalancemachine.aws.data import (
    load_aws_data,
    load_aws_monthly_precipitation,
)
from massbalancemachine.aws.features import _interpolate_glacier_topography


LOCAL_AWS_DATA = Path(__file__).parent / "data"


def test_load_aws_data_from_local_sample():
    metadata = parse_aws_metadata(LOCAL_AWS_DATA / "Metadata")
    data = load_aws_data("FVG0001", data_dir=LOCAL_AWS_DATA)

    assert metadata.loc[0, "Code"] == "FVG0001"
    assert data["Date"].tolist() == [
        pd.Timestamp("2007-01-01"),
        pd.Timestamp("2007-01-02"),
    ]


def test_parse_aws_metadata_directory(tmp_path):
    metadata = (
        "Code,Name,Longitude,Latitude,Elevation,T,TMIN,TMAX,P\n"
        "FVG0001,Alesso,13.056,46.309,213,x,,x,\n"
    )
    (tmp_path / "ARPA_FVG_meta.txt").write_text(metadata)

    result = parse_aws_metadata(tmp_path)

    expected = pd.DataFrame(
        {
            "Code": ["FVG0001"],
            "Name": ["Alesso"],
            "Longitude": [13.056],
            "Latitude": [46.309],
            "Elevation": [213],
            "T": [True],
            "TMIN": [False],
            "TMAX": [True],
            "P": [False],
            "Source": ["ARPA_FVG"],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_parse_aws_metadata_rejects_missing_required_columns(tmp_path):
    (tmp_path / "bad_meta.txt").write_text("Code,Name,Latitude\nA,Station,46.0\n")

    with pytest.raises(ValueError, match="missing required columns"):
        parse_aws_metadata(tmp_path)


def test_load_aws_data(tmp_path):
    provider_dir = tmp_path / "ARPA_FVG"
    provider_dir.mkdir()
    data_file = provider_dir / "FVG0072_1837_13.47_46.375_20071003-20201231.txt"
    data_file.write_text(
        "Date,T,TMIN,TMAX,P,T_QC\n"
        "2007-10-03,,11,15.9,0,\n"
        "2007-10-04,11.6,9.8,13.1,0.4,11.6\n"
    )

    result = load_aws_data("FVG0072", data_dir=tmp_path)

    assert result["Date"].tolist() == [
        pd.Timestamp("2007-10-03"),
        pd.Timestamp("2007-10-04"),
    ]
    assert result["T"].isna().iloc[0]
    assert result.loc[1, "T"] == 11.6
    assert result["T_QC"].isna().iloc[0]


def test_load_aws_data_rejects_unknown_code(tmp_path):
    with pytest.raises(FileNotFoundError, match="No data file found"):
        load_aws_data("UNKNOWN", data_dir=tmp_path)


def test_load_aws_monthly_precipitation_discards_incomplete_months(tmp_path):
    provider_dir = tmp_path / "ARPA_FVG"
    provider_dir.mkdir()
    data_file = provider_dir / "FVG0072_1837_13.47_46.375_20070101-20070228.txt"
    rows = ["Date,T,TMIN,TMAX,P"]
    rows.extend(f"2007-01-{day:02d},0,0,0,3" for day in range(1, 32))
    rows.extend(f"2007-02-{day:02d},0,0,0,6" for day in range(1, 28))
    rows.remove("2007-02-14,0,0,0,6")
    data_file.write_text("\n".join(rows) + "\n")

    result = load_aws_monthly_precipitation("FVG0072", data_dir=tmp_path)

    expected = pd.DataFrame({"Date": [pd.Timestamp("2007-01-01")], "P": [3.0]})
    pd.testing.assert_frame_equal(result, expected)


def test_load_aws_monthly_precipitation_includes_metadata(tmp_path):
    provider_dir = tmp_path / "ARPA_FVG"
    provider_dir.mkdir()
    (tmp_path / "Metadata").mkdir()
    (tmp_path / "Metadata" / "ARPA_FVG_meta.txt").write_text(
        "Code,Name,Longitude,Latitude,Elevation\n" "FVG0072,Station,13.0,46.0,1837\n"
    )
    data_file = provider_dir / "FVG0072_1837_13.0_46.0_20070101-20070131.txt"
    rows = ["Date,T,TMIN,TMAX,P"]
    rows.extend(f"2007-01-{day:02d},0,0,0,3" for day in range(1, 32))
    data_file.write_text("\n".join(rows) + "\n")
    rgi_gdf = gpd.GeoDataFrame(
        {"RGIId": ["RGI_TEST"]},
        geometry=[box(12.99, 45.99, 13.01, 46.01)],
        crs="EPSG:4326",
    )

    result = load_aws_monthly_precipitation(
        "FVG0072",
        data_dir=tmp_path,
        include_metadata=True,
        rgi_gdf=rgi_gdf,
    )

    assert result.columns.tolist() == [
        "Date",
        "P",
        "Latitude",
        "Longitude",
        "Elevation",
        "RGIId",
    ]
    assert result.loc[0, "Latitude"] == 46.0
    assert result.loc[0, "Longitude"] == 13.0
    assert result.loc[0, "Elevation"] == 1837
    assert result.loc[0, "RGIId"] == "RGI_TEST"


def test_check_aws_glacier_proximity():
    metadata = pd.DataFrame(
        {
            "Code": ["inside", "near", "far"],
            "Longitude": [13.0, 13.0, 13.0],
            "Latitude": [46.0, 46.018, 46.03],
        }
    )
    rgi_gdf = gpd.GeoDataFrame(
        {"RGIId": ["RGI_TEST"]},
        geometry=[box(12.99, 45.99, 13.01, 46.01)],
        crs="EPSG:4326",
    )

    result = check_aws_glacier_proximity(metadata, rgi_gdf=rgi_gdf)

    assert result.loc[result.Code == "inside", "inside_glacier"].item()
    assert result.loc[result.Code == "near", "within_1km"].item()
    assert not result.loc[result.Code == "near", "inside_glacier"].item()
    assert not result.loc[result.Code == "far", "within_1km"].item()


def test_interpolate_glacier_topography_returns_nan_outside_grid():
    metadata = pd.DataFrame({"POINT_LON": [2.0], "POINT_LAT": [0.5], "P": [1.0]})
    ds = xr.Dataset(
        {
            "aspect": (("y", "x"), [[1.0, 2.0], [3.0, 4.0]]),
            "slope": (("y", "x"), [[5.0, 6.0], [7.0, 8.0]]),
        },
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    )
    glacier_indices = (pd.array([0, 0, 1, 1]), pd.array([0, 1, 0, 1]))
    gdir = SimpleNamespace(grid=SimpleNamespace(proj=pyproj.Proj("epsg:4326")))

    result = _interpolate_glacier_topography(metadata, ds, glacier_indices, gdir)

    assert result[["aspect", "slope"]].isna().all().all()


def _run_with_tmp_path(test):
    with TemporaryDirectory() as directory:
        test(Path(directory))


if __name__ == "__main__":
    test_load_aws_data_from_local_sample()
    _run_with_tmp_path(test_parse_aws_metadata_directory)
    _run_with_tmp_path(test_parse_aws_metadata_rejects_missing_required_columns)
    _run_with_tmp_path(test_load_aws_data)
    _run_with_tmp_path(test_load_aws_data_rejects_unknown_code)
    _run_with_tmp_path(test_load_aws_monthly_precipitation_discards_incomplete_months)
    _run_with_tmp_path(test_load_aws_monthly_precipitation_includes_metadata)
    test_check_aws_glacier_proximity()
    test_interpolate_glacier_topography_returns_nan_outside_grid()
