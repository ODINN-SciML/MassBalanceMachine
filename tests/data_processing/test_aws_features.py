import pandas as pd

import massbalancemachine.aws.features as aws_features


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
    monkeypatch.setattr(aws_features, "build_features", lambda aws_code: monthly)

    result = aws_features.create_aws_grid("TEST0001")

    assert result["POINT_ID"].tolist() == [1, 1]
    assert result["N_MONTHS"].tolist() == [1, 1]
    assert result["YEAR"].tolist() == [2007, 2007]
    assert result["MONTHS"].tolist() == [1, 2]
    assert result["FROM_DATE"].tolist() == ["20070101", "20070201"]
    assert result["TO_DATE"].tolist() == ["20070131", "20070228"]
    assert result["PERIOD"].tolist() == ["monthly", "monthly"]
