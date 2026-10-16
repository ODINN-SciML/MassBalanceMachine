import pandas as pd
import massbalancemachine as mbm


def _points(source, rows):
    df = pd.DataFrame(
        rows,
        columns=[
            "FROM_DATE",
            "TO_DATE",
            "POINT_LAT",
            "POINT_LON",
            "POINT_BALANCE",
            "PERIOD",
        ],
    )
    df["FROM_DATE_DT"] = pd.to_datetime(df.FROM_DATE, format="%Y%m%d")
    df["TO_DATE_DT"] = pd.to_datetime(df.TO_DATE, format="%Y%m%d")
    df["YEAR"] = df.TO_DATE_DT.dt.year
    return df.assign(RGIId="RGI60-08.00001", SOURCE=source)


def test_drop_duplicate_measurements():
    first = _points(
        "wgms",
        [
            ["20101001", "20110501", 61.0, 8.0, 1.50, "winter"],
            ["20101001", "20111001", 61.0, 8.0, -1.00, "annual"],
        ],
    )
    second = _points(
        "nve",
        [
            # Same measurement: dates 3 days apart, 50 m away, same balance
            ["20101004", "20110501", 61.00045, 8.0, 1.50, "winter"],
            # Same stake and dates but another period
            ["20110501", "20111001", 61.0, 8.0, -2.50, "summer"],
            # Same dates and position, revised balance: still the same measurement
            ["20101001", "20111001", 61.0, 8.0, -1.40, "annual"],
            # Same dates and balance, 1 km away
            ["20101001", "20110501", 61.009, 8.0, 1.50, "winter"],
        ],
    )
    points = mbm.data_preprocessing.stakes.drop_duplicate_measurements(
        pd.concat([first, second], ignore_index=True)
    )
    assert len(points) == 4
    assert points.SOURCE.value_counts().to_dict() == {"wgms": 2, "nve": 2}
    # The measurement of the source listed first is kept, with its balance
    assert not ((points.SOURCE == "nve") & (points.FROM_DATE == "20101004")).any()
    assert points[points.PERIOD == "annual"].POINT_BALANCE.tolist() == [-1.00]
