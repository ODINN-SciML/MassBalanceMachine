from pathlib import Path
import pandas as pd

from .metadata import check_aws_glacier_proximity, parse_aws_metadata
from .download import _ensure_dataset, eear_dir


def load_aws_data(
    aws_code: str,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load the daily data for one AWS.

    Parameters
    ----------
    aws_code
        Station code, for example ``"FVG0072"``.
    data_dir
        Local EEAR-Clim directory. When omitted, EEAR-Clim is downloaded. Used mainly for the tests.

    Returns
    -------
    pandas.DataFrame
        Daily station data with ``Date`` parsed as datetime and all other
        columns converted to numeric values where possible.

    Raises
    ------
    FileNotFoundError
        If the data directory does not exist or no station file matches.
    ValueError
        If ``aws_code`` is empty or multiple station files match it.
    """
    if not isinstance(aws_code, str) or not aws_code.strip():
        raise ValueError("aws_code must be a non-empty string")

    if data_dir is None:
        _ensure_dataset()
        data_dir = Path(eear_dir)
    else:
        data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"AWS data directory does not exist: {data_dir}")

    matches = sorted(
        path
        for path in data_dir.rglob(f"{aws_code}_*.txt")
        if path.parent.name != "Metadata"
    )
    if not matches:
        raise FileNotFoundError(
            f"No data file found for AWS {aws_code!r} in {data_dir}"
        )
    if len(matches) > 1:
        raise ValueError(f"Multiple data files found for AWS {aws_code!r}: {matches}")

    data = pd.read_csv(matches[0], encoding="latin-1")
    if "Date" not in data.columns:
        raise ValueError(f"AWS data file {matches[0]} does not contain a Date column")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    if data["Date"].isna().any():
        raise ValueError(f"AWS data file {matches[0]} contains invalid dates")

    for column in data.columns:
        if column != "Date":
            data[column] = pd.to_numeric(data[column], errors="coerce")

    return data


def load_aws_monthly_precipitation(
    aws_code: str,
    include_metadata: bool = False,
    region_id: int | str = 11,
    rgi_gdf=None,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load complete-month mean daily precipitation for one AWS.

    The returned ``P`` values are monthly mean precipitation per day in mm.
    Months are discarded when any calendar day is missing, when ``P`` is
    missing for a day, or when duplicate dates prevent a complete daily record.
    If ``include_metadata`` is true, the result also contains ``Latitude``,
    ``Longitude``, ``Elevation``, and the nearest ``RGIId`` from
    ``check_aws_glacier_proximity``.
    """
    data = load_aws_data(aws_code, data_dir=data_dir)
    if "P" not in data.columns:
        raise ValueError(f"AWS data for {aws_code!r} does not contain a P column")

    data = data[["Date", "P"]].copy()
    data["month"] = data["Date"].dt.to_period("M")
    monthly = data.groupby("month", sort=True).agg(
        observed_days=("Date", "size"),
        precipitation_days=("P", "count"),
        P=("P", "mean"),
    )
    monthly["expected_days"] = monthly.index.days_in_month
    complete = monthly["observed_days"].eq(monthly["expected_days"]) & monthly[
        "precipitation_days"
    ].eq(monthly["expected_days"])

    result = monthly.loc[complete, ["P"]].reset_index()
    result["Date"] = result.pop("month").dt.to_timestamp()
    result = result[["Date", "P"]]

    if include_metadata:
        if data_dir is None:
            metadata_path = None
        else:
            metadata_path = Path(data_dir) / "Metadata"
        metadata = parse_aws_metadata(metadata_path)
        station_metadata = metadata[metadata["Code"].eq(aws_code)]
        if station_metadata.empty:
            raise ValueError(f"No metadata found for AWS {aws_code!r}")
        if len(station_metadata) > 1:
            raise ValueError(f"Multiple metadata rows found for AWS {aws_code!r}")

        station_metadata = check_aws_glacier_proximity(
            station_metadata,
            region_id=region_id,
            rgi_gdf=rgi_gdf,
        )
        metadata_columns = ["Latitude", "Longitude", "Elevation", "RGIId"]
        for column in metadata_columns:
            result[column] = station_metadata.iloc[0][column]
        result = result[["Date", "P", *metadata_columns]]

    return result
