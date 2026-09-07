"""Utilities for loading AWS metadata and checking glacier proximity."""

from pathlib import Path

import geopandas as gpd
import pandas as pd


REQUIRED_METADATA_COLUMNS = {
    "Code",
    "Name",
    "Longitude",
    "Latitude",
    "Elevation",
}
AVAILABILITY_COLUMNS = ("T", "TMIN", "TMAX", "P")


def parse_aws_metadata(metadata_path: str | Path) -> pd.DataFrame:
    """Load one or more EEAR-Clim AWS metadata files."""
    metadata_path = Path(metadata_path)
    if metadata_path.is_dir():
        metadata_files = sorted(metadata_path.glob("*_meta.txt"))
        if not metadata_files:
            raise FileNotFoundError(
                f"No '*_meta.txt' files found in metadata directory: {metadata_path}"
            )
    elif metadata_path.is_file():
        metadata_files = [metadata_path]
    else:
        raise FileNotFoundError(f"Metadata path does not exist: {metadata_path}")

    frames = []
    for metadata_file in metadata_files:
        frame = pd.read_csv(
            metadata_file,
            encoding="latin-1",
            na_values=[""],
            keep_default_na=True,
        )
        missing_columns = REQUIRED_METADATA_COLUMNS.difference(frame.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{metadata_file} is missing required columns: {missing}")

        for column in ("Longitude", "Latitude", "Elevation"):
            values = pd.to_numeric(frame[column], errors="coerce")
            invalid = frame[column].notna() & values.isna()
            if invalid.any():
                raise ValueError(
                    f"{metadata_file} contains non-numeric values in {column}"
                )
            frame[column] = values

        for column in AVAILABILITY_COLUMNS:
            if column in frame:
                frame[column] = frame[column].eq("x")

        frame["Source"] = metadata_file.stem.removesuffix("_meta")
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def check_aws_glacier_proximity(
    metadata: pd.DataFrame,
    region_id: int | str = 11,
    max_distance_m: float = 1000,
    rgi_gdf: gpd.GeoDataFrame | None = None,
) -> pd.DataFrame:
    """Find the nearest RGI glacier and proximity for every AWS.

    ``metadata`` must contain ``Code``, ``Longitude``, and ``Latitude``.
    Distances are computed in metres using EPSG:3035. An outline GeoDataFrame
    can be supplied to avoid loading the RGI region data repeatedly.
    """
    if rgi_gdf is None:
        from data_processing.glacier_utils import get_region_shape_file

        shp_path = get_region_shape_file(region_id)
        rgi_gdf = gpd.read_file(shp_path)

    aws = gpd.GeoDataFrame(
        metadata.copy(),
        geometry=gpd.points_from_xy(metadata.Longitude, metadata.Latitude),
        crs="EPSG:4326",
    )
    rgi_gdf = rgi_gdf[["RGIId", "geometry"]].to_crs("EPSG:3035")
    aws_metric = aws.to_crs("EPSG:3035")

    nearest = gpd.sjoin_nearest(
        aws_metric[["Code", "geometry"]],
        rgi_gdf,
        how="left",
        distance_col="distance_to_glacier_m",
    )
    nearest = nearest.groupby("Code", as_index=False).agg(
        distance_to_glacier_m=("distance_to_glacier_m", "min"),
        RGIId=("RGIId", lambda values: ", ".join(sorted(values.dropna().unique()))),
    )

    result = metadata.merge(nearest, on="Code", how="left", validate="one_to_one")
    result["inside_glacier"] = result["distance_to_glacier_m"].eq(0)
    result["within_1km"] = result["distance_to_glacier_m"].le(max_distance_m)
    return result
