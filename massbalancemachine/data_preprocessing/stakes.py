"""
Stake measurements of an RGI region combined from several sources.

The sources are combined at the level of the point measurements, before any feature is
computed: the points of every source are put in the format of the WGMS point table,
assigned to their RGI glacier, filtered, de-duplicated, and the topographical and climate
features are then computed once for all of them by the WGMS pipeline. All the stakes of
the monthly table therefore share the same feature definitions, whatever their source.
"""

import os
import numpy as np
import pandas as pd

from data_processing.product_utils import get_data_path
from data_processing.Product import Product
from data_processing.utils import get_rgi
from data_processing.wgms import load_processed_wgms
import data_preprocessing.wgms
import data_preprocessing.nve

# Source name: RGI regions it covers (None for all)
STAKE_SOURCES = {
    "wgms": None,
    "nve": [8],
}
DEFAULT_STAKE_SOURCES = ["wgms"]

POINT_COLUMNS = [
    "YEAR",
    "FROM_DATE",
    "TO_DATE",
    "POINT_LAT",
    "POINT_LON",
    "POINT_ELEVATION",
    "POINT_BALANCE",
    "PERIOD",
    "RGIId",
    "FROM_DATE_DT",
    "TO_DATE_DT",
    "MONTH_DIFF",
]


def combined_stakes_folder(rgi_region):
    return os.path.join(get_data_path(), "stakes", "combined", f"region_{rgi_region}")


def combined_monthly_path(rgi_region, sources):
    return os.path.join(combined_stakes_folder(rgi_region), "+".join(sources) + ".csv")


def combined_points_path(rgi_region, sources):
    return os.path.join(
        combined_stakes_folder(rgi_region), "+".join(sources) + "_points.csv"
    )


def load_points(source, rgi_region):
    """The stake measurements of `source` in `rgi_region`, assigned to their RGI glacier
    and filtered with `data_preprocessing.wgms.filter_points`."""
    regions = STAKE_SOURCES.get(source, [])
    assert (
        regions is None or rgi_region in regions
    ), f"Stake source {source} is unknown or does not cover RGI region {rgi_region}."
    if source == "wgms":
        points = data_preprocessing.wgms.prepare_points(
            load_processed_wgms(rgi_region=rgi_region), rgi_region
        )
    elif source == "nve":
        points = data_preprocessing.nve.load_points()
        points = get_rgi(data=points, region=rgi_region)
        points = data_preprocessing.wgms.filter_points(points)
    return points[POINT_COLUMNS].assign(SOURCE=source).reset_index(drop=True)


def drop_duplicate_measurements(
    points, max_days=10, max_distance=100, max_balance_diff=0.05
):
    """Drop the measurements reported by several sources: same glacier and period, start
    and end dates within `max_days` days, positions within `max_distance` m and balances
    within `max_balance_diff` m w.e. The measurement of the source listed first in
    `points.SOURCE` is kept."""
    sources = list(dict.fromkeys(points.SOURCE))
    points = points.reset_index(drop=True)
    duplicates = []
    for i, source in enumerate(sources[1:], start=1):
        kept = points[points.SOURCE.isin(sources[:i])].drop(index=duplicates)
        candidates = points[points.SOURCE == source].reset_index()
        pairs = candidates.merge(
            kept, on=["RGIId", "PERIOD", "YEAR"], suffixes=("", "_kept")
        )
        distance = np.hypot(
            (pairs.POINT_LAT - pairs.POINT_LAT_kept) * 111e3,
            (pairs.POINT_LON - pairs.POINT_LON_kept)
            * 111e3
            * np.cos(np.radians(pairs.POINT_LAT)),
        )
        is_duplicate = (
            ((pairs.FROM_DATE_DT - pairs.FROM_DATE_DT_kept).abs().dt.days <= max_days)
            & ((pairs.TO_DATE_DT - pairs.TO_DATE_DT_kept).abs().dt.days <= max_days)
            & (distance <= max_distance)
            & (
                (pairs.POINT_BALANCE - pairs.POINT_BALANCE_kept).abs()
                <= max_balance_diff
            )
        )
        found = pairs.loc[is_duplicate, "index"].unique().tolist()
        print(
            f"{source}: {len(found)} of {len(candidates)} measurements already provided by {sources[:i]}"
        )
        duplicates += found
    return points.drop(index=duplicates).reset_index(drop=True)


def combine_points(rgi_region, sources):
    """The stake measurements of `rgi_region` from all `sources`, without duplicates."""
    points = pd.concat(
        [load_points(source, rgi_region) for source in sources], ignore_index=True
    )
    return drop_duplicate_measurements(points)


def load_monthly_data(rgi_region, sources=None, cfg=None):
    """The monthly stake table of an RGI region combining the stake measurements of
    `sources` (default `DEFAULT_STAKE_SOURCES`), built on first use and read from the data
    tree afterwards. With WGMS alone, it is the table of
    `data_preprocessing.wgms.load_monthly_data`. The combined point table, with the
    source of each measurement in the column SOURCE, is written next to the monthly
    table. `cfg` defaults to `data_preprocessing.wgms.default_monthly_config()`."""
    sources = list(sources or DEFAULT_STAKE_SOURCES)
    assert len(set(sources)) == len(sources), f"Stake sources repeated: {sources}"
    if sources == ["wgms"]:
        return data_preprocessing.wgms.load_monthly_data(rgi_region, cfg)

    assert isinstance(
        rgi_region, int
    ), "Stake sources can only be combined for a single RGI region at a time."
    path = combined_monthly_path(rgi_region, sources)
    product = Product(path)
    if not product.is_up_to_date():
        points = combine_points(rgi_region, sources)
        points.to_csv(combined_points_path(rgi_region, sources), index=False)
        print(
            "Stake measurements per source:",
            points.SOURCE.value_counts().to_dict(),
        )
        data_preprocessing.wgms.points_to_monthly(
            points.drop(columns=["SOURCE"]),
            data_preprocessing.wgms.default_monthly_config() if cfg is None else cfg,
            rgi_region,
            path,
        )
        product.gen_chk()
    return pd.read_csv(path)
