"""GeoDataLoader on the source Maurer19_Hugonnet21: two windows per glacier, the
1975-2000 one of Maurer19 on the 1975 outlines and the 2000-2010 one of Hugonnet21 on
the outlines of 2000, read from two grid trees.

Like `test_multisource_geodataloader.py`, this test reads the grids already cached
under `.data/grids` and is skipped when they are not there, or when the Maurer19
product is not downloaded (its download needs a NASA Earthdata login).
"""

import glob
import os

import numpy as np
import pytest

import massbalancemachine as mbm
from data_processing.gridded_utils import years_from_time_ranges
from data_processing.maurer19 import (
    END_DATE,
    PERIOD,
    geodetic_target_Maurer19_with_Hugonnet21,
    maurer19_avg_folder,
)
from data_processing.product_utils import get_data_path, glacier_id_to_folders

from tests.dataloader.test_multisource_geodataloader import _cfg, _fake_stakes

GLACIER_ID = "RGI50-13.26243"
SOURCE = "Maurer19_Hugonnet21"


def _skip_without_cached_grids(target):
    windows = list(zip(target.FROM_DATE, target.TO_DATE))
    for year in years_from_time_ranges(windows):
        tree = "Maurer19" if year < END_DATE.year else "Maurer19_2000"
        path = os.path.join(
            get_data_path(),
            "grids",
            tree,
            *glacier_id_to_folders(GLACIER_ID),
            f"{year}.parquet",
        )
        if not os.path.isfile(path):
            pytest.skip(f"no cached {tree} grid for {GLACIER_ID} at {path}")


@pytest.mark.skipif(
    not glob.glob(os.path.join(maurer19_avg_folder(), f"*{PERIOD}*.shp")),
    reason="HMA_GlacierAvg_dH not downloaded",
)
def test_geodataloader_on_maurer19_with_hugonnet21():
    target = geodetic_target_Maurer19_with_Hugonnet21(
        glacier_ids_to_keep=[GLACIER_ID], verbose=False
    )
    assert list(target.target) == ["Maurer19", "Hugonnet21"]
    _skip_without_cached_grids(target)

    cfg = _cfg()
    dataloader = mbm.dataloader.GeoDataLoader(
        cfg,
        glacierList=[GLACIER_ID],
        trainStakesDf=_fake_stakes([GLACIER_ID]),
        months_head_pad=[],
        months_tail_pad=[],
        geodeticSource=[SOURCE],
        geodeticSourceOptions={SOURCE: {"glacier_ids_to_keep": [GLACIER_ID]}},
        keyGlacierSel="GLACIER",
    )
    try:
        assert dataloader.sourceOfGlacier == {GLACIER_ID: SOURCE}
        assert dataloader.geodetic_periods(GLACIER_ID) == list(
            zip(target.FROM_DATE.values, target.TO_DATE.values)
        )
        features, metadata, y, err, _ = dataloader.geo(GLACIER_ID)
        np.testing.assert_allclose(y.numpy(), target.mwe_per_year, rtol=1e-6)
        np.testing.assert_allclose(err.numpy(), target.sigma_mwe_per_year, rtol=1e-6)
        assert features.shape == (len(metadata), len(cfg.featureColumns))

        # Each epoch is read on its own outlines: the number of points per month
        # changes at the switch, and is constant within each epoch
        points = (
            metadata.groupby("YEAR").size() / metadata.groupby("YEAR").MONTHS.nunique()
        )
        before, after = (
            points[points.index < END_DATE.year],
            points[points.index >= END_DATE.year],
        )
        assert before.nunique() == 1 and after.nunique() == 1
        assert before.iloc[0] != after.iloc[0]

        # The area of each window is that of the outlines of its epoch (each on its
        # own OGGM grid, so the pixel size differs too): the glacier shrank
        areas = dataloader.geodetic_areas(GLACIER_ID)
        assert len(areas) == 2 and (areas > 0).all()
        assert areas[1] < areas[0]
    finally:
        dataloader.close()
