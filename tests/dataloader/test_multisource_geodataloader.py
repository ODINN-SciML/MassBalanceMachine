"""GeoDataLoader combining several windowed geodetic sources, GLAMOS and Rabatel16.

Every glacier keeps the target, the windows and the grids of its own source, and the
glacier lists may mix SGI ids, GLIMS ids and RGI 6.2 ids. Unlike
`test_glamos_geodataloader.py` this test does not rebuild anything: it reads the grids
already cached under `.data/grids`, and is skipped when they are not there, since
generating them needs the sky view factor environment.
"""

import os

import numpy as np
import pandas as pd
import pytest

import massbalancemachine as mbm
from data_processing.product_utils import data_path, glacier_id_to_folders
from dataloader.GeoDataLoader import buildGlacierMappingMultiSource

# Pizolgletscher (one RGI id) and Plattalva / Griessfirn, as in the GLAMOS test
SGI_ID = "A50d-01"
SGI_RGI_ID = "RGI60-11.00638"
SGI_ID_VAL = "A50i-07"

# Glacier du Tour and Glacier de Tré la Tête, each matched to one RGI id
GLIMS_ID = "G006988E45987N"
GLIMS_ID_VAL = "G006784E45784N"
GLIMS_RGI_ID_VAL = "RGI60-11.03651"

GEODETIC_SOURCES = ["GLAMOS", "Rabatel16"]
GEODETIC_SOURCE_OPTIONS = {
    "GLAMOS": dict(min_year=1950),
    "Rabatel16": dict(start_year=1984, end_year=1999),
}


def _cfg():
    cfg = mbm.Config(
        metaData=["RGIId", "POINT_ID", "ID", "N_MONTHS", "MONTHS", "PERIOD"]
    )
    cfg.setFeatures(["t2m", "tp", "slope", "aspect", "svf", "ELEVATION_DIFFERENCE"])
    return cfg


def _fake_stakes(glacier_ids):
    """A minimal stakes frame: GeoDataLoader needs stake data to be constructed."""
    n = len(glacier_ids) * 2
    return pd.DataFrame(
        {
            "RGIId": (glacier_ids * 2),
            "GLACIER": (glacier_ids * 2),
            "ID": np.arange(n),
            "POINT_ID": np.arange(n),
            "N_MONTHS": 12,
            "MONTHS": ["jan"] * n,
            "PERIOD": "annual",
            "YEAR": 1990,
            "POINT_BALANCE": 0.0,
            "POINT_LAT": 46.9,
            "POINT_LON": 9.4,
            "ALTITUDE_CLIMATE": 2500.0,
            "ELEVATION_DIFFERENCE": 100.0,
            "POINT_ELEVATION": 2600.0,
            "t2m": 0.0,
            "tp": 0.0,
            "slope": 20.0,
            "aspect": 180.0,
            "svf": 0.9,
        }
    )


def _skip_without_cached_grids():
    for source, glacier_id in [
        ("GLAMOS", SGI_ID),
        ("GLAMOS", SGI_ID_VAL),
        ("Rabatel16", GLIMS_ID),
        ("Rabatel16", GLIMS_ID_VAL),
    ]:
        svf = os.path.join(
            data_path, "grids", source, *glacier_id_to_folders(glacier_id), "svf.nc"
        )
        if not os.path.isfile(svf):
            pytest.skip(f"no cached {source} grid for {glacier_id} at {svf}")
    try:
        outlines = mbm.data_processing.rabatel16.rabatel16_outlines_file()
    except ValueError as e:  # unknown host
        pytest.skip(f"the raw Rabatel16 files are not available here: {e}")
    if not os.path.isfile(outlines):
        pytest.skip(f"the raw Rabatel16 outlines are not at {outlines}")


def test_mapping_attributes_native_ids_to_the_source_holding_them():
    mapping = buildGlacierMappingMultiSource(
        [SGI_ID, GLIMS_ID, "B99-99"],
        {"GLAMOS": {SGI_ID}, "Rabatel16": {GLIMS_ID}},
    )
    # An SGI id and a GLIMS id are both "native" to either source: only the target
    # tells them apart. An id no target holds has no geodetic data and is left out.
    assert mapping == {"GLAMOS": {SGI_ID: SGI_ID}, "Rabatel16": {GLIMS_ID: GLIMS_ID}}


def test_mapping_refuses_a_glacier_held_by_two_sources():
    with pytest.raises(AssertionError):
        buildGlacierMappingMultiSource(
            [SGI_ID], {"GLAMOS": {SGI_ID}, "Rabatel16": {SGI_ID}}
        )


def test_only_windowed_sources_can_be_combined():
    cfg = _cfg()
    with pytest.raises(AssertionError):
        mbm.dataloader.GeoDataLoader(
            cfg,
            glacierList=[SGI_ID],
            trainStakesDf=_fake_stakes([SGI_ID]),
            months_head_pad=[],
            months_tail_pad=[],
            geodeticSource=["GLAMOS", "Hugonnet21"],
        )
    # With several sources the options are keyed by source
    with pytest.raises(AssertionError):
        mbm.dataloader.GeoDataLoader(
            cfg,
            glacierList=[SGI_ID],
            trainStakesDf=_fake_stakes([SGI_ID]),
            months_head_pad=[],
            months_tail_pad=[],
            geodeticSource=GEODETIC_SOURCES,
            geodeticSourceOptions={"min_year": 1950},
        )


def test_geodataloader_on_glamos_and_rabatel16():
    _skip_without_cached_grids()
    cfg = _cfg()

    target_glamos = mbm.data_processing.glamos.geodetic_target_GLAMOS(
        **GEODETIC_SOURCE_OPTIONS["GLAMOS"]
    )
    target_rabatel = mbm.data_processing.geodetic_target_Rabatel16(
        **GEODETIC_SOURCE_OPTIONS["Rabatel16"]
    )

    # Identifier schemes mixed on purpose: an RGI id and a GLIMS id in training, an
    # SGI id and an RGI id in validation
    train = [SGI_RGI_ID, GLIMS_ID]
    val = [SGI_ID_VAL, GLIMS_RGI_ID_VAL]
    dataloader = mbm.dataloader.GeoDataLoader(
        cfg,
        glacierList=train,
        glacierListVal=val,
        trainStakesDf=_fake_stakes(train),
        valStakesDf=_fake_stakes(val),
        months_head_pad=[],
        months_tail_pad=[],
        geodeticSource=GEODETIC_SOURCES,
        geodeticSourceOptions=GEODETIC_SOURCE_OPTIONS,
        keyGlacierSel="GLACIER",
    )
    try:
        assert dataloader.sourceOfGlacier == {
            SGI_ID: "GLAMOS",
            SGI_ID_VAL: "GLAMOS",
            GLIMS_ID: "Rabatel16",
            GLIMS_ID_VAL: "Rabatel16",
        }
        assert dataloader.glacier_id_map == {SGI_RGI_ID: SGI_ID, GLIMS_ID: GLIMS_ID}
        assert dataloader.glacier_id_map_val == {
            SGI_ID_VAL: SGI_ID_VAL,
            GLIMS_RGI_ID_VAL: GLIMS_ID_VAL,
        }
        assert set(dataloader.glaciersWithGeo) == {SGI_ID, GLIMS_ID}
        assert set(dataloader.glaciersValWithGeo) == {SGI_ID_VAL, GLIMS_ID_VAL}
        assert all(dataloader.hasGeo(g) for g in train + val)

        # Each glacier keeps the window convention of its own source: the 1st of
        # January for GLAMOS, the hydrological year for Rabatel16
        for g in [SGI_RGI_ID, SGI_ID_VAL]:
            for start, end in dataloader.geodetic_periods(g):
                assert (pd.Timestamp(start).month, pd.Timestamp(start).day) == (1, 1)
                assert (pd.Timestamp(end).month, pd.Timestamp(end).day) == (1, 1)
        for g in [GLIMS_ID, GLIMS_RGI_ID_VAL]:
            assert [
                tuple(pd.Timestamp(d) for d in window)
                for window in dataloader.geodetic_periods(g)
            ] == [(pd.Timestamp("1983-10-01"), pd.Timestamp("1999-10-01"))]

        # ... and the rate and uncertainty of its own source
        for g, glacier_id, target in [
            (SGI_RGI_ID, SGI_ID, target_glamos),
            (SGI_ID_VAL, SGI_ID_VAL, target_glamos),
            (GLIMS_ID, GLIMS_ID, target_rabatel),
            (GLIMS_RGI_ID_VAL, GLIMS_ID_VAL, target_rabatel),
        ]:
            rows = target[target.RGIId == glacier_id].sort_values("FROM_DATE")
            features, metadata, y, err, precomputed_meta = dataloader.geo(g)
            assert features.shape == (len(metadata), len(cfg.featureColumns))
            np.testing.assert_allclose(y.numpy(), rows.mwe_per_year, rtol=1e-6)
            np.testing.assert_allclose(err.numpy(), rows.sigma_mwe_per_year, rtol=1e-6)
            # The grid read is the one of the glacier's own source
            assert set(metadata.RGIId) == {glacier_id}
            lo, hi = dataloader.elevation_diff_range(g)
            assert np.isfinite(lo) and np.isfinite(hi) and lo < hi
    finally:
        dataloader.close()
