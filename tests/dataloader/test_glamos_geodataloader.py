"""GeoDataLoader on glacier grids built from GLAMOS / Swiss Glacier Inventory outlines.

This exercises the whole custom-outline path end to end: the SGI outlines are turned
into OGGM glacier directories, the per-year gridded features are generated on that
geometry, assembled over the geodetic window, and finally read back through
`GeoDataLoader` together with the GLAMOS target and its uncertainty.

To be a real test of that path it has to start from nothing, so it deletes the
products it is about to rebuild. Those deletions are the risky part of the test: the
`.data/` tree also holds the downloaded GLAMOS *inputs*, the grids of the other
geodetic sources and the stake measurements, none of which are reproducible in a
test run. Every path is therefore derived from the package's own path functions and
checked against an allow-list of three GLAMOS *product* roots before it is removed
(`_assert_purgeable`), and the test asserts afterwards that the neighbouring trees
are untouched.

Note `.data/GLAMOS` (the volume-change table and the SGI shapefile) is an input and
is never purged, while `.data/grids/GLAMOS` is a product and is.
"""

import os
import shutil

import numpy as np
import pandas as pd
import pytest

import massbalancemachine as mbm
from data_processing.glacier_utils import is_usable_rvt_venv, rvt_venv_path
from data_processing.glamos import geodetic_target_GLAMOS, glamos_outline_spec
from data_processing.gridded_utils import (
    load_grid_multi_years,
    prepared_grid_dir_multi_years,
    prepared_metadata_dir,
    years_from_time_range,
)
from data_processing.product_utils import data_path, glacier_id_to_folders

# The glacier the assertions are about: Pizolgletscher, the smallest SGI entity with
# a usable window, and one that maps to a single RGI id.
SGI_ID = "A50d-01"
RGI_IDS = ["RGI60-11.00638"]

# GeoDataLoader raises NotImplementedError when exactly one glacier carries geodetic
# data, so a second entity is needed to reach the code under test. Plattalva /
# Griessfirn is the next smallest, and it maps to *two* RGI ids, which covers the
# many-RGI-to-one-SGI case that the Swiss inventory makes unavoidable.
SGI_ID_SECOND = "A50i-07"
RGI_IDS_SECOND = ["RGI60-11.00892", "RGI60-11.00901"]

SGI_EPOCH = 1973

# Restricting the eligible windows to 1985-2000 keeps the test to ~12 and ~14 years
# per glacier instead of the 36 and 53 the unrestricted selection would pick. The
# lower bound also has to stay inside the ERA5 forcing.
TARGET_OPTIONS = dict(min_year=1985, max_year=2000, min_window_years=10)

# The only three trees this test is allowed to delete from. Everything else under
# `.data/` belongs to another source or is a download that a test must not throw away.
PURGEABLE_ROOTS = (
    os.path.join(data_path, "grids", "GLAMOS"),
    os.path.join(data_path, "grids_multiyears", "GLAMOS"),
    os.path.join(data_path, "oggm", "GLAMOS"),
)

# Trees that must come out of the test exactly as they went in.
PROTECTED_TREES = (
    os.path.join(data_path, "GLAMOS"),  # the downloaded GLAMOS inputs
    os.path.join(data_path, "grids", "PGO"),
    os.path.join(data_path, "grids", "Hugonnet21"),
    os.path.join(data_path, "grids_multiyears", "PGO"),
    os.path.join(data_path, "grids_multiyears", "Hugonnet21"),
    os.path.join(data_path, "oggm", "PGO"),
    os.path.join(data_path, "ERA5"),
    os.path.join(data_path, "stakes"),
)


def _assert_purgeable(path):
    """Refuse to delete anything that is not inside a GLAMOS product root.

    `os.path.commonpath` on the resolved paths rather than a string prefix, so that
    neither a symlink nor a sibling whose name merely starts the same way (`.data/
    GLAMOS` next to `.data/grids/GLAMOS`) can be mistaken for a purgeable product.
    """
    resolved = os.path.realpath(path)
    for root in PURGEABLE_ROOTS:
        root = os.path.realpath(root)
        if resolved == root:
            continue  # a root itself is shared with other glaciers, never removed
        if os.path.commonpath([resolved, root]) == root:
            return
    raise AssertionError(
        f"{path} is not inside one of the GLAMOS product roots {PURGEABLE_ROOTS} "
        "and must not be deleted by this test."
    )


def _glacier_product_paths(sgi_id, time_range, feature_columns):
    """Every path holding a product derived from one GLAMOS glacier.

    Built from the same functions the pipeline writes with, so the test cannot drift
    away from the layout it is meant to clear.
    """
    paths = [
        # the per-year grids, the DEM and the sky view factor
        os.path.join(
            glamos_outline_spec(SGI_EPOCH).grid_root(), *glacier_id_to_folders(sgi_id)
        ),
    ]
    # the OGGM glacier directory, wherever OGGM chose to nest it
    per_glacier = os.path.join(data_path, "oggm", "GLAMOS", "per_glacier")
    for dirpath, dirnames, _ in os.walk(per_glacier):
        for d in dirnames:
            if d == sgi_id:
                paths.append(os.path.join(dirpath, d))
    # the multi-year assembly and the precomputed metadata. These directories are
    # shared with the other glaciers of the same id group, so only this glacier's own
    # files are listed, never the directory.
    years = [time_range]
    for folder in (
        prepared_grid_dir_multi_years(sgi_id, years, "GLAMOS"),
        prepared_metadata_dir(sgi_id, years, "GLAMOS", feature_columns),
    ):
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name == f"{sgi_id}.parquet" or name.startswith(f"{sgi_id}.parquet."):
                paths.append(os.path.join(folder, name))
            elif name.startswith(f"{sgi_id}_"):
                paths.append(os.path.join(folder, name))
    return paths


def _purge(paths):
    for path in paths:
        _assert_purgeable(path)
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)


def _prune_empty_dirs():
    """Remove the directories a purge emptied, so a run leaves no trace. Directories
    still holding another glacier's products are left alone, and the three roots
    themselves are never removed."""
    for root in PURGEABLE_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, _, _ in os.walk(root, topdown=False):
            if os.path.realpath(dirpath) == os.path.realpath(root):
                continue
            if not os.listdir(dirpath):
                _assert_purgeable(dirpath)
                os.rmdir(dirpath)


def _tree_signature(path):
    """Number of files and total size below `path`, enough to notice a tree losing
    or truncating anything."""
    if not os.path.exists(path):
        return None
    n, total = 0, 0
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            n += 1
            total += os.path.getsize(os.path.join(dirpath, name))
    return n, total


def test_purge_refuses_to_delete_anything_but_glamos_products():
    """The guard standing between this test and the rest of `.data/`."""
    for forbidden in PROTECTED_TREES:
        with pytest.raises(AssertionError):
            _assert_purgeable(forbidden)
    # a product root itself is shared with other glaciers and is not purgeable either
    for root in PURGEABLE_ROOTS:
        with pytest.raises(AssertionError):
            _assert_purgeable(root)
    # ... but one glacier's products below it are
    _assert_purgeable(os.path.join(data_path, "grids", "GLAMOS", "A50d", "A50d-01"))
    _assert_purgeable(
        os.path.join(data_path, "oggm", "GLAMOS", "per_glacier", "A50d", "A50d-01")
    )


@pytest.mark.integration
def test_geodataloader_on_glamos_outlines():
    if not is_usable_rvt_venv(rvt_venv_path()):
        pytest.skip(
            "the sky view factor environment at "
            f"{rvt_venv_path()} is not built (it needs libgdal), so the gridded "
            "features cannot be generated from scratch"
        )

    cfg = mbm.Config(
        metaData=["RGIId", "POINT_ID", "ID", "N_MONTHS", "MONTHS", "PERIOD"]
    )
    cfg.setFeatures(["t2m", "tp", "slope", "aspect", "svf", "ELEVATION_DIFFERENCE"])

    target = geodetic_target_GLAMOS(**TARGET_OPTIONS)
    row = target.loc[target.RGIId == SGI_ID].iloc[0]
    time_range = (row.FROM_DATE, row.TO_DATE)

    def purge_products():
        """The paths are resolved afresh on every call on purpose: the multi-year
        assemblies and the OGGM glacier directories do not exist yet when the test
        clears the ground, and they have to be removed when it is done."""
        for sgi_id in (SGI_ID, SGI_ID_SECOND):
            window = target.loc[target.RGIId == sgi_id].iloc[0]
            _purge(
                _glacier_product_paths(
                    sgi_id, (window.FROM_DATE, window.TO_DATE), cfg.featureColumns
                )
            )
        _prune_empty_dirs()

    before = {tree: _tree_signature(tree) for tree in PROTECTED_TREES}
    purge_products()
    try:
        stakes = _fake_stakes(RGI_IDS + RGI_IDS_SECOND)
        dataloader = mbm.dataloader.GeoDataLoader(
            cfg,
            glacierList=RGI_IDS + RGI_IDS_SECOND,
            trainStakesDf=stakes,
            months_head_pad=[],
            months_tail_pad=[],
            geodeticSource="GLAMOS",
            geodeticSourceOptions=TARGET_OPTIONS,
            keyGlacierSel="GLACIER",
        )

        # The RGI ids of the glacier list are resolved to SGI entities, several to one
        assert dataloader.glacier_id_map == {
            **{r: SGI_ID for r in RGI_IDS},
            **{r: SGI_ID_SECOND for r in RGI_IDS_SECOND},
        }
        assert set(dataloader.glaciersWithGeo) == {SGI_ID, SGI_ID_SECOND}
        assert all(dataloader.hasGeo(r) for r in RGI_IDS + RGI_IDS_SECOND)

        # The window comes from GLAMOS, not from a fixed set of calendar years
        (period,) = dataloader.geodetic_periods(RGI_IDS[0])
        assert [pd.Timestamp(d) for d in period] == list(time_range)
        assert dataloader.years is None

        # The grids were built on the SGI outline, one folder per SGI id
        grid_dir = os.path.join(
            glamos_outline_spec(SGI_EPOCH).grid_root(), *glacier_id_to_folders(SGI_ID)
        )
        assert os.path.isfile(os.path.join(grid_dir, "dem.nc"))
        assert os.path.isfile(os.path.join(grid_dir, "svf.nc"))
        expected_years = list(years_from_time_range(*time_range))
        assert (
            sorted(int(f[:-8]) for f in os.listdir(grid_dir) if f.endswith(".parquet"))
            == expected_years
        )

        one_year = pd.read_parquet(
            os.path.join(grid_dir, f"{expected_years[0]}.parquet")
        )
        # Glaciers of a custom source are keyed by their own id, not by an RGI one
        assert one_year.RGIId.unique().tolist() == [SGI_ID]

        features, metadata, y, err, precomputed_meta = dataloader.geo(RGI_IDS[0])

        assert features.shape[0] == len(metadata)
        assert features.shape[1] == len(cfg.featureColumns)
        # One month of the window per aggregation group
        assert precomputed_meta["nunique_glwd_m_ids"] == 12 * len(expected_years)
        # The GLAMOS rate and its own uncertainty both reach the model
        assert y.item() == pytest.approx(row.mwe_per_year)
        assert err.item() == pytest.approx(row.sigma_mwe_per_year)
        assert err.item() > 0

        # The assembled window is in degrees, not the radians OGGM stores
        assembled = load_grid_multi_years(SGI_ID, [time_range], "GLAMOS")
        assert assembled.aspect.between(0, 360).all()
        assert assembled.slope.between(0, 90).all()
        assert metadata.ELEVATION_DIFFERENCE.notna().all()
        dataloader.close()
    finally:
        purge_products()

    after = {tree: _tree_signature(tree) for tree in PROTECTED_TREES}
    assert after == before, "the test modified data outside the GLAMOS products"


def _fake_stakes(rgi_ids):
    """A minimal stakes frame. The geodetic path is what is under test here, but
    GeoDataLoader needs stake data to be constructed at all."""
    n = len(rgi_ids) * 2
    return pd.DataFrame(
        {
            "RGIId": (rgi_ids * 2),
            "GLACIER": (rgi_ids * 2),
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


if __name__ == "__main__":
    test_purge_refuses_to_delete_anything_but_glamos_products()
    test_geodataloader_on_glamos_outlines()
