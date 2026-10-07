"""Per-region configs: merging the region blocks, and splitting each region's stakes with
its own validation rule."""

import numpy as np
import pandas as pd
import pytest
import yaml

import massbalancemachine as mbm
from cli.common import mergeRegions, trainValData

EUROPE = {
    "rgi_regions": [11],
    "splitVal": "group-year",
    "val_years": [1990, 1991],
    "val_glaciers": ["RGI60-11.00002"],
    "train_glaciers": ["RGI60-11.00001", "RGI60-11.00002"],
    "test_glaciers": ["RGI60-11.00001"],
    "geodetic_source": ["GLAMOS", "Rabatel16"],
    "geodetic_source_options": {
        "GLAMOS": {"allowed_years": [1980, 1981]},
        "Rabatel16": {"allowed_years": [1980, 1981]},
    },
    "geodetic_source_options_val": {
        "GLAMOS": {"allowed_years": [1990, 1991]},
        "Rabatel16": {"allowed_years": [1990, 1991]},
    },
    "geodetic_source_test": "Hugonnet21",
    "train_glaciers_geo": ["B36-26", "A50i-07"],
    "val_glaciers_geo": ["A50i-07"],
    "test_glaciers_geo": ["RGI60-11.00001"],
}
HIMALAYA = {
    "rgi_regions": [13, 14, 15],
    "splitVal": "group-rgi",
    "val_glaciers": ["RGI60-13.00002"],
    "train_glaciers": ["RGI60-13.00001"],
    "test_glaciers": ["RGI60-15.00001"],
    "geodetic_source": ["Maurer19"],
    "geodetic_source_options": {"Maurer19": {"exclude_lake_terminating": True}},
    "geodetic_source_test": "Maurer19",
    "geodetic_source_test_options": {"period": "2000-2016"},
    "train_glaciers_geo": ["RGI50-13.1"],
    "val_glaciers_geo": ["RGI50-15.2"],
    "test_glaciers_geo": ["RGI50-13.1", "RGI50-15.2"],
}


def test_merge_regions_inline_and_split_file(tmp_path):
    (tmp_path / "splits").mkdir()
    with open(tmp_path / "splits" / "himalaya.yml", "w") as f:
        yaml.safe_dump(HIMALAYA, f)
    training = {
        "splitTest": "YEAR:<2000",
        "wGeo": 0.0,
        "regions": {
            "central_europe": EUROPE,
            "himalaya": {"split_file": "splits/himalaya.yml"},
        },
    }
    merged = mergeRegions(training, str(tmp_path))

    assert merged["source_data"] == "wgms:11,13,14,15"
    assert merged["splitVal"] == "per-region"
    assert merged["splitTest"] == "YEAR:<2000" and merged["wGeo"] == 0.0
    assert merged["train_glaciers_geo"] == ["B36-26", "A50i-07", "RGI50-13.1"]
    assert merged["val_glaciers"] == ["RGI60-11.00002", "RGI60-13.00002"]
    assert merged["geodetic_source"] == ["GLAMOS", "Rabatel16", "Maurer19"]
    assert merged["geodetic_source_options"]["GLAMOS"] == {
        "allowed_years": [1980, 1981]
    }
    assert merged["geodetic_source_options_val"]["GLAMOS"] == {
        "allowed_years": [1990, 1991]
    }
    # A region without validation options uses its training ones on both sides
    assert merged["geodetic_source_options_val"]["Maurer19"] == {
        "exclude_lake_terminating": True
    }
    assert merged["regions"]["himalaya"]["geodetic_source_test_options"] == {
        "period": "2000-2016"
    }
    assert merged["regions"]["central_europe"]["val_years"] == [1990, 1991]
    # Each region keeps its whole block, read from its split file if it has one
    assert merged["regions"]["himalaya"] == HIMALAYA


def test_merge_regions_rejects_conflicts():
    with pytest.raises(AssertionError, match="declared by two regions"):
        mergeRegions(
            {
                "regions": {
                    "a": EUROPE,
                    "b": {**HIMALAYA, "geodetic_source": ["GLAMOS"]},
                }
            },
            ".",
        )
    with pytest.raises(AssertionError, match="belongs to two regions"):
        mergeRegions(
            {"regions": {"a": EUROPE, "b": {**HIMALAYA, "rgi_regions": [11]}}}, "."
        )
    with pytest.raises(AssertionError, match="not in the training section"):
        mergeRegions({"val_years": [1990], "regions": {"a": EUROPE}}, ".")
    with pytest.raises(AssertionError, match="does not match"):
        mergeRegions(
            {"source_data": "wgms:11", "regions": {"a": EUROPE, "b": HIMALAYA}}, "."
        )


def test_train_val_data_per_region():
    rows = [
        # RGIId, year, measurement ID
        ("RGI60-11.00001", 1980, 0),
        ("RGI60-11.00001", 1990, 1),  # validation year
        (
            "RGI60-11.00002",
            1981,
            2,
        ),  # validation glacier of Europe, ignored: split by year
        (
            "RGI60-13.00001",
            1990,
            3,
        ),  # validation year of Europe, ignored: split by glacier
        ("RGI60-13.00002", 1980, 4),  # validation glacier
    ]
    df = pd.DataFrame(
        [(g, y, i, m, 0.1 * i, 1.0) for g, y, i in rows for m in ("oct", "nov")],
        columns=["RGIId", "YEAR", "ID", "MONTHS", "POINT_BALANCE", "t2m"],
    ).assign(POINT_LAT=46.0, POINT_LON=8.0, ALTITUDE_CLIMATE=2000.0)
    cfg = mbm.Config(metaData=["RGIId", "ID", "YEAR", "MONTHS"])
    cfg.setFeatures(["t2m"])
    train_set = {"df_X": df, "y": df.POINT_BALANCE}
    merged = mergeRegions({"regions": {"ceu": EUROPE, "him": HIMALAYA}}, ".")

    df_X_train, _, df_X_val, _ = trainValData(
        cfg, train_set, ["t2m"], split_key="per-region", regions=merged["regions"]
    )
    assert sorted(df_X_train.ID.unique()) == [0, 2, 3]
    assert sorted(df_X_val.ID.unique()) == [1, 4]

    # The single-rule splits apply their rule to every stake, as before
    _, _, df_X_val, _ = trainValData(
        cfg, train_set, ["t2m"], split_key="group-year", val_years=[1990, 1991]
    )
    assert sorted(df_X_val.ID.unique()) == [1, 3]
    _, _, df_X_val, _ = trainValData(
        cfg,
        train_set,
        ["t2m"],
        split_key="group-rgi",
        val_glaciers=["RGI60-11.00002", "RGI60-13.00002"],
    )
    assert sorted(df_X_val.ID.unique()) == [2, 4]

    # Stakes of a region no block declares are refused
    with pytest.raises(AssertionError, match="belong to no region"):
        trainValData(
            cfg,
            train_set,
            ["t2m"],
            split_key="per-region",
            regions={"ceu": merged["regions"]["ceu"]},
        )


def test_test_splits_per_region():
    from cli.common import testSplits

    df_test = pd.DataFrame(
        {
            "RGIId": ["RGI60-11.00001", "RGI60-13.00001", "RGI60-15.00001"],
            "ID": [0, 1, 2],
        }
    )
    merged = mergeRegions({"regions": {"ceu": EUROPE, "him": HIMALAYA}}, ".")
    splits = testSplits({"training": merged}, df_test)

    assert list(splits) == ["ceu", "him"]
    assert list(splits["ceu"]["stakes"].ID) == [0]
    assert list(splits["him"]["stakes"].ID) == [1, 2]
    assert splits["ceu"]["geodeticSource"] == "Hugonnet21"
    assert splits["ceu"]["glaciers"] == ["RGI60-11.00001"]
    assert splits["him"]["geodeticSource"] == "Maurer19"
    assert splits["him"]["geodeticSourceOptions"] == {"period": "2000-2016"}

    # Without regions, a single test set on the flat keys, as before
    flat = {
        "training": {
            "geodetic_source": "Hugonnet21",
            "test_glaciers_geo": ["RGI60-11.00001"],
        }
    }
    (name,) = testSplits(flat, df_test)
    assert name is None
    assert testSplits(flat, df_test)[None]["stakes"] is df_test
