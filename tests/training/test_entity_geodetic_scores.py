import numpy as np


def test_entity_geodetic_table_merges_outlines_area_weighted():
    from training.training import entityGeodeticTable

    # Two RGI outlines of entity "E" (3 and 1 km2) and one glacier of no entity
    target = {"a": np.array([-1.0]), "b": np.array([-2.0]), "c": np.array([0.5])}
    pred = {"a": np.array([-0.5]), "b": np.array([-1.5]), "c": np.array([0.0])}
    err = {"a": np.array([0.2]), "b": np.array([0.6]), "c": np.array([0.3])}
    area = {"a": np.array([3.0]), "b": np.array([1.0]), "c": np.array([2.0])}

    t = entityGeodeticTable(target, err, pred, area, {"a": "E", "b": "E"})

    assert sorted(t.index) == ["E", "c"]
    assert np.isclose(t.loc["E", "target"], -1.25)
    assert np.isclose(t.loc["E", "pred"], -0.75)
    assert np.isclose(t.loc["E", "err"], 0.3)  # fully correlated: area-weighted mean
    assert t.loc["E", "area"] == 4.0 and t.loc["E", "n_glaciers"] == 2
    assert t.loc["c", "target"] == 0.5 and t.loc["c", "n_glaciers"] == 1
