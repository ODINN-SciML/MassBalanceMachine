"""Tests of the NVE source: the geodetic balances of the NVE glacier database,
their versions, groups of glaciers, units and uncertainty, and the crosswalk to RGI 6.2.

The table is not public and is read from `.data/NVE`, the outlines from OGGM's
local copy of RGI 6.2: every test is skipped when either is missing.
"""

import os

import pandas as pd
import pytest

from data_processing.andreassen16 import ANDREASSEN16_GLACIERS, andreassen16_periods
from data_processing.nve_geodetic import (
    AGGREGATED_PERIODS,
    LANGFJORDJOKELEN,
    RAW_FILE,
    nve_data_folder,
    nve_periods,
    entity_id,
    entity_members,
    geodetic_target_NVE,
    load_nve_outlines,
    read_nve_table,
    table_RGI62_to_NVE,
)


def _data_available():
    try:
        from data_processing.glacier_utils import get_region_shape_file

        return os.path.exists(get_region_shape_file("08")) and os.path.exists(
            os.path.join(nve_data_folder(), RAW_FILE)
        )
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _data_available(), reason="NVE table or RGI 6.2 region 08 not available"
)


@pytest.fixture(scope="module")
def periods():
    return nve_periods()


def test_coordinates_are_read_the_right_way_round():
    raw = read_nve_table()
    assert raw.lat.between(58, 71).all() and raw.lon.between(4, 32).all()


def test_only_current_periods_are_kept(periods):
    raw = read_nve_table()
    assert len(periods) == (raw.generation_no == 0).sum() - 1
    # Ålfotbreen 1988-1997: the reanalysis of 2016, not the value of 2004 (+0.50)
    alf = periods.set_index(["RGIId", "y0", "y1"]).loc[("NVE-2078", 1988, 1997)]
    assert alf.mwe_per_year == pytest.approx(0.87)


def test_every_group_of_the_comments_is_listed():
    raw = read_nve_table()
    raw = raw[raw.generation_no == 0]
    grouped = raw.comment.str.contains(
        "Inkluderer|samlet|15 brefelt", case=True, na=False
    ) & ~raw.comment.str.contains("trolig", na=False)
    keys = set(zip(raw.glacier_id, raw.survey_year1, raw.survey_year2))
    assert set(
        zip(
            raw[grouped].glacier_id,
            raw[grouped].survey_year1,
            raw[grouped].survey_year2,
        )
    ) == set(AGGREGATED_PERIODS).difference({(2148, 1966, 1993)})
    assert set(AGGREGATED_PERIODS) <= keys


def test_groups_are_entities_of_their_own(periods):
    ids = set(periods.RGIId)
    assert entity_id(LANGFJORDJOKELEN) in ids
    assert "NVE-54" in ids
    assert entity_members("NVE-3113+3115+3117") == (3113, 3115, 3117)
    # The fifteen glaciers around Strupbreen include some without a position
    assert not any(g.startswith("NVE-195+") for g in ids)
    # Strupbreen is itself on 1952-2010 and a pair with Koppangsbreen on 1978-1985
    by_period = periods.set_index(["breid", "y0", "y1"]).RGIId
    assert by_period[(200, 1952, 2010)] == "NVE-200"
    assert by_period[(200, 1978, 1985)] == "NVE-200+205"
    assert by_period[(2148, 1966, 1993)].startswith("NVE-2141+")


def test_rates_signs_and_density(periods):
    p = periods.set_index(["RGIId", "y0", "y1"])
    # Harbardsbreen: the total of the table has the wrong sign, the rate is right
    assert p.loc[("NVE-2514", 1966, 1996)].mwe_per_year < 0
    # Gråfjellsbrea 1959-1997, converted at 860 kg m-3
    assert p.loc[("NVE-3127", 1959, 1997)].mwe_per_year == pytest.approx(
        -0.46 * 850 / 860
    )
    assert (periods.cumulative_mwe == periods.mwe_per_year * periods.n_years).all()
    assert (periods.FROM_DATE.dt.strftime("%m-%d") == "10-01").all()
    assert (periods.TO_DATE.dt.strftime("%m-%d") == "10-01").all()
    assert ((periods.survey_to.dt.year - periods.y1).abs() <= 1).all()


def test_default_uncertainty(periods):
    assert (periods.sigma_mwe_per_year > 0).all()
    p = periods.set_index(["RGIId", "y0", "y1"])
    # Storbreen 1968-1984, two photogrammetric maps: 2 m w.e. over 16 years
    assert p.loc[("NVE-2636", 1968, 1984)].sigma_mwe_per_year == pytest.approx(2 / 16)
    assert p.loc[("NVE-1094", 1968, 2001)].sigma_mwe_per_year == pytest.approx(0.064)


def test_agrees_with_andreassen16(periods):
    # Recomputed after the paper, once a date error was found in the laser scan of
    # 2010 (comment of the table)
    recomputed = {("NVE-2078", 1997, 2010), ("NVE-2085", 1997, 2010)}
    nve_of = {
        code: f"NVE-{breid}" for code, (_, _, breid, _) in ANDREASSEN16_GLACIERS.items()
    }
    p = periods.set_index(["RGIId", "y0", "y1"])
    compared = 0
    for r in andreassen16_periods().itertuples():
        key = (nve_of[r.RGIId], r.y0, r.y1)
        if key in p.index and key not in recomputed:
            assert p.loc[key].mwe_per_year == pytest.approx(
                r.mwe_per_year, abs=0.015
            ), key
            compared += 1
    assert compared >= 15


def test_pre_2000_target():
    target = geodetic_target_NVE(max_year=2002)
    assert len(target) == 40 and target.RGIId.nunique() == 32
    assert target.y1.max() <= 2002
    # Divides of 1995 that do not match RGI 6.2 on Hardangerjøkulen
    assert not {"NVE-2971", "NVE-2972"} & set(target.RGIId)
    assert {"NVE-2971", "NVE-2972"} <= set(
        geodetic_target_NVE(max_year=2002, max_area_ratio=None).RGIId
    )
    for _, g in target.groupby("RGIId"):
        g = g.sort_values("FROM_DATE")
        assert (g.TO_DATE.to_numpy()[:-1] <= g.FROM_DATE.to_numpy()[1:]).all()
    # Engabreen: the longer of its two overlapping periods
    eng = target[target.RGIId == "NVE-1094"]
    assert list(zip(eng.y0, eng.y1)) == [(1968, 2001)]
    assert not any("2478" in g for g in geodetic_target_NVE().RGIId)


def test_crosswalk_and_outlines():
    table = table_RGI62_to_NVE()
    assert (table.frac_rgi == 1).all()
    assert table_RGI62_to_NVE(region_id=6).empty
    # The RGI ids of Andreassen16 are those of the positions of NVE
    rgi_of = table[~table.custom_id.str.contains(r"\+")].set_index("custom_id").RGIId
    for _, rgi_id, breid, _ in ANDREASSEN16_GLACIERS.values():
        assert rgi_of[f"NVE-{breid}"] == rgi_id
    # Langfjordjøkelen lies in its own entity and in the group of its outlets
    lan = table[table.RGIId == "RGI60-08.01258"]
    assert set(lan.custom_id) == {"NVE-54", entity_id(LANGFJORDJOKELEN)}

    group = entity_id(LANGFJORDJOKELEN)
    outlines = load_nve_outlines([group, "NVE-54"]).set_index("NVE_ID")
    assert len(outlines.loc[group].rgi_ids) == len(LANGFJORDJOKELEN)
    assert outlines.loc[group].area_rgi == pytest.approx(
        table[table.custom_id == group].area_rgi.sum()
    )
    assert outlines.loc[group].geometry.area > outlines.loc["NVE-54"].geometry.area


def test_nested_entities_map_to_the_one_with_a_target():
    from dataloader.GeoDataLoader import buildGlacierMapping

    group = entity_id(LANGFJORDJOKELEN)
    rgi_ids = ["RGI60-08.01258", "RGI60-08.01259"]
    # 52 only has a period as part of the group, 54 one of its own as well
    assert buildGlacierMapping(rgi_ids, "NVE") == {"RGI60-08.01259": group}
    target = geodetic_target_NVE(max_year=2002)
    assert buildGlacierMapping(rgi_ids, "NVE", idsWithTarget=set(target.RGIId)) == {
        rgi_id: group for rgi_id in rgi_ids
    }


def test_named_periods():
    periods = {"NVE-2078": [[1968, 1988]], "NVE-2971": [[1961, 1995]]}
    target = geodetic_target_NVE(periods=periods)
    assert list(zip(target.RGIId, target.y0, target.y1)) == [
        ("NVE-2078", 1968, 1988),
        ("NVE-2971", 1961, 1995),  # named: kept whatever its area ratio
    ]
    with pytest.raises(AssertionError):
        geodetic_target_NVE(periods={"NVE-2078": [[1968, 1990]]})
    with pytest.raises(AssertionError):
        geodetic_target_NVE(periods=periods, max_year=2000)
