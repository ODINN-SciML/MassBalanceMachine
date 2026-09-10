"""Tests of the source-agnostic helpers used by the custom outline datasets (GLAMOS,
Rabatel16): the crosswalk from RGI 6.2 ids by area overlap, and the check that a
user-supplied DEM holds elevations.

None of them needs downloaded data: the outlines and the rasters are synthetic.
"""

import numpy as np
import geopandas as gpd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from data_processing.custom_outlines import match_rgi62_by_overlap
from data_processing.oggm_utils import check_elevation_raster

UTM32N = "EPSG:32632"


def _outlines(boxes, id_column):
    """A frame of rectangular outlines from {id: (xmin, ymin, xmax, ymax)} in metres."""
    return gpd.GeoDataFrame(
        {id_column: list(boxes)},
        geometry=[box(*b) for b in boxes.values()],
        crs=UTM32N,
    )


def test_overlap_maps_several_rgi_ids_onto_one_entity():
    custom = _outlines({"G1": (0, 0, 2000, 1000), "G2": (3000, 0, 4000, 1000)}, "ID")
    rgi = _outlines(
        {
            # the two halves of G1, which split after the custom inventory
            "RGI60-11.00001": (0, 0, 1000, 1000),
            "RGI60-11.00002": (1000, 0, 2000, 1000),
            # mostly inside G2, and slightly overlapping the gap next to it
            "RGI60-11.00003": (2900, 0, 4000, 1000),
            # only a sliver of it lies on G2: no match at all
            "RGI60-11.00004": (3900, 0, 6000, 1000),
        },
        "RGIId",
    ).to_crs("EPSG:4326")

    matches = match_rgi62_by_overlap(custom, "ID", rgi).set_index("RGIId")

    assert matches.custom_id.to_dict() == {
        "RGI60-11.00001": "G1",
        "RGI60-11.00002": "G1",
        "RGI60-11.00003": "G2",
    }
    assert matches.loc["RGI60-11.00001", "frac_rgi"] == pytest.approx(1, abs=1e-3)
    assert matches.loc["RGI60-11.00001", "frac_custom"] == pytest.approx(0.5, abs=1e-3)
    assert matches.loc["RGI60-11.00003", "frac_rgi"] == pytest.approx(1 / 1.1, abs=1e-3)
    assert matches.loc["RGI60-11.00001", "area_custom"] == pytest.approx(2, abs=1e-3)


def test_overlap_needs_projected_outlines():
    custom = _outlines({"G1": (0, 0, 1000, 1000)}, "ID").to_crs("EPSG:4326")
    with pytest.raises(AssertionError):
        match_rgi62_by_overlap(custom, "ID", custom.rename(columns={"ID": "RGIId"}))


def _write_raster(path, bands, dtype, colorinterp=None):
    data = np.zeros((len(bands), 4, 4), dtype=dtype)
    for i, value in enumerate(bands):
        data[i] = value
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=len(bands),
        dtype=dtype,
        crs=UTM32N,
        transform=from_origin(300000, 5000000, 50, 50),
    ) as dst:
        dst.write(data)
        if colorinterp is not None:
            dst.colorinterp = colorinterp
    return str(path)


def test_elevation_raster_accepts_a_single_band_of_elevations(tmp_path):
    check_elevation_raster(_write_raster(tmp_path / "dem.tif", [2500.0], "float32"))
    check_elevation_raster(_write_raster(tmp_path / "dem_int.tif", [2500], "int16"))


def test_elevation_raster_refuses_a_rendered_image(tmp_path):
    ci = rasterio.enums.ColorInterp
    rendered = _write_raster(
        tmp_path / "rendered.tif",
        [180, 180, 180, 255],
        "uint8",
        colorinterp=[ci.red, ci.green, ci.blue, ci.alpha],
    )
    with pytest.raises(AssertionError, match="rendered image"):
        check_elevation_raster(rendered)

    grey = _write_raster(tmp_path / "grey.tif", [180], "uint8")
    with pytest.raises(AssertionError, match="rendered image"):
        check_elevation_raster(grey)
