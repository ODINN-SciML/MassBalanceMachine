"""
Geodetic mass balance of the Bavarian glaciers from Hagg et al. (2012), gridded on
their RGI 6.2 outlines.

Hagg, W., Mayer, C., Mayr, E. and Heilig, A.: Climate and glacier fluctuations in the
Bavarian Alps in the past 120 years, Erdkunde, 66(2), 121-142,
doi:10.3112/erdkunde.2012.02.03, 2012.

Tables 4 to 7 of the paper give the mean annual geodetic balance of the five Bavarian
glaciers between successive surveys, from 1889 to 2009, in cm w.e. per year. They are
kept as constants in `HAGG12_PERIODS`: the paper publishes no data file. The glaciers
are Nördlicher and Südlicher Schneeferner and Höllentalferner in the Wetterstein, and
Blaueis and Watzmanngletscher in the Berchtesgaden Alps. None of them carries stakes in
WGMS, so they only bring geodetic targets.

**Geometry.** The paper divides a volume change by the mean glacier area between the
two surveys (Sect. 3.2.3), but no outline before 1999 is published. The RGI 6.2
outlines of these glaciers date from the last survey of the paper (`BgnDate` 2009 and
2010) and match the areas of its Table 2, except Nördlicher Schneeferner (0.351 km²
against 0.278). Every period is therefore gridded on the RGI 6.2 outlines, and the
glaciers were larger when most periods were measured. Table 2 gives the area at every
survey (`HAGG12_AREAS_HA`), and `hagg12_periods` reports the mean area a rate refers
to (`area_mean_km2`), to compare with `area_rgi` of `load_hagg12_outlines`. Over the
periods of at least 10 years between 1950 and 2000 the outline holds 87-94 % of that
area for Nördlicher Schneeferner, 78-85 % for Höllentalferner, 52-58 % for Blaueis,
28-42 % for Watzmanngletscher and 22-25 % for Südlicher Schneeferner. The rates are specific
balances, m w.e. per year, and the glaciers are small and fairly flat, so the smaller
outline changes the elevation range the model sees more than the rate itself.

Glaciers are identified by a short code of their own, `HAGG12_GLACIERS`, since two of
them are split into two RGI 6.2 outlines (Südlicher Schneeferner and Blaueis). As for
GLAMOS, the gridded products of this source carry `RGIId == "NSF"`, and the crosswalk
built by `table_RGI62_to_Hagg12` is many RGI ids to one glacier.

The paper converts volume changes to water equivalent with an ice density of
900 kg m-3, while this workflow uses 850 kg m-3. The balances and uncertainties are
therefore multiplied by `ICE_DENSITY_FACTOR` = 850 / 900, as for Rabatel16.

A period runs over whole hydrological years: the surveys are end-of-summer ones, and
a period between the surveys of `y0` and `y1` is taken to cover the hydrological
years `y0 + 1` to `y1`, from the 1st of October of `y0` to the 1st of October of `y1`.
"""

import geopandas as gpd
import pandas as pd

from data_processing.custom_outlines import CustomOutlineSpec
from data_processing.glacier_utils import get_region_shape_file
from data_processing.utils.periods import select_dem_periods

# The Bavarian Alps are entirely inside RGI region 11 (Central Europe), second-order
# region 11-01 (Alps).
BAVARIA_REGION_ID = 11
BAVARIA_SUBREGION = "01"

# Name, RGI 6.2 outlines and year of the last survey of the paper (Table 2), which is
# the date of those outlines
HAGG12_GLACIERS = {
    "NSF": ("Nördlicher Schneeferner", ["RGI60-11.03246"], 2009),
    "SSF": ("Südlicher Schneeferner", ["RGI60-11.03247", "RGI60-11.03248"], 2009),
    "HTF": ("Höllentalferner", ["RGI60-11.03249"], 2010),
    "BLA": ("Blaueis", ["RGI60-11.03250", "RGI60-11.03251"], 2009),
    "WMG": ("Watzmanngletscher", ["RGI60-11.03252"], 2009),
}
# Year of the outlines, for OGGM
OUTLINES_YEAR = 2009

# Tables 4 to 7: (glacier, year of the first survey, year of the second survey, mean
# balance in cm w.e. per year at an ice density of 900 kg m-3). Table 6 dates the last
# survey of Höllentalferner to 2009 although Table 2 and Sect. 3.2.2 date it to October
# 2010; the period is kept as published.
HAGG12_PERIODS = [
    ("NSF", 1892, 1949, -44),
    ("NSF", 1949, 1959, -43),
    ("NSF", 1959, 1969, -19),
    ("NSF", 1969, 1979, 14),
    ("NSF", 1979, 1990, -28),
    ("NSF", 1990, 1999, -65),
    ("NSF", 1999, 2006, -78),
    ("NSF", 2006, 2009, -89),
    ("SSF", 1892, 1949, -42),
    ("SSF", 1949, 1959, -48),
    ("SSF", 1959, 1971, -15),
    ("SSF", 1971, 1979, 27),
    ("SSF", 1979, 1990, -36),
    ("SSF", 1990, 1999, -27),
    ("SSF", 1999, 2006, -42),
    ("SSF", 2006, 2009, -75),
    ("HTF", 1950, 1959, -13),
    ("HTF", 1959, 1970, 23),
    ("HTF", 1970, 1981, 40),
    ("HTF", 1981, 1989, -41),
    ("HTF", 1989, 1999, -53),
    ("HTF", 1999, 2006, -86),
    ("HTF", 2006, 2009, -68),
    ("BLA", 1889, 1924, 2),
    ("BLA", 1924, 1949, -68),
    ("BLA", 1949, 1959, -38),
    ("BLA", 1959, 1970, -26),
    ("BLA", 1970, 1980, 34),
    ("BLA", 1980, 1989, -68),
    ("BLA", 1989, 1999, -36),
    ("BLA", 1999, 2009, -21),
    ("WMG", 1897, 1959, -23),
    ("WMG", 1959, 1970, 29),
    ("WMG", 1970, 1980, 47),
    ("WMG", 1980, 1989, -31),
    ("WMG", 1989, 1999, -38),
    ("WMG", 1999, 2009, -51),
]

# Table 2: area in ha at every survey. Blaueis and Watzmanngletscher have none for 1999.
HAGG12_AREAS_HA = {
    "NSF": {
        2009: 27.8,
        2006: 30.7,
        1999: 36.0,
        1990: 33.5,
        1979: 40.9,
        1969: 39.7,
        1959: 36.4,
        1949: 37.9,
        1892: 103.6,
    },
    "SSF": {
        2009: 4.8,
        2006: 8.4,
        1999: 11.6,
        1990: 12.3,
        1979: 31.4,
        1971: 20.5,
        1959: 19.4,
        1949: 27.0,
        1892: 85.5,
    },
    "HTF": {
        2010: 22.3,
        2006: 24.7,
        1999: 25.7,
        1989: 29.8,
        1981: 30.2,
        1970: 26.7,
        1959: 25.7,
        1950: 27.1,
    },
    "BLA": {
        2009: 7.5,
        2006: 11.0,
        1989: 12.3,
        1980: 16.4,
        1970: 12.6,
        1959: 13.1,
        1949: 15.2,
        1924: 20.2,
        1889: 16.4,
    },
    "WMG": {
        2009: 5.6,
        2006: 10.1,
        1989: 18.1,
        1980: 24.0,
        1970: 17.7,
        1959: 10.0,
        1897: 27.9,
    },
}

# Ice density of the paper and of this workflow, see the module docstring
HAGG12_ICE_DENSITY = 900.0
ICE_DENSITY = 850.0
ICE_DENSITY_FACTOR = ICE_DENSITY / HAGG12_ICE_DENSITY

# Elevation error of one survey, m (Sect. 3.2.1 and 3.2.2): the maps of the Bavarian
# Topographic Office (1889, 1897), 2-5 m and taken at 5; the photogrammetric maps
# (Finsterwalder 1892, Thiersch 1924, and every survey from 1949 to 2006), 1 m; the
# survey of 2009/10 by laser scanning, tachymetry and GPS, 0.7 m. The paper adds the
# errors of two surveys linearly, see `period_sigma_mwe_per_year`.
SURVEY_SIGMA_M = {"map": 5.0, "photogrammetry": 1.0, "ground": 0.7}
MAP_SURVEY_YEARS = (1889, 1897)
GROUND_SURVEY_FIRST_YEAR = 2009


def survey_method(year: int):
    """Method of the survey of `year`, a key of `SURVEY_SIGMA_M`."""
    if year in MAP_SURVEY_YEARS:
        return "map"
    if year >= GROUND_SURVEY_FIRST_YEAR:
        return "ground"
    return "photogrammetry"


def period_sigma_mwe_per_year(n_years: int, start_year: int, end_year: int):
    """Uncertainty of the mean annual balance over a period of `n_years` between the
    surveys of `start_year` and `end_year`, in m w.e. per year.

    It follows the paper, which adds the elevation errors of both surveys linearly and
    spreads them over the period: 6 m over the 35 years of Blaueis 1889-1924, or
    0.17 m per year, 6 m over the 62 years of Watzmanngletscher 1897-1959, 0.10 m per
    year, and 2 m over a decade of photogrammetric surveys, 0.2 m per year. The paper
    gives no uncertainty on the density, so the elevation error is only converted to
    water equivalent with `ICE_DENSITY`: 0.17 m w.e. per year for a decade.
    """
    assert n_years > 0, f"A period of {n_years} years has no mass balance."
    error_m = (
        SURVEY_SIGMA_M[survey_method(start_year)]
        + SURVEY_SIGMA_M[survey_method(end_year)]
    )
    return float(error_m / n_years * ICE_DENSITY / 1000)


def _area_km2(glacier_id: str, year: int):
    """Area of Table 2 at the survey of `year`, km², or NaN when it gives none.
    Table 6 dates the last survey of Höllentalferner to 2009, Table 2 to 2010."""
    areas = HAGG12_AREAS_HA[glacier_id]
    if glacier_id == "HTF" and year == 2009:
        year = 2010
    return areas.get(year, float("nan")) / 100


def hagg12_periods():
    """Every period of `HAGG12_PERIODS` with its dates, rate and uncertainty, one row
    per period.

    Columns: RGIId (glacier code), name, y0, y1, n_years, FROM_DATE, TO_DATE,
    mwe_per_year, cumulative_mwe, sigma_mwe_per_year, and area_mean_km2, the mean of
    the areas of Table 2 at both surveys, which the rate refers to (NaN where Table 2
    lacks one).
    """
    df = pd.DataFrame(HAGG12_PERIODS, columns=["RGIId", "y0", "y1", "cm_we_per_year"])
    df.insert(1, "name", df.RGIId.map(lambda g: HAGG12_GLACIERS[g][0]))
    df["n_years"] = df.y1 - df.y0
    df["FROM_DATE"] = pd.to_datetime(df.y0.astype(str) + "-10-01")
    df["TO_DATE"] = pd.to_datetime(df.y1.astype(str) + "-10-01")
    df["mwe_per_year"] = df.cm_we_per_year / 100 * ICE_DENSITY_FACTOR
    df["cumulative_mwe"] = df.mwe_per_year * df.n_years
    df["sigma_mwe_per_year"] = [
        period_sigma_mwe_per_year(r.n_years, r.y0, r.y1) for r in df.itertuples()
    ]
    df["area_mean_km2"] = [
        (_area_km2(r.RGIId, r.y0) + _area_km2(r.RGIId, r.y1)) / 2
        for r in df.itertuples()
    ]
    return df.drop(columns="cm_we_per_year")


def geodetic_target_Hagg12(
    max_year: int = None,
    min_year: int = None,
    min_period_years: int = 1,
    glacier_ids_to_keep=None,
    allowed_years=None,
    max_outside_years: int = 0,
    multi_period: bool = True,
    tie_break: str = "sigma",
):
    """Geodetic targets of Hagg et al. (2012), one row per period between two surveys.

    Returns a dataframe shaped like the one `data_processing.glamos.geodetic_target_GLAMOS`
    returns, so that both drive the same code path in `GeoDataLoader`:

        RGIId                 the glacier code, e.g. "NSF"
        FROM_DATE, TO_DATE    1st of October of the years of the two surveys
        mwe_per_year          the balance of Tables 4-7, m w.e. per year at 850 kg m-3
        cumulative_mwe        that rate times the number of years
        sigma_mwe_per_year    its uncertainty, see `period_sigma_mwe_per_year`

    The periods are selected by `utils.periods.select_dem_periods`, which documents
    the arguments: every eligible period by default, or with `multi_period` off the
    longest one of each glacier, ties going to the lowest sigma.
    """
    return select_dem_periods(
        hagg12_periods(),
        max_year=max_year,
        min_year=min_year,
        min_period_years=min_period_years,
        glacier_ids_to_keep=glacier_ids_to_keep,
        allowed_years=allowed_years,
        max_outside_years=max_outside_years,
        multi_period=multi_period,
        tie_break=tie_break,
    )


def _rgi_outlines():
    """The RGI 6.2 outlines of the glaciers, one row per RGI id, with the glacier code
    in a column named "HAGG12_ID"."""
    of_rgi = {r: g for g, (_, rgi_ids, _) in HAGG12_GLACIERS.items() for r in rgi_ids}
    rgi = gpd.read_file(get_region_shape_file(f"{BAVARIA_REGION_ID:02d}"))
    rgi = rgi[rgi.RGIId.isin(of_rgi)].copy()
    missing = set(of_rgi).difference(rgi.RGIId)
    assert not missing, f"RGI 6.2 has no outline {sorted(missing)}."
    rgi["HAGG12_ID"] = rgi.RGIId.map(of_rgi)
    return rgi


def load_hagg12_outlines(glacier_ids_to_keep=None):
    """Read the RGI 6.2 outlines of the Bavarian glaciers, one entity per glacier.

    Südlicher Schneeferner and Blaueis are two RGI outlines each, dissolved here into
    one entity. The returned frame is in EPSG:4326 with the glacier code in a column
    named "HAGG12_ID", ready for `custom_outlines.build_custom_gdirs`, along with the
    name and the area of the entity in km², summed from the RGI areas.
    """
    rgi = _rgi_outlines()
    outlines = rgi.dissolve("HAGG12_ID", aggfunc={"Area": "sum"}).reset_index()
    outlines = outlines.rename(columns={"Area": "area_rgi"})
    outlines["name"] = outlines.HAGG12_ID.map(lambda g: HAGG12_GLACIERS[g][0])
    if glacier_ids_to_keep is not None:
        missing = set(glacier_ids_to_keep).difference(outlines.HAGG12_ID)
        assert not missing, f"Hagg12 has no glacier {sorted(missing)}."
        outlines = outlines[outlines.HAGG12_ID.isin(glacier_ids_to_keep)]
    return outlines[["HAGG12_ID", "name", "area_rgi", "geometry"]].to_crs("EPSG:4326")


def hagg12_outline_spec(dem_source: str = "COPDEM30", **overrides):
    """Description of the Bavarian outlines for the generic custom-outline machinery.

    The DEM defaults to the Copernicus GLO-30 DEM, 30 m, rather than the 90 m SRTM
    GLAMOS and Fischer11 are gridded on: Südlicher Schneeferner is 0.05 km², about six
    cells of 90 m. Its acquisitions, 2011-2015, are also the closest to the date of the
    outlines, 2009/10. NASADEM, the 30 m DEM of February 2000, cannot be used: OGGM
    1.6.3 downloads it from the USGS data pool (e4ftl01.cr.usgs.gov), which no longer
    serves it, and reads the 404 as a tile without land.
    """
    spec = dict(
        name="Hagg12",
        id_column="HAGG12_ID",
        o1_region=f"{BAVARIA_REGION_ID:02d}",
        o2_region=BAVARIA_SUBREGION,
        src_date=f"{OUTLINES_YEAR}-01-01 00:00:00",
        bgndate=f"{OUTLINES_YEAR}0101",
        dem_source=dem_source,
    )
    spec.update(overrides)
    return CustomOutlineSpec(**spec)


def table_RGI62_to_Hagg12(region_id=BAVARIA_REGION_ID):
    """Crosswalk from RGI 6.2 ids to the Bavarian glaciers.

    The outlines of this source are the RGI 6.2 ones, so the crosswalk is read off
    `HAGG12_GLACIERS` rather than matched by overlap: every RGI outline lies entirely in
    its glacier. Only region 11 holds any.

    Returns a dataframe with columns RGIId, custom_id (the glacier code), area_rgi,
    area_custom, frac_rgi, frac_custom, `custom_id` matching the column name
    `table_RGI62_to_GLAMOS` uses, so `GeoDataLoader` reads both the same way.
    """
    columns = [
        "RGIId",
        "custom_id",
        "area_rgi",
        "area_custom",
        "frac_rgi",
        "frac_custom",
    ]
    if int(region_id) != BAVARIA_REGION_ID:
        return pd.DataFrame(columns=columns)
    rgi = _rgi_outlines().rename(columns={"HAGG12_ID": "custom_id", "Area": "area_rgi"})
    rgi["area_custom"] = rgi.groupby("custom_id").area_rgi.transform("sum")
    rgi["frac_rgi"] = 1.0
    rgi["frac_custom"] = rgi.area_rgi / rgi.area_custom
    return pd.DataFrame(rgi[columns]).sort_values("RGIId").reset_index(drop=True)
