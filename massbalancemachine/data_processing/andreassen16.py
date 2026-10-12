"""
Geodetic mass balance of 10 Norwegian glaciers from Andreassen et al. (2016), gridded on
their RGI 6.2 outlines.

Andreassen, L. M., Elvehøy, H., Kjøllmoen, B. and Engeset, R. V.: Reanalysis of
long-term series of glaciological and geodetic mass balance for 10 Norwegian glaciers,
The Cryosphere, 10, 535-552, doi:10.5194/tc-10-535-2016, 2016.

Table 4 of the paper gives 21 geodetic balances between successive surveys of the 10
glaciers with long-term mass-balance programmes of NVE, from 1962 to 2013, in m w.e.
per year. They are kept as constants in `ANDREASSEN16_PERIODS`: the supplement only
holds the glaciological series of Nigardsbreen. Before 2001 the surveys are aerial
photogrammetry, from 2001 on airborne laser scanning (Table 2).

**Dates.** The geodetic balances of Table 4 are corrected for the ablation and
accumulation between the dates of the geodetic surveys (Table 2, `SURVEY_DATES`) and those of the glaciological ones (Sect. 3.3.2), so
they refer to whole mass-balance years: a period between the surveys of `y0` and `y1`
covers the hydrological years `y0 + 1` to `y1`, from the 1st of October of `y0` to the
1st of October of `y1`, and Table 4 divides by `y1 - y0` years. For Engabreen and
Rembesdalskåka the first survey (1968 and 1961) is one year before the first
glaciological year, and Table 4 shifts the period to that year: 1969-2001 and
1962-1995. Those shifted periods are the ones kept. The WGMS copy of these balances
(`change.csv`) is not, since it gives the elevation change between the survey dates.

**Internal balance.** The geodetic balance is the total one: it includes the internal
and basal ablation by dissipation of energy, which the paper estimates separately
(`B.int`, kept as `mwe_per_year_internal`) and subtracts before comparing with the
glaciological balance. It is small, 0.01-0.08 m w.e. per year, except on Nigardsbreen
and Engabreen (0.15-0.16).

**Uncertainty.** Table 4 splits the random error of the geodetic balance into the one
of the DTMs and the one of the density conversion. They are one sigma, and added in
quadrature: this is how the reduced discrepancies of Table 5 are reproduced
(Nigardsbreen 1984-2013, 2.81), and how the totals of Sect. 3.4.3 follow from their
components.

**Geometry.** The paper refers each balance to the mean area of the two surveys, on
the drainage basins NVE delineated from the latest laser scanning (Eq. 4), but does
not publish them. NVE's map services hold the outlines of its mass-balance glaciers,
from the inventory of 2018/19, and an older one from maps of 1947-1985 that does not
divide the ice caps into outlets. Every period is therefore gridded on the RGI 6.2
outlines, from NVE's inventory of Landsat scenes of 1999 to 2006, between the surveys
of most periods: one outline per glacier, with the same ice divides. Their areas
agree with the mean areas of the periods (`area_mean_km2`, from WGMS) to within 10 %,
except Nigardsbreen: 41.9 km² against 47.7-48.6, since the paper uses the
hydrological basin draining to Nigardsbrevatn, which includes fringes that do not
flow to the outlet (Fig. 4).

**Calving.** Austdalsbreen calves into a regulated lake, and its geodetic balance
includes the calving loss, 0.30 m w.e. per year on average (Sect. 4.1), which a model
of the surface balance cannot reproduce. It is left out of the targets by default, see
`CALVING_GLACIERS` and `geodetic_target_Andreassen16(exclude_calving=...)`.

Glaciers are identified by the abbreviation of Table 4 in ASCII ("GRA" for "Grå"), as
for Hagg12 and Belart20, so the gridded products of this source carry
`RGIId == "NIG"`, and the crosswalk built by `table_RGI62_to_Andreassen16` maps one RGI
id to each glacier.

The paper converts volume changes to water equivalent with a density of
850 kg m-3, the one of this workflow, so no conversion is needed.
"""

import geopandas as gpd
import numpy as np
import pandas as pd

from data_processing.custom_outlines import CustomOutlineSpec
from data_processing.glacier_utils import get_region_shape_file
from data_processing.utils.periods import select_dem_periods

# Scandinavia is RGI region 08. The glaciers lie in its three second-order regions,
# read glacier by glacier from the RGI: 08-01 (N Scandinavia) for Engabreen and
# Langfjordjøkelen, 08-03 (SE) for Storbreen, Hellstugubreen and Gråsubreen in
# Jotunheimen, 08-02 (SW) for the five others.
NORWAY_REGION_ID = 8
# Year of the RGI 6.2 outlines of these glaciers (1999 to 2006), for OGGM
OUTLINES_YEAR = 2003

# Name, RGI 6.2 outline, NVE id of the latest inventory and area of the latest survey
# (Table 1, km²) of every glacier, keyed by the abbreviation of Table 4 in ASCII
ANDREASSEN16_GLACIERS = {
    "ALF": ("Ålfotbreen", "RGI60-08.02666", 2078, 4.5),
    "HAN": ("Hansebreen", "RGI60-08.02650", 2085, 3.1),
    "NIG": ("Nigardsbreen", "RGI60-08.01126", 2297, 46.6),
    "AUS": ("Austdalsbreen", "RGI60-08.01286", 2478, 10.6),
    "REM": ("Rembesdalskåka", "RGI60-08.01779", 2968, 17.3),
    "STO": ("Storbreen", "RGI60-08.00312", 2636, 5.1),
    "HEL": ("Hellstugubreen", "RGI60-08.00449", 2768, 2.9),
    "GRA": ("Gråsubreen", "RGI60-08.00987", 2743, 2.1),
    "ENG": ("Engabreen", "RGI60-08.01657", 1094, 36.8),
    "LAN": ("Langfjordjøkelen", "RGI60-08.01258", 54, 3.2),
}

# Table 2: date of every geodetic survey, keyed by glacier and year
SURVEY_DATES = {
    "ALF": ("1968-08-05", "1988-09-07", "1997-08-14", "2010-09-02"),
    "HAN": ("1968-08-05", "1988-09-07", "1997-08-14", "2010-09-02"),
    "NIG": ("1964-09-02", "1984-08-10", "2009-10-17", "2013-09-10"),
    "AUS": ("1988-08-10", "2009-10-17"),
    "REM": ("1961-08-31", "1995-08-31", "2010-09-30"),
    "STO": ("1968-08-27", "1984-08-24", "1997-08-08", "2009-10-17"),
    "HEL": ("1968-08-27", "1980-09-26", "1997-08-08", "2009-10-17"),
    "GRA": ("1984-08-23", "1997-08-08", "2009-10-17"),
    "ENG": ("1968-08-25", "2001-09-24", "2008-09-02"),
    "LAN": ("1966-07-11", "1994-08-01", "2008-09-02"),
}
# First periods of Table 4 shifted to the first glaciological year: (glacier, y0) to the
# year of their first survey
SHIFTED_PERIODS = {("REM", 1962): 1961, ("ENG", 1969): 1968}

# Table 4: (glacier, first mass-balance year minus one, last mass-balance year,
# geodetic balance in m w.e. per year, sigma of the DTMs, sigma of the density
# conversion, internal balance, its sigma, mean area of the survey period in km² from
# WGMS)
ANDREASSEN16_PERIODS = [
    ("ALF", 1968, 1988, -0.12, 0.09, 0.01, -0.06, 0.02, 4.330),
    ("ALF", 1988, 1997, 0.87, 0.08, 0.05, -0.06, 0.02, 4.327),
    ("ALF", 1997, 2010, -1.05, 0.04, 0.06, -0.06, 0.02, 4.226),
    ("HAN", 1988, 1997, 0.61, 0.05, 0.04, -0.04, 0.01, 3.124),
    ("HAN", 1997, 2010, -1.34, 0.06, 0.08, -0.04, 0.01, 2.966),
    ("NIG", 1964, 1984, 0.14, 0.16, 0.01, -0.16, 0.05, 48.585),
    ("NIG", 1984, 2013, -0.16, 0.08, 0.01, -0.16, 0.05, 47.735),
    ("AUS", 1988, 2009, -0.32, 0.05, 0.02, -0.03, 0.01, 10.963),
    ("REM", 1962, 1995, 0.19, 0.06, 0.01, -0.06, 0.02, 17.628),
    ("REM", 1995, 2010, -0.73, 0.05, 0.04, -0.06, 0.02, 17.451),
    ("STO", 1968, 1984, -0.31, 0.08, 0.02, -0.02, 0.01, 5.475),
    ("STO", 1984, 1997, 0.19, 0.15, 0.01, -0.02, 0.01, 5.351),
    ("STO", 1997, 2009, -0.45, 0.13, 0.03, -0.02, 0.01, 5.248),
    ("HEL", 1968, 1980, -0.38, 0.20, 0.02, -0.02, 0.01, 3.201),
    ("HEL", 1980, 1997, -0.08, 0.07, 0.01, -0.02, 0.01, 3.045),
    ("HEL", 1997, 2009, -0.51, 0.06, 0.03, -0.02, 0.01, 2.977),
    ("GRA", 1984, 1997, -0.06, 0.10, 0.00, -0.01, 0.00, 2.251),
    ("GRA", 1997, 2009, -0.44, 0.09, 0.03, -0.01, 0.00, 2.185),
    ("ENG", 1969, 2001, -0.03, 0.06, 0.00, -0.15, 0.05, 37.394),
    ("ENG", 2001, 2008, -0.48, 0.04, 0.03, -0.08, 0.03, 37.050),
    ("LAN", 1994, 2008, -1.18, 0.13, 0.07, -0.04, 0.01, 3.415),
]

# Glaciers losing mass by calving (Sect. 2): Austdalsbreen ends in a hydropower lake
CALVING_GLACIERS = {"AUS"}

# Density of the paper, the one of this workflow: no conversion is needed
ICE_DENSITY = 850.0


def _survey_date(glacier_id: str, year: int):
    """Date of the survey of `year` of Table 2."""
    (date,) = [d for d in SURVEY_DATES[glacier_id] if d.startswith(f"{year}-")]
    return pd.Timestamp(date)


def andreassen16_periods():
    """Every period of `ANDREASSEN16_PERIODS` with its dates, rate and uncertainty, one
    row per period.

    Columns: RGIId (glacier code), name, y0, y1, n_years, FROM_DATE and TO_DATE (the
    1st of October of y0 and y1), survey_from and survey_to (the dates of the geodetic
    surveys), mwe_per_year, cumulative_mwe, sigma_mwe_per_year (the errors of the DTMs
    and of the density conversion in quadrature), sigma_dtm, sigma_density,
    mwe_per_year_internal, sigma_internal and area_mean_km2 (the mean area of the
    survey period, from WGMS).
    """
    df = pd.DataFrame(
        ANDREASSEN16_PERIODS,
        columns=[
            "RGIId",
            "y0",
            "y1",
            "mwe_per_year",
            "sigma_dtm",
            "sigma_density",
            "mwe_per_year_internal",
            "sigma_internal",
            "area_mean_km2",
        ],
    )
    df.insert(1, "name", df.RGIId.map(lambda g: ANDREASSEN16_GLACIERS[g][0]))
    df["survey_from"] = [
        _survey_date(r.RGIId, SHIFTED_PERIODS.get((r.RGIId, r.y0), r.y0))
        for r in df.itertuples()
    ]
    df["survey_to"] = [_survey_date(r.RGIId, r.y1) for r in df.itertuples()]
    df["n_years"] = df.y1 - df.y0
    df["FROM_DATE"] = pd.to_datetime(df.y0.astype(str) + "-10-01")
    df["TO_DATE"] = pd.to_datetime(df.y1.astype(str) + "-10-01")
    df["cumulative_mwe"] = df.mwe_per_year * df.n_years
    df["sigma_mwe_per_year"] = np.hypot(df.sigma_dtm, df.sigma_density)
    return df[
        [
            "RGIId",
            "name",
            "y0",
            "y1",
            "n_years",
            "FROM_DATE",
            "TO_DATE",
            "survey_from",
            "survey_to",
            "mwe_per_year",
            "cumulative_mwe",
            "sigma_mwe_per_year",
            "sigma_dtm",
            "sigma_density",
            "mwe_per_year_internal",
            "sigma_internal",
            "area_mean_km2",
        ]
    ]


def geodetic_target_Andreassen16(
    max_year: int = None,
    min_year: int = None,
    min_period_years: int = 1,
    glacier_ids_to_keep=None,
    allowed_years=None,
    max_outside_years: int = 0,
    multi_period: bool = True,
    tie_break: str = "sigma",
    exclude_calving: bool = True,
):
    """Geodetic targets of Andreassen et al. (2016), one row per period between two
    surveys.

    Returns a dataframe shaped like the one `data_processing.glamos.geodetic_target_GLAMOS`
    returns, so that both drive the same code path in `GeoDataLoader`:

        RGIId                 the glacier code, e.g. "NIG"
        FROM_DATE, TO_DATE    1st of October of the first and last mass-balance years
        mwe_per_year          the geodetic balance of Table 4, m w.e. per year
        cumulative_mwe        that rate times the number of years
        sigma_mwe_per_year    its uncertainty, one sigma

    The periods are selected by `utils.periods.select_dem_periods`, which documents
    the arguments: every eligible period by default, or with `multi_period` off the
    longest one of each glacier, ties going to the lowest sigma. The periods of a
    glacier follow each other and never overlap.

    With `exclude_calving`, the default, the glaciers of `CALVING_GLACIERS` are left
    out, since their geodetic balance includes a calving loss.
    """
    df = andreassen16_periods()
    if exclude_calving:
        df = df[~df.RGIId.isin(CALVING_GLACIERS)]
    return select_dem_periods(
        df,
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
    """The RGI 6.2 outlines of the glaciers, one row per glacier, with the glacier code
    in a column named "ANDREASSEN16_ID"."""
    of_rgi = {rgi_id: g for g, (_, rgi_id, _, _) in ANDREASSEN16_GLACIERS.items()}
    rgi = gpd.read_file(get_region_shape_file(f"{NORWAY_REGION_ID:02d}"))
    rgi = rgi[rgi.RGIId.isin(of_rgi)].copy()
    missing = set(of_rgi).difference(rgi.RGIId)
    assert not missing, f"RGI 6.2 has no outline {sorted(missing)}."
    rgi["ANDREASSEN16_ID"] = rgi.RGIId.map(of_rgi)
    return rgi


def load_andreassen16_outlines(glacier_ids_to_keep=None):
    """Read the RGI 6.2 outlines of the Norwegian glaciers, one entity per glacier.

    The returned frame is in EPSG:4326 with the glacier code in a column named
    "ANDREASSEN16_ID", ready for `custom_outlines.build_custom_gdirs`, along with the
    RGI first- and second-order regions of the glacier ("08", "02"), the name, the RGI
    id, the area of the outline in km² and the area of the latest survey of Table 1.
    """
    outlines = _rgi_outlines().rename(columns={"Area": "area_rgi"})
    outlines["o1_region"] = outlines.O1Region.astype(int).map("{:02d}".format)
    outlines["o2_region"] = outlines.O2Region.astype(int).map("{:02d}".format)
    outlines["name"] = outlines.ANDREASSEN16_ID.map(
        lambda g: ANDREASSEN16_GLACIERS[g][0]
    )
    outlines["area_latest"] = outlines.ANDREASSEN16_ID.map(
        lambda g: ANDREASSEN16_GLACIERS[g][3]
    )
    if glacier_ids_to_keep is not None:
        missing = set(glacier_ids_to_keep).difference(outlines.ANDREASSEN16_ID)
        assert not missing, f"Andreassen16 has no glacier {sorted(missing)}."
        outlines = outlines[outlines.ANDREASSEN16_ID.isin(glacier_ids_to_keep)]
    return (
        outlines[
            [
                "ANDREASSEN16_ID",
                "o1_region",
                "o2_region",
                "name",
                "RGIId",
                "area_rgi",
                "area_latest",
                "geometry",
            ]
        ]
        .sort_values("ANDREASSEN16_ID")
        .reset_index(drop=True)
        .to_crs("EPSG:4326")
    )


def andreassen16_outline_spec(dem_source: str = "COPDEM30", **overrides):
    """Description of the Norwegian outlines for the generic custom-outline machinery.

    The DEM defaults to the Copernicus GLO-30 DEM, 30 m, acquired in 2011-2015: SRTM,
    the February 2000 surface GLAMOS and Maurer19 are gridded on, stops at 60°N, and
    the glaciers lie between 60.5°N and 70.1°N. The second-order region is left to the
    `o2_region` column of the outlines, since the glaciers span three of them.
    """
    spec = dict(
        name="Andreassen16",
        id_column="ANDREASSEN16_ID",
        o1_region=f"{NORWAY_REGION_ID:02d}",
        o2_region=None,
        src_date=f"{OUTLINES_YEAR}-01-01 00:00:00",
        bgndate=f"{OUTLINES_YEAR}0101",
        dem_source=dem_source,
    )
    spec.update(overrides)
    return CustomOutlineSpec(**spec)


def table_RGI62_to_Andreassen16(region_id=NORWAY_REGION_ID):
    """Crosswalk from RGI 6.2 ids to the Norwegian glaciers.

    The outlines of this source are the RGI 6.2 ones, so the crosswalk is read off
    `ANDREASSEN16_GLACIERS` rather than matched by overlap, one RGI outline per
    glacier. Only region 08 holds any.

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
    if int(region_id) != NORWAY_REGION_ID:
        return pd.DataFrame(columns=columns)
    rgi = _rgi_outlines().rename(
        columns={"ANDREASSEN16_ID": "custom_id", "Area": "area_rgi"}
    )
    rgi["area_custom"] = rgi.area_rgi
    rgi["frac_rgi"] = 1.0
    rgi["frac_custom"] = 1.0
    return pd.DataFrame(rgi[columns]).sort_values("RGIId").reset_index(drop=True)
