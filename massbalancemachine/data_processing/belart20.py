"""
Geodetic mass balance of 14 Icelandic glaciers from Belart et al. (2020), gridded on
their RGI 6.2 outlines.

Belart, J. M. C., Magnússon, E., Berthier, E., Gunnlaugsson, Á. Þ., Pálsson, F.,
Aðalgeirsdóttir, G., Jóhannesson, T., Thorsteinsson, T. and Björnsson, H.: Mass
Balance of 14 Icelandic Glaciers, 1945-2017: Spatial Variations and Links With
Climate, Front. Earth Sci., 8, 163, doi:10.3389/feart.2020.00163, 2020.

Table S1 of the supplement (Data Sheet 1) gives 96 geodetic balances between
successive DEMs of aerial photographs, Hexagon KH-9, SPOT 5, ASTER, Pléiades,
ArcticDEM and lidar, from 1945 to 2018, in m w.e. per year. They are kept as
constants in `BELART20_PERIODS`. The **floating-date** balances are used: the change
between the actual dates of the two surveys, rather than the fixed-date ones the
authors correct to the 1st of October with a degree-day model of their own. The
dates of a period are rounded to the first of the nearest month, see
`belart20_periods`. The same balances are in the WGMS Fluctuations of Glaciers
database (`change.csv`), which gives the dates S1 lacks for Barkárdals- and
Tungnahryggsjökull and the mean area of every period (`area_mean_km2`).

**Rate convention.** The published rate is the elevation change, converted to water
equivalent, divided by the difference of the **years** of the two surveys, not by
the time between them: Mýrdalsjökull from the 5th of August 1999 to the 5th of
October 2004 is the change over 5 years, -2.64 m w.e. per year, not over 5.17. Every
balance of S1 is reproduced this way from the WGMS elevation changes. The change over
the period, `cumulative_mwe`, is therefore the rate times that difference of years,
and `mwe_per_year` is that change over the length of the period in months, as
`gridded_utils.geodetic_window_weights` counts it.

**Geometry.** The authors delineated an outline at every survey, but did not publish
them: the paper says they "will be uploaded to GLIMS", which holds no outline of
Iceland other than those of Sigurðsson dated 1999-2004 (checked in 2026), and IMO
publishes none. Every period is therefore gridded on the RGI 6.2 outlines, dated 1999
to 2004, which divide the ice caps into drainage basins: each glacier of the paper is
the explicit list of basins of `BELART20_GLACIERS`, dissolved into one entity. The
lists were drawn from the elevation change maps of the supplement (Figures S2-S15):
Tindfjallajökull and Snæfell include detached parts, Barkárdals- and
Tungnahryggsjökull are three basins, and Öræfajökull is the southern part of
Vatnajökull, without Skaftafellsjökull. Their areas agree with the WGMS mean areas
around 2000 to within 10 %, except Barkárdals- and Tungnahryggsjökull (16.7 km²
against 18.7) and Eiríksjökull (21.9 against 24.1), whose outlines in the paper
include debris-covered ice to the north. The further a period is from 2000, the more
the glacier it describes differs from the one gridded: compare `area_mean_km2` with
`area_rgi` of `load_belart20_outlines`.

Glaciers are identified by a code of their own, `BELART20_GLACIERS`, the abbreviation
of S1 in ASCII ("ORA" for "Öræ"), since every one of them is several RGI outlines. As
for Hagg12, the gridded products of this source carry `RGIId == "ORA"`, and the
crosswalk built by `table_RGI62_to_Belart20` is many RGI ids to one glacier.

The paper converts volume changes to water equivalent with a density of
850 kg m-3, the one of this workflow, so no conversion is needed. Its uncertainties
are at the 95 % confidence level, and are divided by 1.96 to give one sigma.

**Known errors of the WGMS copy.** WGMS gives Hofsjökull Eystri 1967-1990 with the
wrong sign (S1 gives -0.22, consistent with its three sub-periods) and holds two rows
of Eiríksjökull 1960-1995 but none of 1960-1978; the values of S1 are kept.
"""

import geopandas as gpd
import pandas as pd

from data_processing.custom_outlines import CustomOutlineSpec
from data_processing.glacier_utils import get_region_shape_file
from data_processing.utils.periods import (
    best_period_chain,
    first_of_nearest_month,
    select_dem_periods,
)

# Iceland is RGI region 06, with a single second-order region
ICELAND_REGION_ID = 6
ICELAND_SUBREGION = "01"
# Year of the RGI 6.2 outlines of Iceland (1999 to 2004), for OGGM
OUTLINES_YEAR = 2000


def _rgi(*numbers):
    return [f"RGI60-06.{n:05d}" for n in numbers]


# Name, RGI 6.2 outlines and area in 2017 (Table 1, km²) of every glacier, keyed by
# the abbreviation of Table S1 in ASCII
BELART20_GLACIERS = {
    "BAR": (
        "Barkárdals- and Tungnahryggsjökull",
        _rgi(25, 26, 27, 35),
        17.7,
    ),
    "DRA": ("Drangajökull", _rgi(539, 540, 541, 542, 543, 544, 545, 549), 137.6),
    "EIR": ("Eiríksjökull", _rgi(276, 277, 278, 279, 281, 283, 568), 18.6),
    "EYJ": ("Eyjafjallajökull", _rgi(*range(313, 327)), 65.5),
    "HOF": ("Hofsjökull Eystri", _rgi(512, 555), 3.0),
    "HRU": ("Hrútfell", _rgi(270, 271, 272, 273, 274, 275, 567), 4.2),
    "MYR": ("Mýrdalsjökull", _rgi(*range(328, 348)), 517.0),
    "ORA": (
        "Öræfajökull",
        _rgi(406, 407, 423, 425, 426, 427, 428, 429, 436, 445, 453, 474, 476),
        163.2,
    ),
    "SNA": ("Snæfell", _rgi(516, 517, 518, 519, 520, 521, 522, 523), 4.3),
    "SNJ": ("Snæfellsjökull", _rgi(1, 2, 3, 5, 6, 7, 548), 8.3),
    "THR": ("Þrándarjökull", _rgi(513, 514, 553, 554), 14.0),
    "TIN": (
        "Tindfjallajökull",
        _rgi(349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 564),
        10.8,
    ),
    "TOR": ("Torfajökull", _rgi(365, 366, 367, 368, 369, 565), 8.1),
    "TUN": ("Tungnafellsjökull", _rgi(*range(284, 291)), 32.5),
}

# Table S1, floating-date system: (glacier, date of the first survey, date of the
# second survey, balance in m w.e. per year, its uncertainty at the 95 % confidence
# level, mean area of the period in km² from WGMS). The dates of Barkárdals- and
# Tungnahryggsjökull, which S1 gives as years only, are those of WGMS; its first
# survey is dated 1946 in S1 but 1945 in WGMS, which reproduces the published rate.
BELART20_PERIODS = [
    ("BAR", "1945-08-30", "1960-08-24", -0.16, 0.10, 19.09),
    ("BAR", "1960-08-24", "1980-08-22", -0.02, 0.05, 18.43),
    ("BAR", "1960-08-24", "1994-08-07", 0.07, 0.02, 18.84),
    ("BAR", "1980-08-22", "1994-08-07", 0.19, 0.05, 18.39),
    ("BAR", "1994-08-07", "2004-08-08", -0.44, 0.08, 18.67),
    ("BAR", "1994-08-07", "2016-10-10", -0.47, 0.03, 18.21),
    ("BAR", "2004-08-08", "2016-10-10", -0.59, 0.07, 18.08),
    ("DRA", "2011-07-20", "2016-07-05", 0.18, 0.02, 144.26),
    ("EIR", "1960-08-18", "1978-09-05", -0.02, 0.10, 25.06),
    ("EIR", "1960-08-18", "1995-08-24", 0.03, 0.05, 25.24),
    ("EIR", "1978-09-05", "1986-08-15", 0.13, 0.19, 25.54),
    ("EIR", "1986-08-15", "1995-08-24", -0.01, 0.17, 25.72),
    ("EIR", "1995-08-24", "2004-08-19", -0.55, 0.17, 24.09),
    ("EIR", "1995-08-24", "2008-09-02", -0.62, 0.11, 24.12),
    ("EIR", "2004-08-19", "2008-09-02", -0.95, 0.19, 22.54),
    ("EIR", "2008-09-02", "2017-07-10", 0.03, 0.02, 22.63),
    ("EYJ", "1994-08-06", "2010-08-10", -1.51, 0.11, 82.26),
    ("EYJ", "2010-08-10", "2017-08-10", 0.26, 0.15, 71.11),
    ("HOF", "1946-10-13", "1967-09-30", -0.49, 0.08, 6.64),
    ("HOF", "1967-09-30", "1976-08-26", -0.34, 0.12, 6.06),
    ("HOF", "1967-09-30", "1990-08-02", -0.22, 0.13, 6.23),
    ("HOF", "1976-08-26", "1983-07-27", 0.11, 0.04, 6.46),
    ("HOF", "1983-07-27", "1990-08-02", -0.31, 0.05, 6.63),
    ("HOF", "1990-08-02", "2004-10-08", -0.78, 0.08, 4.99),
    ("HOF", "1990-08-02", "2012-08-03", -0.90, 0.07, 4.87),
    ("HOF", "2004-10-08", "2012-08-03", -1.49, 0.15, 3.61),
    ("HRU", "1946-10-13", "1960-08-08", -0.27, 0.11, 7.37),
    ("HRU", "1960-08-08", "1980-08-22", 0.01, 0.08, 7.44),
    ("HRU", "1960-08-08", "1995-08-24", 0.09, 0.04, 7.66),
    ("HRU", "1980-08-22", "1987-08-05", 0.09, 0.15, 8.09),
    ("HRU", "1987-08-05", "1995-08-24", -0.10, 0.22, 8.31),
    ("HRU", "1995-08-24", "2004-08-19", -1.02, 0.10, 7.00),
    ("HRU", "1995-08-24", "2013-08-01", -1.03, 0.17, 6.68),
    ("HRU", "2004-08-19", "2013-08-01", -0.78, 0.11, 5.35),
    ("MYR", "1960-08-13", "1980-08-22", -0.04, 0.06, 612.46),
    ("MYR", "1960-08-13", "1999-08-05", -0.12, 0.02, 610.59),
    ("MYR", "1980-08-22", "1984-09-04", -0.57, 0.36, 602.50),
    ("MYR", "1984-09-04", "1999-08-05", -0.10, 0.10, 600.63),
    ("MYR", "1999-08-05", "2004-10-05", -2.64, 0.21, 587.88),
    ("MYR", "1999-08-05", "2010-08-08", -1.65, 0.12, 579.49),
    ("MYR", "2004-10-05", "2010-08-08", -1.42, 0.10, 570.79),
    ("MYR", "2010-08-08", "2016-09-19", -0.63, 0.05, 546.95),
    ("ORA", "1945-08-30", "1960-08-23", -0.25, 0.22, 179.61),
    ("ORA", "1960-08-23", "1982-08-20", 0.10, 0.10, 178.33),
    ("ORA", "1960-08-23", "1992-07-27", 0.06, 0.06, 178.73),
    ("ORA", "1982-08-20", "1988-08-22", -0.12, 0.27, 179.45),
    ("ORA", "1988-08-22", "1992-07-27", -0.04, 0.32, 179.85),
    ("ORA", "1992-07-27", "2003-08-06", -0.16, 0.17, 174.16),
    ("ORA", "1992-07-27", "2010-09-13", -0.87, 0.07, 169.93),
    ("ORA", "2003-08-06", "2010-09-13", -1.28, 0.16, 164.43),
    ("ORA", "2010-09-13", "2017-08-16", 0.44, 0.07, 161.71),
    ("SNA", "1984-08-25", "1993-08-25", 0.09, 0.16, 5.05),
    ("SNA", "1993-08-25", "2012-07-12", -0.22, 0.02, 4.85),
    ("SNJ", "1945-09-16", "1959-07-27", 0.05, 0.24, 13.24),
    ("SNJ", "1959-07-27", "1979-07-22", 0.02, 0.05, 13.26),
    ("SNJ", "1959-07-27", "1991-07-19", 0.21, 0.02, 13.41),
    ("SNJ", "1979-07-22", "1985-08-13", 0.69, 0.15, 14.03),
    ("SNJ", "1985-08-13", "1991-07-19", 0.13, 0.05, 14.18),
    ("SNJ", "1991-07-19", "2008-09-02", -1.02, 0.08, 12.06),
    ("SNJ", "2008-09-02", "2018-07-17", 0.01, 0.05, 9.60),
    ("THR", "1976-08-26", "1982-08-01", 0.02, 0.08, 21.28),
    ("THR", "1982-08-01", "1990-08-02", -0.16, 0.23, 20.58),
    ("THR", "1990-08-02", "2004-10-08", -0.71, 0.20, 18.08),
    ("THR", "1990-08-02", "2012-08-03", -0.71, 0.16, 17.05),
    ("THR", "2004-10-08", "2012-08-03", -1.24, 0.12, 15.39),
    ("TIN", "1945-08-30", "1960-08-05", -0.36, 0.12, 16.63),
    ("TIN", "1960-08-05", "1978-09-10", 0.01, 0.01, 16.29),
    ("TIN", "1960-08-05", "1980-08-22", 0.09, 0.06, 16.41),
    ("TIN", "1960-08-05", "1994-08-12", 0.18, 0.02, 16.98),
    ("TIN", "1978-09-10", "1990-04-09", 0.27, 0.07, 16.63),
    ("TIN", "1980-08-22", "1990-04-09", 0.06, 0.15, 16.75),
    ("TIN", "1990-04-09", "1994-08-12", 0.54, 0.15, 17.32),
    ("TIN", "1994-08-12", "2004-10-05", -1.09, 0.14, 16.52),
    ("TIN", "1994-08-12", "2011-08-09", -1.26, 0.09, 15.37),
    ("TIN", "2004-10-05", "2011-08-09", -1.45, 0.19, 14.36),
    ("TIN", "2011-08-09", "2017-07-11", -0.15, 0.08, 13.18),
    ("TOR", "1945-10-14", "1960-08-13", -0.54, 0.13, 15.62),
    ("TOR", "1960-09-13", "1970-09-03", -0.95, 0.25, 14.78),
    ("TOR", "1960-09-13", "1990-09-04", -0.30, 0.10, 14.47),
    ("TOR", "1970-09-03", "1979-08-09", 0.21, 0.19, 14.28),
    ("TOR", "1970-09-03", "1980-08-22", 0.01, 0.18, 14.27),
    ("TOR", "1979-08-09", "1990-09-04", -0.10, 0.09, 13.97),
    ("TOR", "1980-08-22", "1990-09-04", -0.05, 0.12, 13.96),
    ("TOR", "1990-09-04", "1999-08-05", -0.43, 0.15, 13.48),
    ("TOR", "1990-09-04", "2011-08-08", -1.21, 0.12, 11.48),
    ("TOR", "1999-08-05", "2004-10-05", -2.56, 0.43, 12.05),
    ("TOR", "2004-10-05", "2011-08-08", -2.07, 0.20, 10.05),
    ("TOR", "2011-08-08", "2016-09-18", -1.23, 0.09, 8.68),
    ("TUN", "1960-08-12", "1980-08-22", 0.13, 0.24, 40.22),
    ("TUN", "1960-08-12", "1995-08-24", 0.08, 0.02, 40.18),
    ("TUN", "1980-08-22", "1986-08-18", -0.20, 0.63, 40.26),
    ("TUN", "1986-08-18", "1995-08-24", 0.23, 0.14, 40.21),
    ("TUN", "1995-08-24", "2004-08-14", -0.83, 0.12, 38.69),
    ("TUN", "1995-08-24", "2010-06-06", -0.56, 0.07, 38.13),
    ("TUN", "2004-08-14", "2010-06-06", -0.26, 0.05, 35.74),
    ("TUN", "2010-06-06", "2017-10-15", -0.87, 0.09, 34.01),
]

# Density of the paper, the one of this workflow: no conversion is needed
ICE_DENSITY = 850.0
# The uncertainties of the paper are at the 95 % confidence level
CI95_TO_SIGMA = 1 / 1.96


def belart20_periods():
    """Every period of `BELART20_PERIODS` with its dates, rate and uncertainty, one row
    per period.

    Columns: RGIId (glacier code), name, y0, y1 (years of the two surveys),
    survey_from and survey_to (their dates), FROM_DATE and TO_DATE (those dates
    rounded to the first of the nearest month), n_years (the months between them, over
    12), cumulative_mwe, mwe_per_year, sigma_mwe_per_year, mwe_per_year_published,
    sigma95_published and area_mean_km2 (the mean area of the period, from WGMS).

    The published rate is the change over `y1 - y0` years (see the module docstring),
    so the change is `cumulative_mwe = mwe_per_year_published * (y1 - y0)` and the rate
    the pipeline compares the model with is that change over `n_years`. The
    uncertainty is converted the same way, after division by 1.96.
    """
    df = pd.DataFrame(
        BELART20_PERIODS,
        columns=[
            "RGIId",
            "survey_from",
            "survey_to",
            "mwe_per_year_published",
            "sigma95_published",
            "area_mean_km2",
        ],
    )
    df.insert(1, "name", df.RGIId.map(lambda g: BELART20_GLACIERS[g][0]))
    df["survey_from"] = pd.to_datetime(df.survey_from)
    df["survey_to"] = pd.to_datetime(df.survey_to)
    df["y0"] = df.survey_from.dt.year
    df["y1"] = df.survey_to.dt.year
    df["FROM_DATE"] = df.survey_from.map(first_of_nearest_month)
    df["TO_DATE"] = df.survey_to.map(first_of_nearest_month)
    months = (df.TO_DATE.dt.year - df.FROM_DATE.dt.year) * 12 + (
        df.TO_DATE.dt.month - df.FROM_DATE.dt.month
    )
    df["n_years"] = months / 12
    published_years = df.y1 - df.y0
    df["cumulative_mwe"] = df.mwe_per_year_published * published_years
    df["mwe_per_year"] = df.cumulative_mwe / df.n_years
    df["sigma_mwe_per_year"] = (
        df.sigma95_published * CI95_TO_SIGMA * published_years / df.n_years
    )
    return df[
        [
            "RGIId",
            "name",
            "y0",
            "y1",
            "survey_from",
            "survey_to",
            "FROM_DATE",
            "TO_DATE",
            "n_years",
            "cumulative_mwe",
            "mwe_per_year",
            "sigma_mwe_per_year",
            "mwe_per_year_published",
            "sigma95_published",
            "area_mean_km2",
        ]
    ]


def geodetic_target_Belart20(
    max_year: int = None,
    min_year: int = None,
    min_period_years: int = 1,
    glacier_ids_to_keep=None,
    allowed_years=None,
    max_outside_years: int = 0,
    multi_period: bool = True,
    tie_break: str = "sigma",
):
    """Geodetic targets of Belart et al. (2020), one row per period between two surveys.

    Returns a dataframe shaped like the one `data_processing.glamos.geodetic_target_GLAMOS`
    returns, so that both drive the same code path in `GeoDataLoader`:

        RGIId                 the glacier code, e.g. "ORA"
        FROM_DATE, TO_DATE    the survey dates, rounded to the first of a month
        cumulative_mwe        the change over the period, m w.e.
        mwe_per_year          that change over the length of the period
        sigma_mwe_per_year    its uncertainty, one sigma

    The periods are first selected by `utils.periods.select_dem_periods`, which
    documents the arguments. Unlike those of Hagg12, the periods of a glacier are
    nested (1960-1980, 1960-1994 and 1980-1994), so with `multi_period` a glacier keeps
    the best chain of non-overlapping periods, see `utils.periods.best_period_chain`,
    and without it the longest period, ties going to the lowest sigma.
    """
    df = select_dem_periods(
        belart20_periods(),
        max_year=max_year,
        min_year=min_year,
        min_period_years=min_period_years,
        glacier_ids_to_keep=glacier_ids_to_keep,
        allowed_years=allowed_years,
        max_outside_years=max_outside_years,
        multi_period=multi_period,
        tie_break=tie_break,
    )
    if multi_period and len(df):
        df = pd.concat([best_period_chain(g) for _, g in df.groupby("RGIId")])
    return df.sort_values(["RGIId", "FROM_DATE"]).reset_index(drop=True)


def _rgi_outlines():
    """The RGI 6.2 outlines of the glaciers, one row per RGI id, with the glacier code
    in a column named "BELART20_ID"."""
    of_rgi = {r: g for g, (_, rgi_ids, _) in BELART20_GLACIERS.items() for r in rgi_ids}
    rgi = gpd.read_file(get_region_shape_file(f"{ICELAND_REGION_ID:02d}"))
    rgi = rgi[rgi.RGIId.isin(of_rgi)].copy()
    missing = set(of_rgi).difference(rgi.RGIId)
    assert not missing, f"RGI 6.2 has no outline {sorted(missing)}."
    rgi["BELART20_ID"] = rgi.RGIId.map(of_rgi)
    return rgi


def load_belart20_outlines(glacier_ids_to_keep=None):
    """Read the RGI 6.2 outlines of the Icelandic glaciers, one entity per glacier.

    The basins of `BELART20_GLACIERS` are dissolved into one entity per glacier. The
    returned frame is in EPSG:4326 with the glacier code in a column named
    "BELART20_ID", ready for `custom_outlines.build_custom_gdirs`, along with the name,
    the area of the entity in km², summed from the RGI areas, and the area of 2017 of
    Table 1.
    """
    rgi = _rgi_outlines()
    outlines = rgi.dissolve("BELART20_ID", aggfunc={"Area": "sum"}).reset_index()
    outlines = outlines.rename(columns={"Area": "area_rgi"})
    outlines["name"] = outlines.BELART20_ID.map(lambda g: BELART20_GLACIERS[g][0])
    outlines["area_2017"] = outlines.BELART20_ID.map(lambda g: BELART20_GLACIERS[g][2])
    if glacier_ids_to_keep is not None:
        missing = set(glacier_ids_to_keep).difference(outlines.BELART20_ID)
        assert not missing, f"Belart20 has no glacier {sorted(missing)}."
        outlines = outlines[outlines.BELART20_ID.isin(glacier_ids_to_keep)]
    return outlines[
        ["BELART20_ID", "name", "area_rgi", "area_2017", "geometry"]
    ].to_crs("EPSG:4326")


def belart20_outline_spec(dem_source: str = "COPDEM30", **overrides):
    """Description of the Icelandic outlines for the generic custom-outline machinery.

    The DEM defaults to the Copernicus GLO-30 DEM, 30 m, acquired in 2011-2015: SRTM,
    the February 2000 surface GLAMOS and Maurer19 are gridded on, stops at 60°N, and
    Iceland lies between 63°N and 67°N.
    """
    spec = dict(
        name="Belart20",
        id_column="BELART20_ID",
        o1_region=f"{ICELAND_REGION_ID:02d}",
        o2_region=ICELAND_SUBREGION,
        src_date=f"{OUTLINES_YEAR}-01-01 00:00:00",
        bgndate=f"{OUTLINES_YEAR}0101",
        dem_source=dem_source,
    )
    spec.update(overrides)
    return CustomOutlineSpec(**spec)


def table_RGI62_to_Belart20(region_id=ICELAND_REGION_ID):
    """Crosswalk from RGI 6.2 ids to the Icelandic glaciers.

    The outlines of this source are the RGI 6.2 ones, so the crosswalk is read off
    `BELART20_GLACIERS` rather than matched by overlap: every RGI outline lies entirely
    in its glacier. Only region 06 holds any.

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
    if int(region_id) != ICELAND_REGION_ID:
        return pd.DataFrame(columns=columns)
    rgi = _rgi_outlines().rename(
        columns={"BELART20_ID": "custom_id", "Area": "area_rgi"}
    )
    rgi["area_custom"] = rgi.groupby("custom_id").area_rgi.transform("sum")
    rgi["frac_rgi"] = 1.0
    rgi["frac_custom"] = rgi.area_rgi / rgi.area_custom
    return pd.DataFrame(rgi[columns]).sort_values("RGIId").reset_index(drop=True)
