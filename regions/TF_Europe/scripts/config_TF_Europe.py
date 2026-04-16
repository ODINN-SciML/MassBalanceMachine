from pathlib import Path

# <------------------ PATHS ------------------>
path_cache = Path("cache/TF_Europe")  # Cache directory for intermediate files
path_ERA5_raw = Path("ERA5Land/raw/")  # ERA5-Land
path_OGGM = Path("OGGM")  # OGGM working directory (relative to cfg.dataPath)
path_PMB_WGMS_csv = Path("WGMS")
path_PMB_GLACIOCLIM_csv = "GLACIOCLIM/point/csv/"

# Base folder for RGI v6 relative to cfg.dataPath (or whatever root you use)
RGI_V6_ROOT = Path("RGI_v6")

RGI_REGIONS = {
    "01": {
        "name": "Alaska",
        "folder": "RGI_01_Alaska",
        "file": "01_rgi60_Alaska.shp",
        "code": "ALA",
    },
    "02": {
        "name": "WesternCanadaUS",
        "folder": "RGI_02_WesternCanadaUS",
        "file": "02_rgi60_WesternCanadaUS.shp",
        "code": "CAW",
    },
    "06": {
        "name": "Iceland",
        "folder": "RGI_06_Iceland",
        "file": "06_rgi60_Iceland.shp",
        "code": "ISL",
    },
    "07": {
        "name": "Svalbard",
        "folder": "RGI_07_Svalbard",
        "file": "07_rgi60_Svalbard.shp",
        "code": "SJM",
    },
    "08": {
        "name": "Scandinavia",
        "folder": "RGI_08_Scandinavia",
        "file": "08_rgi60_Scandinavia.shp",
        "code": "SCA",
        "subregions": ["Norway"],
        "subregions_codes": ["NOR"],
    },
    "11": {
        "name": "CentralEurope",
        "folder": "RGI_11_CentralEurope",
        "file": "11_rgi60_CentralEurope.shp",
        "code": "CEU",
        "subregions": ["France", "Switzerland", "Italy_Austria"],
        "subregions_codes": ["FR", "CH", "IT_AT"],
    },
}


# --- 1) One place to define the per-target region metadata you need for mapping ---
TARGET_REGION_META = {
    "ISL": {
        "rgi_region_id": "06",
        "outline_shp_rel": "RGI_v6/RGI_06_Iceland/06_rgi60_Iceland.shp",
        "title": "Glacier PMB locations Iceland",
        "extent": (-25, -11, 62, 68),
    },
    "NOR": {
        "rgi_region_id": "08",
        "outline_shp_rel": "RGI_v6/RGI_08_Scandinavia/08_rgi60_Scandinavia.shp",
        # If you actually use the Norway-only file in your repo, swap path accordingly.
        "title": "Glacier PMB locations Norway",
        # rough bbox for mainland Norway + Svalbard excluded; adjust if needed
        "extent": (4, 32, 57, 72),
    },
    "CH": {
        "rgi_region_id": "11",
        "outline_shp_rel": "RGI_v6/RGI_11_CentralEurope/11_rgi60_CentralEurope.shp",
        # If you have a Switzerland-only outline, swap path accordingly.
        "title": "Glacier PMB locations Switzerland",
        # rough bbox for Switzerland
        "extent": (5.8, 13.7, 44.5, 47.9),
    },
    "CEU": {
        "rgi_region_id": "11",
        "outline_shp_rel": "RGI_v6/RGI_11_CentralEurope/11_rgi60_CentralEurope.shp",
        # If you have a Switzerland-only outline, swap path accordingly.
        "title": "Glacier PMB locations Central Europe (FR+CH+IT+AT)",
        # rough bbox for CEU
        "extent": (5.8, 13.7, 44.5, 47.9),
    },
    "USCA": {
        "rgi_region_id": ["01", "02"],
        "outline_shp_rel": "RGI_v6/RGI_01_Alaska/01_rgi60_Alaska.shp",
        # If you have a Switzerland-only outline, swap path accordingly.
        "title": "Glacier PMB locations US CA (ALA+CAW)",
        # rough bbox for CEU
        "extent": (5.8, 13.7, 44.5, 47.9),
    },
}


def rgi_outline_path(rgi_id: str) -> str:
    """Return the relative path to the RGI outlines shapefile for a region id like '07'."""
    rgi_id = str(rgi_id).zfill(2)
    spec = RGI_REGIONS[rgi_id]
    return str(RGI_V6_ROOT / spec["folder"] / spec["file"])


vois_climate_long_name = {
    "t2m": "Temperature",
    "tp": "Precipitation",
    "slhf": "Surf. latent heat flux",
    "sshf": "Surf. sensible heat flux",
    "ssrd": "Surf. solar rad. down.",
    "fal": "Albedo",
    "str": "Surf. net thermal rad.",
    "pcsr": "Pot. in. clear sky solar rad.",
    "u10": "10m E wind",
    "v10": "10m N wind",
    "ELEVATION_DIFFERENCE": "Elev. difference",
    "hugonnet_dhdt": "Hugonnet dH/dt",
    "consensus_ice_thickness": "Cons. ice thickness",
    "millan_v": "Millan ice velocity",
    "aspect_sgi": "Aspect",
    "slope_sgi": "Slope",
    "svf": "Sky view factor",
    "aspect": "Aspect",
    "slope": "Slope",
    "POINT_BALANCE": "Point mass balance",
}

vois_units = {
    "t2m": "C",
    "tp": "m w.e.",
    "t2m_corr": "C",
    "tp_corr": "m w.e.",
    "slhf": "J m-2",
    "sshf": "J m-2",
    "ssrd": "J m-2",
    "fal": "",
    "str": "J m-2",
    "pcsr": "J m-2",
    "u10": "m s-1",
    "v10": "m s-1",
    "aspect_sgi": "rad",
    "slope_sgi": "rad",
    "aspect": "rad",
    "slope": "rad",
    "svf": "",
    "ELEVATION_DIFFERENCE": "m",
    "POINT_BALANCE": "m w.e.",
}

VOIS_CLIMATE = [
    "t2m",
    "tp",
    "slhf",
    "sshf",
    "ssrd",
    "fal",
    "str",
]

VOIS_TOPOGRAPHICAL = ["aspect", "slope", "svf"]

FR_gl_name = {
    "FR4N01235A08 dArgentiere": "Argentiere",
    "FR4N01236A02 des Grands Montets": "Grands Montets",
    "FR4N01146D09+E06 Gebroulaz": "Gebroulaz",
    "FR4N01083B21 Blanc": "Blanc",
    "FR4N01236A01 Mer de Glace/Geant": "Mer de Glace",
    "FR4N01236A01 Leschaux": "Leschaux",
    "FR4N01236A07 de Talefre": "Talefre",
    "FR4N01163A02 de Sarennes 1": "Sarennes",
    "FR4N01162B09+154D03 de Saint Sorlin": "Saint Sorlin",
}
