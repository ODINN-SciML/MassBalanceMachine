from pathlib import Path

# <------------------ PATHS ------------------>
path_ERA5_raw = Path("ERA5Land/raw/")  # ERA5-Land
path_OGGM = Path("OGGM")  # OGGM working directory (relative to cfg.dataPath)
path_PMB_WGMS_csv = Path("WGMS")
path_PMB_GLACIOCLIM_csv = "GLACIOCLIM/point/csv/"

# Base folder for RGI v6 relative to cfg.dataPath (or whatever root you use)
RGI_V6_ROOT = Path("RGI_v6")

RGI_REGIONS = {
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


def rgi_outline_path(rgi_id: str) -> str:
    """Return the relative path to the RGI outlines shapefile for a region id like '07'."""
    rgi_id = str(rgi_id).zfill(2)
    spec = RGI_REGIONS[rgi_id]
    return str(RGI_V6_ROOT / spec["folder"] / spec["file"])


vois_climate_long_name = {
    "t2m": "Temperature",
    "tp": "Precipitation",
    "t2m_corr": "Temperature corr.",
    "tp_corr": "Precipitation corr.",
    "slhf": "Surf. latent heat flux",
    "sshf": "Surf. sensible heat flux",
    "ssrd": "Surf. solar rad. down.",
    "fal": "Albedo",
    "str": "Surf. net thermal rad.",
    "pcsr": "Pot. in. clear sky solar rad.",
    "u10": "10m E wind",
    "v10": "10m N wind",
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
