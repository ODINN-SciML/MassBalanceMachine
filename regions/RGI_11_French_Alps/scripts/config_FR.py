# <------------------ GLACIOCLIM DATA: ------------------>
# Point data
path_PMB_GLACIOCLIM_raw = "GLACIOCLIM/point/unzipped/"
path_PMB_GLACIOCLIM_csv = "GLACIOCLIM/point/csv/"
path_PMB_GLAMOS_csv = "GLAMOS/point/csv/"

# <------------------ OTHER PATHS: ------------------>
path_ERA5_raw = "ERA5Land/raw/"  # ERA5-Land
path_OGGM = "OGGM/rgi_region_11/"
path_OGGM_xrgrids = "OGGM/rgi_region_11/xr_grids/"
path_rgi_outlines = "RGI_v6/RGI_11_CentralEurope/11_rgi60_CentralEurope.shp"

# <------------------OTHER USEFUL FUNCTIONS & ATTRIBUTES: ------------------>
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
