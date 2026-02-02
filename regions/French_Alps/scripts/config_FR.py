# <------------------ GLACIOCLIM DATA: ------------------>
# Point data
path_PMB_GLACIOCLIM_raw = "GLACIOCLIM/unzipped/"
path_PMB_GLACIOCLIM_csv = "GLACIOCLIM/csv/"

# <------------------ OTHER PATHS: ------------------>
path_ERA5_raw = "ERA5Land/raw/"  # ERA5-Land
path_OGGM = "OGGM/"
path_OGGM_xrgrids = "OGGM/xr_grids/"
path_rgi_outlines = (
    "GLAMOS/RGI/nsidc0770_11.rgi60.CentralEurope/11_rgi60_CentralEurope.shp"
)

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
