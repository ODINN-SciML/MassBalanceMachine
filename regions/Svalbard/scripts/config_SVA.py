# <------------------ PATHS ------------------>
path_PMB_WGMS_raw = "WGMS/Svalbard/raw/"  # Raw Stake measurement ".csv"s
path_PMB_WGMS_csv = "WGMS/Svalbard/csv/"  # Processed stake measurements
path_ERA5_raw = "ERA5Land/raw/"  # ERA5-Land
path_OGGM_SVA = "OGGM/rgi_region_07"  # OGGM Data
path_OGGM_SVA_xrgrids = "OGGM/xr_grids/"  # OGGM Data Grids
path_rgi_outlines_SVA = "RGI_v6/RGI_07_Svalbard/07_rgi60_Svalbard.shp"

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
