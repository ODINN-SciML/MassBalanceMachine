# <------------------ GLAMOS DATA: ------------------>
# Point data
path_PMB_GLAMOS_raw = "GLAMOS/point/point_raw/"
path_PMB_GLAMOS_w_raw = path_PMB_GLAMOS_raw + "winter/"
path_PMB_GLAMOS_a_raw = path_PMB_GLAMOS_raw + "annual/"

path_PMB_GLAMOS_csv = "GLAMOS/point/csv/"
path_PMB_GLAMOS_csv_w = path_PMB_GLAMOS_csv + "winter/"
path_PMB_GLAMOS_csv_w_clean = path_PMB_GLAMOS_csv + "winter_clean/"
path_PMB_GLAMOS_csv_a = path_PMB_GLAMOS_csv + "annual/"

# Glacier wide data
path_SMB_GLAMOS_raw = "GLAMOS/glacier-wide/raw/"
path_SMB_GLAMOS_csv = "GLAMOS/glacier-wide/csv/"

# Gridded data for MBM to use for making predictions over whole grid (SGI or RGI grid)
path_glacier_grid_rgi = "GLAMOS/topo/gridded_topo_inputs/RGI_grid/"  # DEMs & topo
path_glacier_grid_sgi = "GLAMOS/topo/gridded_topo_inputs/SGI_grid/"  # DEMs & topo
path_glacier_grid_glamos = "GLAMOS/topo/gridded_topo_inputs/GLAMOS_grid/"

# Topo data
path_SGI_topo = "GLAMOS/topo/SGI2020/"  # DEMs & topo from SGI
path_GLAMOS_topo = "GLAMOS/topo/GLAMOS_DEM/"  # yearly DEMs from GLAMOS
path_pcsr = (
    "GLAMOS/topo/pcsr/"  # Potential incoming clear sky solar radiation from GLAMOS
)

path_distributed_MB_glamos = "GLAMOS/distributed_MB_grids/"
path_geodetic_MB_glamos = "GLAMOS/geodetic/"
path_glacier_ids = "GLAMOS/CH_glacier_ids_long.csv"  # glacier ids for CH glaciers

# <------------------ OTHER PATHS: ------------------>
path_ERA5_raw = "ERA5Land/raw/"  # ERA5-Land
path_S2 = "Sentinel/"  # Sentinel-2
path_OGGM = "OGGM/"
path_OGGM_xrgrids = "OGGM/xr_grids/"
path_glogem = "GloGEM"  # glogem c_prec and t_off factors
path_rgi_outlines = (
    "GLAMOS/RGI/nsidc0770_11.rgi60.CentralEurope/11_rgi60_CentralEurope.shp"
)

# <------------------OTHER USEFUL FUNCTIONS & ATTRIBUTES: ------------------>
vois_climate_long_name = {
    "t2m": "Temp.",
    "tp": "Precip.",
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

TEST_GLACIERS = [
    "tortin",
    "plattalva",
    "albigna",
    "schwarzberg",
    "hohlaub",
    "sanktanna",
    "corvatsch",
    "tsanfleuron",
    "forno",
]
