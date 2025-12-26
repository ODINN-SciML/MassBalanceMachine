# <------------------ MODEL INFO: ------------------>

TEST_GLACIERS = [
    "tortin",
    "plattalva",
    "schwarzberg",
    "hohlaub",
    "sanktanna",
    "corvatsch",
    "tsanfleuron",
    "forno",
]

# Saved models
LSTM_OOS_NORM_Y = "models/lstm_model_2025-12-16_OOS_norm_y.pt"
LSTM_IS_NORM_Y = "models/lstm_model_2025-12-16_IS_norm_y.pt"
LSTM_IS_ORIGIN_Y = "models/lstm_model_2025-12-16_IS_orig_y.pt"

LSTM_OOS_NORM_Y_PAST = "models/lstm_model_2025-12-23_OOS_norm_y_past.pt"
LSTM_IS_NORM_Y_PAST = "models/lstm_model_2025-12-23_IS_norm_y_past.pt"
LSTM_IS_ORIGIN_Y_PAST = "models/lstm_model_2025-12-23_IS_orig_y_past.pt"

VOIS_CLIMATE = [
    "t2m",
    "tp",
    "slhf",
    "sshf",
    "ssrd",
    "fal",
    "str",
]

VOIS_TOPOGRAPHICAL = ["aspect_sgi", "slope_sgi", "svf"]

# Model parameters
PARAMS_LSTM_OOS = {
    "lr": 0.001,
    "weight_decay": 0.0001,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "head_dropout": 0.0,
    "static_layers": 2,
    "static_hidden": [128, 64],
    "static_dropout": 0.1,
    "Fm": 9,
    "Fs": 3,
    "bidirectional": False,
    "loss_name": "neutral",
    "loss_spec": None,
    "two_heads": False,
}

PARAMS_LSTM_IS = {
    "lr": 0.001,
    "weight_decay": 0.0,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.1,
    "head_dropout": 0.1,
    "static_layers": 2,
    "static_hidden": [128, 64],
    "static_dropout": 0.1,
    "Fm": 9,
    "Fs": 3,
    "bidirectional": False,
    "loss_name": "neutral",
    "loss_spec": None,
    "two_heads": False,
}

PARAMS_LSTM_OOS_PAST = {
    "Fm": 9,
    "Fs": 3,
    "hidden_size": 64,
    "num_layers": 2,
    "bidirectional": False,
    "dropout": 0.1,
    "static_layers": 2,
    "static_hidden": 128,
    "static_dropout": 0.2,
    "lr": 0.002,
    "weight_decay": 0.0001,
    "loss_name": "neutral",
    "two_heads": False,
    "head_dropout": 0.1,
    "loss_spec": None,
}

PARAMS_LSTM_IS_past = {
    "Fm": 9,
    "Fs": 3,
    "hidden_size": 64,
    "num_layers": 2,
    "bidirectional": False,
    "dropout": 0.1,
    "static_layers": 2,
    "static_hidden": 32,
    "static_dropout": 0.1,
    "lr": 0.0005,
    "weight_decay": 1e-05,
    "loss_name": "neutral",
    "two_heads": False,
    "head_dropout": 0.0,
    "loss_spec": None,
}

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
    "aspect_sgi": "rad",
    "slope_sgi": "rad",
    "aspect": "rad",
    "slope": "rad",
    "svf": "",
    "ELEVATION_DIFFERENCE": "m",
}

COLOR_ANNUAL = "#c51b7d"
COLOR_WINTER = "#011959"
