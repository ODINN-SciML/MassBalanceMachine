import os

# <------------------ BASE PATHS: ------------------>
# Input data base path (read-only)
path_input_base = '/home/vmarijn/scratch/data/'
# Output data base path (writable)
path_output_base = '/home/mburlet/DATA_MBM/'

# <------------------ GLAMOS DATA: ------------------>
# Point data - Input paths
path_PMB_GLAMOS_raw = os.path.join(path_input_base, 'GLAMOS', 'point', 'point_raw')
path_PMB_GLAMOS_w_raw = os.path.join(path_PMB_GLAMOS_raw, 'winter')
path_PMB_GLAMOS_a_raw = os.path.join(path_PMB_GLAMOS_raw, 'annual')

# Point data - Output paths
path_PMB_GLAMOS_csv = os.path.join(path_output_base, 'GLAMOS', 'point', 'csv')
path_PMB_GLAMOS_csv_w = os.path.join(path_PMB_GLAMOS_csv, 'winter')
path_PMB_GLAMOS_csv_w_clean = os.path.join(path_PMB_GLAMOS_csv, 'winter_clean')
path_PMB_GLAMOS_csv_a = os.path.join(path_PMB_GLAMOS_csv, 'annual')

# Glacier wide data
path_SMB_GLAMOS_raw = os.path.join(path_input_base, 'GLAMOS', 'glacier-wide', 'raw')
path_SMB_GLAMOS_csv = os.path.join(path_output_base, 'GLAMOS', 'glacier-wide', 'csv')
path_SMB_GLAMOS_csv_obs = os.path.join(path_SMB_GLAMOS_csv, 'obs')

# Gridded data for MBM to use for making predictions over whole grid (SGI or RGI grid)
path_glacier_grid_rgi = os.path.join(path_input_base, 'GLAMOS', 'topo', 'gridded_topo_inputs', 'RGI_grid')
path_glacier_grid_sgi = os.path.join(path_input_base, 'GLAMOS', 'topo', 'gridded_topo_inputs', 'SGI_grid')
path_glacier_grid_glamos = os.path.join(path_input_base, 'GLAMOS', 'topo', 'gridded_topo_inputs', 'GLAMOS_grid')

# Topo data
path_SGI_topo = os.path.join(path_input_base, 'GLAMOS', 'topo', 'SGI2020')
path_GLAMOS_topo = os.path.join(path_input_base, 'GLAMOS', 'topo', 'GLAMOS_DEM')
path_pcsr = os.path.join(path_input_base, 'GLAMOS', 'topo', 'pcsr')
path_pcsr_save = os.path.join(path_output_base, 'GLAMOS', 'topo', 'pcsr', 'csv')


#path_distributed_MB_glamos = '../../../data/GLAMOS/distributed_MB_grids/'
#path_geodetic_MB_glamos = '../../../data/GLAMOS/geodetic/'
#path_glacier_ids = '../../../data/GLAMOS/CH_glacier_ids_long.csv' # glacier ids for CH glaciers

# <------------------ OTHER PATHS: ------------------>
path_ERA5_raw = '../../../data/ERA5Land/raw/' # ERA5-Land
path_S2 = '../../../data/Sentinel/' # Sentinel-2
path_OGGM = '../../../data/OGGM/'
path_glogem = '../../../data/GloGEM' # glogem c_prec and t_off factors
path_rgi_outlines = '../../../data/GLAMOS/RGI/nsidc0770_11.rgi60.CentralEurope/11_rgi60_CentralEurope.shp'

# <------------------OTHER USEFUL FUNCTIONS & ATTRIBUTES: ------------------>
vois_climate_long_name = {
    't2m': 'Temperature',
    'tp': 'Precipitation',
    't2m_corr': 'Temperature corr.',
    'tp_corr': 'Precipitation corr.',
    'slhf': 'Surf. latent heat flux',
    'sshf': 'Surf. sensible heat flux',
    'ssrd': 'Surf. solar rad. down.',
    'fal': 'Albedo',
    'str': 'Surf. net thermal rad.',
    'pcsr': 'Pot. in. clear sky solar rad.',
    'u10': '10m E wind',
    'v10': '10m N wind',
}

vois_units = {
    't2m': 'C',
    'tp': 'm w.e.',
    't2m_corr': 'C',
    'tp_corr': 'm w.e.',
    'slhf': 'J m-2',
    'sshf': 'J m-2',
    'ssrd': 'J m-2',
    'fal': '',
    'str': 'J m-2',
    'pcsr': 'J m-2',
    'u10': 'm s-1',
    'v10': 'm s-1',
}

# Create output directories
output_dirs = [
    path_PMB_GLAMOS_csv_w,
    path_PMB_GLAMOS_csv_w_clean,
    path_PMB_GLAMOS_csv_a,
    path_SMB_GLAMOS_csv,
    path_SMB_GLAMOS_csv_obs,
    path_pcsr_save
]

# Create all output directories
for dir_path in output_dirs:
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created directory (if it didn't exist): {dir_path}")