import os
import platform

def get_base_path():
    """Determine if running on local or SSH and return appropriate base path"""
    if os.name == 'nt' or platform.system() == 'Windows':  # Windows/local
        return r'D:\MBM_data'  
    elif os.name == 'posix' or platform.system() == 'Linux':  # Linux/SSH
        return '/home/mburlet/scratch/data/DATA_MB'

# Set base path based on environment
BASE_PATH = get_base_path()

# <------------------ GLACIOCLIM DATA: ------------------>
# Point data
path_PMB_GLACIOCLIM_raw = os.path.join(BASE_PATH, 'GLACIOCLIM', 'unzipped')
path_PMB_GLACIOCLIM_w_raw = os.path.join(path_PMB_GLACIOCLIM_raw, 'winter')
path_PMB_GLACIOCLIM_a_raw = os.path.join(path_PMB_GLACIOCLIM_raw, 'annual')
path_PMB_GLACIOCLIM_s_raw = os.path.join(path_PMB_GLACIOCLIM_raw, 'summer')

path_PMB_GLACIOCLIM_csv = os.path.join(BASE_PATH, 'GLACIOCLIM', 'WGMS_all')
path_PMB_GLACIOCLIM_csv_w = os.path.join(path_PMB_GLACIOCLIM_csv, 'winter')
path_PMB_GLACIOCLIM_csv_w_clean = os.path.join(path_PMB_GLACIOCLIM_csv, 'winter_clean')
path_PMB_GLACIOCLIM_csv_a = os.path.join(path_PMB_GLACIOCLIM_csv, 'annual')

# Glacier wide data
path_SMB_GLACIOCLIM_raw = os.path.join(BASE_PATH, 'GLACIOCLIM', 'glacier-wide', 'raw')
path_SMB_GLACIOCLIM_csv = os.path.join(BASE_PATH, 'GLACIOCLIM', 'glacier-wide', 'csv')

# Gridded data for MBM to use for making predictions over whole grid (SGI or RGI grid)
path_glacier_grid_rgi = os.path.join(BASE_PATH, 'GLACIOCLIM', 'topo', 'gridded_topo_inputs', 'RGI_grid')
path_glacier_grid_sgi = os.path.join(BASE_PATH, 'GLACIOCLIM', 'topo', 'gridded_topo_inputs', 'SGI_grid')
path_glacier_grid_GLACIOCLIM = os.path.join(BASE_PATH, 'GLACIOCLIM', 'topo', 'gridded_topo_inputs', 'GLACIOCLIM_grid')

# Topo data
path_SGI_topo = os.path.join(BASE_PATH, 'GLACIOCLIM', 'topo', 'SGI2020')
path_GLACIOCLIM_topo = os.path.join(BASE_PATH, 'GLACIOCLIM', 'topo', 'GLACIOCLIM_DEM')
path_pcsr = os.path.join(BASE_PATH, 'GLACIOCLIM', 'topo', 'pcsr')

path_distributed_MB_GLACIOCLIM = os.path.join(BASE_PATH, 'GLACIOCLIM', 'distributed_MB_grids')
path_geodetic_MB_GLACIOCLIM = os.path.join(BASE_PATH, 'GLACIOCLIM', 'geodetic')
path_glacier_ids = os.path.join(BASE_PATH, 'GLACIOCLIM', 'FR_glacier_ids_long.csv')

# <------------------ OTHER PATHS: ------------------>
path_ERA5_raw = os.path.join(BASE_PATH, 'ERA5Land', 'raw')
path_S2 = os.path.join(BASE_PATH, 'Sentinel')
path_OGGM = os.path.join(BASE_PATH, 'OGGM')
path_glogem = os.path.join(BASE_PATH, 'GloGEM')
path_rgi_outlines = os.path.join(BASE_PATH, 'GLACIOCLIM', 'RGI', 'nsidc0770_11.rgi60.CentralEurope', '11_rgi60_CentralEurope.shp')


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