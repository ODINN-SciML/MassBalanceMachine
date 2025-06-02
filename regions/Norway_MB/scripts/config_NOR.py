import os
import platform

def get_base_path():
    """Determine if running on local or SSH and return appropriate base path"""
    if os.name == 'nt' or platform.system() == 'Windows':  # Windows/local
        return r'D:\MBM_data\WGMS\Norway'  # Adjust if needed
    elif os.name == 'posix' or platform.system() == 'Linux':  # Linux/SSH
        return '/home/mburlet/scratch/data/DATA_MB/WGMS/Norway'

# Set base path based on environment
BASE_PATH = get_base_path()

# <------------------ PATHS ------------------>
path_PMB_WGMS_raw = os.path.join(BASE_PATH, 'data')  # Raw Stake measurement ".csv"s
path_PMB_WGMS_csv = os.path.join(BASE_PATH, 'csv')  # Processed stake measurements
path_ERA5_raw = os.path.join(BASE_PATH, 'ERA5Land', 'raw')  # ERA5-Land
path_OGGM = os.path.join(BASE_PATH, 'OGGM')  # OGGM Data
path_OGGM_xrgrids = os.path.join(BASE_PATH, 'OGGM', 'xr_grids')  # OGGM Data Grids


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
