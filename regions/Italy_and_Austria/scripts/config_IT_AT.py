# <------------------ PATHS ------------------>
path_PMB_WGMS_raw = '/home/mburlet/scratch/data/DATA_MB/WGMS/IT_AT/data/'
path_PMB_WGMS_csv = '/home/mburlet/scratch/data/DATA_MB/WGMS/IT_AT/csv/'
path_ERA5_raw = '/home/mburlet/scratch/data/DATA_MB/ERA5Land/raw/'  ###ATTENTION this uses ERA5 from other notebook
path_OGGM = '/home/mburlet/scratch/data/DATA_MB/WGMS/IT_AT/OGGM/'
path_OGGM_xrgrids = '/home/mburlet/scratch/data/DATA_MB/WGMS/IT_AT/OGGM/xr_grids/'



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
