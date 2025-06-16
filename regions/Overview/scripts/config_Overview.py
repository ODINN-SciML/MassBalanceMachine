# <------------------ GLACIOCLIM DATA: ------------------>
# Point data
#path_PMB_GLACIOCLIM_raw = '/home/mburlet/scratch/data/DATA_MB/GLACIOCLIM/unzipped/'
#path_PMB_GLACIOCLIM_w_raw = path_PMB_GLACIOCLIM_raw + 'winter/'
#path_PMB_GLACIOCLIM_a_raw = path_PMB_GLACIOCLIM_raw + 'annual/'
#path_PMB_GLACIOCLIM_s_raw = path_PMB_GLACIOCLIM_raw + 'summer/'

#path_PMB_GLACIOCLIM_csv = '/home/mburlet/scratch/data/DATA_MB/GLACIOCLIM/WGMS_all/'
#path_PMB_GLACIOCLIM_csv_w = path_PMB_GLACIOCLIM_csv + 'winter/'
#path_PMB_GLACIOCLIM_csv_w_clean = path_PMB_GLACIOCLIM_csv + 'winter_clean/'
#path_PMB_GLACIOCLIM_csv_a = path_PMB_GLACIOCLIM_csv + 'annual/'

# Glacier wide data
#path_SMB_GLACIOCLIM_raw = '/home/mburlet/scratch/data/DATA_MB/GLACIOCLIM/glacier-wide/raw/'
#path_SMB_GLACIOCLIM_csv = '/home/mburlet/scratch/data/DATA_MB/GLACIOCLIM/glacier-wide/csv/'

# Gridded data for MBM to use for making predictions over whole grid (SGI or RGI grid)
#path_glacier_grid_rgi = '/home/mburlet/scratch/data/DATA_MB/GLACIOCLIM/topo/gridded_topo_inputs/RGI_grid/' # DEMs & topo
#path_glacier_grid_sgi = '/home/mburlet/scratch/data/DATA_MB/GLACIOCLIM/topo/gridded_topo_inputs/SGI_grid/' # DEMs & topo 
#path_glacier_grid_GLACIOCLIM = '/home/mburlet/scratch/data/DATA_MB/GLACIOCLIM/topo/gridded_topo_inputs/GLACIOCLIM_grid/'

# Topo data
#path_SGI_topo = '/home/mburlet/scratch/data/DATA_MB/GLACIOCLIM/topo/SGI2020/' # DEMs & topo from SGI
#path_GLACIOCLIM_topo = '/home/mburlet/scratch/data/DATA_MB/GLACIOCLIM/topo/GLACIOCLIM_DEM/' # yearly DEMs from GLACIOCLIM
#path_pcsr = '/home/mburlet/scratch/data/DATA_MB/GLACIOCLIM/topo/pcsr/' # Potential incoming clear sky solar radiation from GLACIOCLIM

#path_distributed_MB_GLACIOCLIM = '/home/mburlet/scratch/data/DATA_MB/GLACIOCLIM/distributed_MB_grids/'
#path_geodetic_MB_GLACIOCLIM = '/home/mburlet/scratch/data/DATA_MB/GLACIOCLIM/geodetic/'
#path_glacier_ids = '/home/mburlet/scratch/data/DATA_MB/GLACIOCLIM/FR_glacier_ids_long.csv' # glacier ids for CH glaciers

# <------------------ OTHER PATHS: ------------------>
#path_ERA5_raw = '/home/mburlet/scratch/data/DATA_MB/ERA5Land/raw/' # ERA5-Land
#path_S2 = '/home/mburlet/scratch/data/DATA_MB/Sentinel/' # Sentinel-2
#path_OGGM = '/home/mburlet/scratch/data/DATA_MB/OGGM/'
#path_glogem = '/home/mburlet/scratch/data/DATA_MB/GloGEM' # glogem c_prec and t_off factors
#path_rgi_outlines = '/home/mburlet/scratch/data/DATA_MB/GLACIOCLIM/RGI/nsidc0770_11.rgi60.CentralEurope/11_rgi60_CentralEurope.shp'

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