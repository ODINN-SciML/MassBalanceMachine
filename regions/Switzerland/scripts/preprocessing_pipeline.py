import massbalancemachine as mbm
import re
import os

# Scripts
from scripts.helpers import *
from scripts.glamos_preprocess import *
from scripts.config_CH import *

# set config file
cfg = mbm.SwitzerlandConfig()
seed_all(cfg.seed)
free_up_cuda()

# 1. Transform seasonal and winter PMB .dat files to .csv for simplicity:
print('Processing PMB .dat files to .csv:')
process_pmb_dat_files()

# 2. Assemble measurement periods:
# Annual measurements:
# Process annual measurements and put all stakes into one csv file
print('Processing annual measurements:')
df_annual_raw = process_annual_stake_data(path_PMB_GLAMOS_csv_a)

# Winter measurements:
print('Processing winter measurements:')
process_winter_stake_data(df_annual_raw, path_PMB_GLAMOS_csv_w,
                          path_PMB_GLAMOS_csv_w_clean)

# Assemble both annual and winter measurements:
print('Assembling all measurements (winter & annual):')
df_all_raw = assemble_all_stake_data(df_annual_raw,
                                     path_PMB_GLAMOS_csv_w_clean,
                                     path_PMB_GLAMOS_csv)
