# RGI 08 — Norway

This folder builds the Norwegian point mass balance (PMB) dataset, from the NVE glaciological point mass balance data (https://doi.org/10.58059/sjse-6w92).

| RGI region | Output file |
|---|---|
| 08 (Norway only) | `NOR_wgms_dataset_all.csv` in `cfg.dataPath/WGMS/Norway/csv/` |

## Contents

```
RGI_08_Norway/
├── 1.1. Norway-prepro.ipynb   # Preprocessing notebook
└── scripts/
    ├── config_NOR.py          # Data paths (relative to cfg.dataPath)
    └── norway_preprocess.py   # fill_missing_dates, split_stake_measurements, check_period_consistency,
                               # fix_january_to_october_dates, flag_elevation_mismatch
```

## Required input data

All paths are relative to `cfg.dataPath` (`mbm.NorwayConfig()`):

- `WGMS/Norway/raw/glaciological_point_mass_balance_Norway.csv`: NVE stake data (already contains RGI IDs).
- `RGI_v6/RGI_08_Scandinavia/xr_masked_grids/<RGIId>.zarr`: per-glacier grids with SVF, from `1.2`–`1.4 Create_grids` (part 1).

## What the notebook does

1. Fill missing start dates with 1 October of the previous year, then split each stake-year into a winter row (`prev_yr_min_date` → `curr_yr_max_date`, `bw`) and an annual row (`prev_yr_min_date` → `curr_yr_min_date`, `ba`).
2. Name glaciers from the RGI outlines.
3. Fix inconsistent periods: months entered as 01 instead of 10, a list of manual fixes, and remaining wrong `FROM_DATE` years. Each change is recorded in `DATA_MODIFICATION`.
4. Merge stakes closer than 10 m.
5. Sample OGGM topography; drop stakes outside their RGI outline or with missing `aspect`, `slope` or `topo`.
6. Add the SVF (nearest valid pixel, search radius 30 cells).
7. Rename stakes by elevation.
8. Clean months: remove measurements starting or ending in July or December; relabel winter balances ending in June with a negative value as annual.
9. Save `NOR_wgms_dataset_all.csv`.
