# RGI 11 — French Alps

This folder builds the French point mass balance (PMB) dataset from the GLACIOCLIM observatory data.

| RGI region | Output file |
|---|---|
| 11 (France) | `FR_wgms_dataset_all.csv` in `cfg.dataPath/WGMS/France/csv/` |

## Contents

```
RGI_11_French_Alps/
├── 1.1 GLACIOCLIM-prepro.ipynb    # Preprocessing notebook
└── scripts/
    ├── config_FR.py               # Data paths (relative to cfg.dataPath)
    └── glacioclim_preprocess.py   # GLACIOCLIM readers, Lambert → WGS84, WGMS formatting, QC checks
```

## Required input data

All paths are relative to `cfg.dataPath` (`mbm.FranceConfig()`):

- `GLACIOCLIM/point/`: the GLACIOCLIM zip archives (one folder per glacier and season). Set `read_in = True` once to unzip them into `GLACIOCLIM/point/unzipped/`.
- `GLACIOCLIM/point/Glacier de SARENNES_discontinued/B_SARENNES_94-20.xls`: Sarennes, one sheet per year (needs `xlrd`).
- `GLACIOCLIM/point/Glacier BLANC_unusable/GlacierBlanc2000_2024_pointMB_mwe.xls`: Glacier Blanc annual balances.
- `RGI_v6/RGI_11_CentralEurope/xr_masked_grids/<RGIId>.zarr`: per-glacier grids with SVF, from `1.2`–`1.4 Create_grids` (part 1).

## What the notebook does

1. Read all GLACIOCLIM CSV files (glacier and period come from the file name), plus Sarennes and Glacier Blanc.
2. Convert coordinates to WGS84: Lambert II for Mont-Blanc (`mdg_*`) and Argentière files, Lambert III for all others. Three Saint-Sorlin files (`stso_winter_smb_accu_2018–2020`) have columns misnamed `*_lambert2e` but are Lambert III.
3. Bring all files into the WGMS/MBM format.
4. Keep annual and winter balances and merge stakes closer than 10 m.
5. Drop rows with NaN values, zero elevation or balance, balances above 5 or below −15 m w.e., and two faulty measurements.
6. Assign RGI IDs from the RGI 11 outlines and rename glaciers with the RGI names (names shared by several RGI IDs get the RGI ID appended).
7. Sample OGGM topography and drop stakes outside their RGI outline or without topography.
8. Add the SVF (nearest valid pixel, search radius 30 cells).
9. Rename stakes by elevation; relabel winter balances ending in June with a negative value as annual. No measurements are removed based on their months.
10. Save `FR_wgms_dataset_all.csv`.
