# RGI 07 — Svalbard

This folder builds the Svalbard point mass balance (PMB) dataset from the WGMS Fluctuations of Glaciers database.

| RGI region | Output file |
|---|---|
| 07 | `SJM_wgms_dataset_all.csv` in `cfg.dataPath/WGMS/Svalbard/csv/` |

## Contents

```
RGI_07_Svalbard/
├── 1.1. Svalbard-prepro.ipynb   # Preprocessing notebook
└── scripts/
    └── config_SVA.py            # Output path (relative to cfg.dataPath)
```

## Required input data

All paths are relative to `cfg.dataPath` (`mbm.SvalbardConfig()`):

- `WGMS/raw/mass_balance_point.csv` and `WGMS/raw/glacier.csv`: WGMS point balances and glacier metadata (WGMS id → RGI60 id).
- `RGI_v6/RGI_07_Svalbard/xr_masked_grids/<RGIId>.zarr`: per-glacier grids with SVF, from `1.2`–`1.4 Create_grids` (part 1).

## What the notebook does

1. Keep WGMS stakes with `country == 'SJ'` and map them to RGI IDs.
2. Sample OGGM topography and drop stakes outside their RGI outline.
3. Keep annual and winter balances and merge stakes closer than 10 m.
4. Add the SVF (nearest pixel; stakes on masked pixels are not filled, so check that none are missing).
5. Rename stakes by elevation.
6. Clean dates: shift July starts to 1 August and remove winter measurements ending in July.
7. Drop the helper columns and save `SJM_wgms_dataset_all.csv`.
