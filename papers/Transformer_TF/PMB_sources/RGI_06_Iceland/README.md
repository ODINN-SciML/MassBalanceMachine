# RGI 06 — Iceland

This folder builds the Icelandic point mass balance (PMB) dataset, from the stake data of the Icelandic Met Office (IMO).

| RGI region | Output file |
|---|---|
| 06 | `ISL_wgms_dataset_all.csv` in `cfg.dataPath/WGMS/Iceland/csv/` |

## Contents

```
RGI_06_Iceland/
├── 1.0. Iceland-data-acquisition.ipynb   # Download of the raw stake files from the IMO API
├── 1.1. Iceland-prepro.ipynb             # Preprocessing notebook
└── scripts/
    ├── config_ICE.py                     # Raw and output paths (relative to cfg.dataPath)
    └── iceland_preprocess.py             # split_stake_measurements, check_period_consistency, flag_elevation_mismatch
```

## Required input data

All paths are relative to `cfg.dataPath` (`mbm.IcelandConfig()`):

- `WGMS/Iceland/raw/<stake>.csv`: one file per stake, written by `1.0`. The data used in the paper were downloaded in June 2025.
- `RGI_v6/RGI_06_Iceland/xr_masked_grids/<RGIId>.zarr`: per-glacier grids with SVF, from `1.2`–`1.4 Create_grids` (part 1).

OGGM glacier directories and RGI outlines are downloaded by OGGM on the first run.

## What the notebooks do

**1.0** gets the list of regular and irregular stakes from the IMO GeoServer and downloads the measurements of each stake from `https://api.vedur.is/glaciers/stake/<id>/measurements`.

**1.1**:
1. Read all stake files and split each stake-year into a winter row (`d1` → `d2`, `bw_floating_date`) and an annual row (`d1` → `d3`, `ba_floating_date`).
2. Fill missing dates with the hydrological year (1 Oct – 30 Sep) and fix the start year of stake GL10a.
3. Assign RGI IDs by intersecting the stakes with the RGI 06 outlines (stakes outside any outline are dropped); name glaciers from RGI, or by the RGI ID if RGI has no name.
4. Sample OGGM topography and drop stakes outside their RGI outline.
5. Merge stakes closer than 10 m. Stakes whose elevation differs from OGGM by more than 400 m are listed, not removed.
6. Add the SVF (nearest valid pixel, search radius 30 cells).
7. Rename stakes by elevation.
8. Clean months: remove measurements starting or ending in July, December or January, and winter measurements ending in August; relabel winter balances ending in June with a negative value as annual.
9. Save `ISL_wgms_dataset_all.csv`.
