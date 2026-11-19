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
├── ISL_stake_inventory.csv               # Stakes and measurement counts of the paper dataset
└── scripts/
    ├── config_ICE.py                     # Raw and output paths (relative to cfg.dataPath)
    ├── iceland_preprocess.py             # split_stake_measurements, check_period_consistency, flag_elevation_mismatch
    └── stake_inventory.py                # Writes ISL_stake_inventory.csv
```

## Stake inventory (data version check)

The IMO API has no versions, so a new download may differ from the data used in the paper. `ISL_stake_inventory.csv` lists every IMO stake of the processed paper dataset (`ISL_wgms_dataset_all.csv`) with its glacier, RGI ID, number of annual and winter measurements, and first and last year: 348 stakes on 55 glaciers, 3,293 annual and 3,531 winter measurements (6,824 in total, as in Table A1 of the paper). The raw stake files used for the paper were downloaded on 10 February 2026, and the processed dataset was written on 16 February 2026.

To compare a new download, rerun `1.0` and `1.1` and then, from the repository root:

```
python -m papers.Transformer_TF.PMB_sources.RGI_06_Iceland.scripts.stake_inventory
```

and compare the rewritten `ISL_stake_inventory.csv` with the archived one. The inventory of the paper is not tracked on the development branch; it is archived in release v0.1.3.

The processed dataset renames the stakes by elevation, so the script links each measurement to its IMO stake through its coordinates, which are copied unchanged from the raw files. A few coordinates are shared by two IMO stakes; these rows list both IDs separated by `;` (e.g. `Br0;Br0a`). Some IMO stake IDs (e.g. `D12`, `L07`) occur on two glaciers and therefore appear in two rows.

## Required input data

All paths are relative to `cfg.dataPath` (`mbm.IcelandConfig()`):

- `WGMS/Iceland/raw/<stake>.csv`: one file per stake, written by `1.0`. The data used in the paper were downloaded in February 2026.
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
