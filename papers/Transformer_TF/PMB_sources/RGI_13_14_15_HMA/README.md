# RGI 13–15 — High Mountain Asia

This folder builds the point mass balance (PMB) datasets for High Mountain Asia from the WGMS Fluctuations of Glaciers database. Only Central Asia (RGI 13) is used in the paper, split into RGI O2 sub-regions 3 and 4 later by `add_o2region_to_dfs` (`papers/Transformer_TF/scripts/utils/glaciers.py`).

| RGI region | GTN-G region name | Output file (in `cfg.dataPath/WGMS/HMA/csv/`) |
|---|---|---|
| 13 | `13_central_asia` | `CentralAsia_wgms_dataset_all.csv` |
| 14 | `14_south_asia_west` | `SouthAsiaWest_wgms_dataset_all.csv` |
| 15 | `15_south_asia_east` | `SouthAsiaEast_wgms_dataset_all.csv` |

## Contents

```
RGI_13_14_15_HMA/
├── 1.1. Stakes-prepro_HMA.ipynb   # Preprocessing notebook
└── scripts/
    └── config_HMA.py              # Output path (relative to cfg.dataPath)
```

## Required input data

All paths are relative to `cfg.dataPath`. There is no HMA-specific MBM config; `mbm.USCANADAConfig()` only provides the data root and seed.

- `WGMS/raw/mass_balance_point.csv` and `WGMS/raw/glacier.csv`: WGMS point balances and glacier metadata (WGMS id → RGI60 id, GTN-G region).
- `RGI_v6/RGI_13_CentralAsia/`, `RGI_14_SouthAsiaWest/`, `RGI_15_SouthAsiaEast/` `xr_masked_grids/<RGIId>.zarr`: per-glacier grids with SVF, from `1.2`–`1.4 Create_grids` (part 1).

## What the notebook does

1. Keep WGMS stakes in GTN-G regions 13, 14 and 15 that have an RGI ID.
2. For each region, sample `aspect`, `slope` and `topo` from OGGM and drop stakes outside their RGI outline.
3. Keep annual and winter balances and merge stakes closer than 10 m.
4. Add the SVF (nearest valid pixel, search radius 60 cells).
5. Rename stakes by elevation.
6. Clean dates:
   - shift annual/winter periods starting in July to 1 August;
   - reclassify winter balances ending in June or July as annual;
   - drop annual periods that do not start Aug–Dec, end Jun–Oct, and last ≤ 460 days;
   - drop winter periods that do not start Aug–Dec, end Jan–May, and last ≤ 430 days.
7. Save one CSV per region.
