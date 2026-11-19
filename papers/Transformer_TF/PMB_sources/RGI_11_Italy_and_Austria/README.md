# RGI 11 — Italy and Austria

This folder builds the Italian and Austrian point mass balance (PMB) dataset from the WGMS Fluctuations of Glaciers database (version 10.5904/wgms-fog-2025-02b). The data start in 2006.

| RGI region | Output file |
|---|---|
| 11 (Italy, Austria) | `IT_AT_wgms_dataset_all.csv` in `cfg.dataPath/WGMS/Italy_Austria/csv/` |

## Contents

```
RGI_11_Italy_and_Austria/
├── 1.1. WGMS_IT_AT-prepro.ipynb       # Preprocessing notebook
└── scripts/
    └── italy_austria_preprocess.py    # fix_uncertain_dates, check_period_consistency
```

## Required input data

All paths are relative to `cfg.dataPath` (`mbm.ItalyAustriaConfig()`). The notebook sets its paths itself:

- `WGMS/raw/mass_balance_point.csv` and `WGMS/raw/glacier.csv`: WGMS point balances and glacier metadata.
- `RGI_v6/RGI_11_CentralEurope/11_rgi60_CentralEurope.shp`: RGI outlines (used to find the RGI ID of Careser).
- `RGI_v6/RGI_11_CentralEurope/xr_masked_grids/<RGIId>.zarr`: per-glacier grids with SVF, from `1.2`–`1.4 Create_grids` (part 1).

## What the notebook does

1. Keep WGMS stakes from Austria and Italy and map them to RGI IDs. The four Careser entries only have an RGI 5 ID and get `RGI60-11.01834`.
2. Fix one wrong elevation (Vernagt F., 296 → 2960 m) and dates with year 1012 instead of 2012 (Malavalle).
3. Set dates with an uncertainty of ~182 days to the hydrological year (1 Oct start, 30 Sep annual end, 30 Apr winter end), and fix two inconsistent periods by hand.
4. Sample OGGM topography and drop stakes outside their RGI outline.
5. Keep annual and winter balances and merge stakes closer than 10 m.
6. Add the SVF (nearest valid pixel, search radius 30 cells).
7. Rename stakes by elevation.
8. Final cleaning: remove winter balances above 2 m w.e. on Lupo and winter measurements ending in July; relabel winter balances ending in June with a negative value as annual.
9. Save `IT_AT_wgms_dataset_all.csv`.
