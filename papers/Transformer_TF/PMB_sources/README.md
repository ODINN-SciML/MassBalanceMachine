# PMB_sources — regional stake (PMB) preprocessing

Each folder builds the point mass balance (PMB) dataset of one or more RGI v6 regions. The outputs are read by `scripts/dataset/data_loader.py` (`STAKES_SPECS`) and used from notebook `1.4 Create_grids` (part 2) onwards.

Run these notebooks after `1.4 Create_grids` part 1 (masked glacier grids with SVF) and before `1.4` part 2; see the README in the parent folder.

| Folder | Notebook(s) | Source | Output (under `cfg.dataPath`) |
|---|---|---|---|
| `RGI_01_02_US_CA` | `1.1. US_CA-prepro` | WGMS + extra Alaska file | `WGMS/WesternCanadaUS/csv/ALA_…`, `CAW_wgms_dataset_all.csv` |
| `RGI_06_Iceland` | `1.0` (download), `1.1. Iceland-prepro` | Icelandic Met Office API | `WGMS/Iceland/csv/ISL_wgms_dataset_all.csv` |
| `RGI_07_Svalbard` | `1.1. Svalbard-prepro` | WGMS | `WGMS/Svalbard/csv/SJM_wgms_dataset_all.csv` |
| `RGI_08_Norway` | `1.1. Norway-prepro` | NVE | `WGMS/Norway/csv/NOR_wgms_dataset_all.csv` |
| `RGI_11_French_Alps` | `1.1 GLACIOCLIM-prepro` | GLACIOCLIM | `WGMS/France/csv/FR_wgms_dataset_all.csv` |
| `RGI_11_Italy_and_Austria` | `1.1. WGMS_IT_AT-prepro` | WGMS | `WGMS/Italy_Austria/csv/IT_AT_wgms_dataset_all.csv` |
| `RGI_11_Switzerland` | `1.1 GLAMOS-prepro` | GLAMOS | `WGMS/Switzerland/csv/CH_wgms_dataset_all.csv` |
| `RGI_13_14_15_HMA` | `1.1. Stakes-prepro_HMA` | WGMS | `WGMS/HMA/csv/CentralAsia_…`, `SouthAsiaWest_…`, `SouthAsiaEast_wgms_dataset_all.csv` |

## Common steps

All regions except Switzerland follow the same pattern, using the shared helpers in `papers/Transformer_TF/scripts/`:

1. Read the raw stakes and rename the columns to the MBM convention (`POINT_ID`, `POINT_LAT`, `POINT_LON`, `POINT_ELEVATION`, `FROM_DATE`, `TO_DATE`, `POINT_BALANCE`, `GLACIER`, `PERIOD`, `RGIId`, `YEAR`).
2. Keep annual and winter balances only.
3. Sample `aspect`, `slope` and `topo` from OGGM glacier directories (`oggm/merge_pmb_with_oggm_data`) and drop stakes outside their RGI outline.
4. Merge stakes closer than 10 m in the same glacier, year and period (`pmb_processing/remove_close_points`).
5. Sample the sky-view factor from `RGI_v6/<region>/xr_masked_grids/<RGIId>.zarr` (`pmb_processing/add_svf_nearest_valid`).
6. Rename stakes as `<GLACIER>_<rank>` by elevation (`pmb_processing/rename_stakes_by_elevation`).
7. Clean the measurement months so they match the hydrological year. The rules differ by region; each folder's README lists them.

Switzerland uses GLAMOS DEMs and the SGI 2020 topography instead of OGGM, and its own helper copies in `RGI_11_Switzerland/scripts/`.

## Running

Each notebook adds the repository root (`../../../../`) to `sys.path` and imports helpers as `papers.Transformer_TF.…`. Run it from its own folder in the MBM environment.
