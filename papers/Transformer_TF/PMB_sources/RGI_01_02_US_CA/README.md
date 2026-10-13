# RGI 01 — Alaska

This folder builds the point mass balance (PMB) dataset for Alaskan glaciers (RGI v6 region 01, GTN-G region `01_alaska`) used by MassBalanceMachine. The output is `ALA_wgms_dataset_all.csv`.

The folder name still says `01_02_US_CA`, but Western Canada & USA (RGI 02) is not processed: it is not used in the paper and is not set up in the glacier grid pipeline (`RGI_REGIONS`). Arctic Canada (RGI 03/04) is not included either.

## Contents

```
RGI_01_02_US_CA/
├── 1.1. US_CA-prepro.ipynb   # Preprocessing notebook (the whole pipeline)
├── 01_Alaska.csv             # Extra Alaskan stake data not in WGMS
└── scripts/
    └── config_US_CA.py       # Output path (relative to cfg.dataPath)
```

- **`1.1. US_CA-prepro.ipynb`**: the main notebook. It reads the raw stake data, adds topographic and sky-view factor variables, cleans the measurement periods and writes the Alaska CSV.
- **`01_Alaska.csv`**: additional Alaskan stake measurements (`;`-separated, latin1 encoded, dates as `DD.MM.YYYY`) that are not in the WGMS database. It uses the WGMS column layout but has no glacier/RGI IDs. Sources are given per row in the `source` column (e.g. USGS, https://doi.org/10.5066/F7MP51CB).
- **`scripts/config_US_CA.py`**: the output folder `path_PMB_WGMS_csv`, relative to the data root `cfg.dataPath`.

`01_Alaska.csv` matches the `papers/**/*.csv` rule in `.gitignore`, so it is not tracked by git. Add it with `git add -f` if it should be versioned.

## Required input data

All paths are relative to `cfg.dataPath`, as set by `mbm.USCANADAConfig()`:

- `WGMS/raw/mass_balance_point.csv`: WGMS point mass balance measurements
- `WGMS/raw/glacier.csv`: WGMS glacier metadata (WGMS id → RGI60 id, GTN-G region)
- `RGI_v6/RGI_01_Alaska/01_rgi60_Alaska.shp`: RGI v6 outlines for region 01 (used to assign RGI IDs to `01_Alaska.csv`)
- `RGI_v6/RGI_01_Alaska/xr_masked_grids/<RGIId>.zarr`: per-glacier grids that include the sky-view factor. They are built by three notebooks in `papers/Transformer_TF/`:
  1. `1.2 Create_geotiffs.ipynb`: OGGM grids and DEM GeoTIFFs (`RGI_v6/<region>/geotiff/`)
  2. `1.3 Create_skyview_factor.ipynb`: SVF from the DEMs (`RGI_v6/<region>/svf_nc_latlon/`). This one runs in the separate `rvt` environment.
  3. `1.4 Create_grids.ipynb`, "Masked grids" section only: combines the two into `xr_masked_grids/`

The notebook downloads OGGM glacier directories (OGGM v1.6, `elev_bands_w_data`) itself, so it needs internet access on the first run.

## What the notebook does

1. **Load WGMS data.** Map each WGMS `glacier_id` to its RGI60 ID and GTN-G region. Keep stakes from the US and Canada in region `01_alaska`. Drop stakes without an RGI ID or coordinates, and rename columns to the MBM convention (`POINT_ID`, `POINT_LAT`, `POINT_LON`, `POINT_ELEVATION`, `FROM_DATE`, `TO_DATE`, `POINT_BALANCE`, `GLACIER`, `PERIOD`, `RGIId`, `YEAR`).
2. **Add extra Alaskan stakes** from `01_Alaska.csv`. RGI IDs are assigned by intersecting the stake coordinates with the RGI 01 outlines. Summer balances, rows without valid dates, and rows whose period does not span exactly one year boundary are removed. The result is concatenated with the WGMS stakes.
3. **Add OGGM topography.** Initialise the OGGM glacier directories, export their gridded data, and sample `aspect`, `slope` and `topo` (elevation) at each stake. Stakes outside their RGI glacier outline are dropped.
4. **Merge close stakes.** Keep annual and winter balances (WGMS summer and `index` balances are dropped), then merge stakes less than 10 m apart within a glacier, year and period into one point with the mean balance.
5. **Add sky-view factor (SVF)** from the per-glacier zarr grids. Stakes on masked pixels are filled with the nearest valid pixel (search radius of up to 60 cells).
6. **Re-assign stake IDs** per glacier as `<GLACIER>_<rank>`, ranked by elevation.
7. **Clean measurement dates** so that they match the hydrological year:
   - keep years after 1951 and drop summer balances;
   - shift annual/winter periods starting in July to 1 August;
   - reclassify "winter" balances ending in June/July as annual;
   - drop annual periods that do not start Aug–Dec, end Jun–Oct, and last ≤ 460 days;
   - drop winter periods that do not start Aug–Dec, end Jan–May, and last ≤ 430 days.
8. **Save** `ALA_wgms_dataset_all.csv` to `cfg.dataPath/WGMS/WesternCanadaUS/csv/`, and plot overviews: a histogram of the balances, an interactive folium map of the stakes, and the number of measurements per year and per glacier.

## Shared code

The notebook uses helpers from `papers/Transformer_TF/scripts/`, which are shared across regions:

- `oggm/`: `initialize_oggm_glacier_directories`, `export_oggm_grids`, `merge_pmb_with_oggm_data`
- `plotting/folium.py`: `plot_stakes_folium` (interactive stake map)
- `pmb_processing/`: stake QC (`remove_close_points`, `rename_stakes_by_elevation`, `check_point_ids_contain_glacier`), RGI ID assignment (`add_rgi_ids_to_df`) and SVF sampling (`add_svf_from_rgi_zarr`, `add_svf_nearest_valid`, `plot_missing_svf_for_all_glaciers`)

## Running

Run the notebook from inside this folder, because it adds the repository root (`../../../../`) to `sys.path` and reads `01_Alaska.csv` by relative path. The outputs are read by `papers/Transformer_TF/scripts/dataset/data_loader.py` (code `ALA`).
