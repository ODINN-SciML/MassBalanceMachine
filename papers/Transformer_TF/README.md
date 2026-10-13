# TF_Europe — multi-region data preparation and models

This folder holds the code shared by all regions: the climate and glacier-grid preparation (notebooks `1.x`), the cross-regional and pooled model experiments (notebooks `2.x`–`4.x`), and the shared helpers in `scripts/`.

This README covers the data preparation notebooks `1.1.1`–`1.4` and lists where each figure of the paper (`manuscript.tex`) is made.

## Order of the steps

The `1.x` notebooks interleave with the regional stake (PMB) preprocessing notebooks in `PMB_sources/RGI_*/`:

```
1.1.1 / 1.1.2  ERA5-Land download ───────────────────────────────┐
                                                                 │
1.2  OGGM grids + DEM GeoTIFFs                                   │
 └─> 1.3  Sky-view factor (rvt env)                              │
      └─> 1.4 part 1: masked glacier grids                       │
           └─> PMB preprocessing (PMB_sources/RGI_*/1.1*)        │
                └─> 1.4 part 2: monthly glacier grids  <─────────┘
```

1. **1.1.1 and 1.1.2**: download the ERA5-Land climate data. These can run at any time before 1.4 part 2.
2. **1.2 → 1.3 → 1.4 part 1**: build one grid per glacier with topography and sky-view factor (SVF).
3. **Regional PMB preprocessing**: each region's `1.1*-prepro` notebook samples the SVF at its stakes from the grids made in step 2 and writes `WGMS/<region>/csv/*_wgms_dataset_all.csv`.
4. **1.4 part 2**: builds monthly inputs over whole glaciers. It needs the stake datasets from step 3 and the ERA5-Land files from step 1.

All paths below are relative to the data root `cfg.dataPath`.

## Notebooks

### 1.1.1 ERA5Land-prepro_HMA / 1.1.2 ERA5Land-prepro_US_CA_EU

These download monthly ERA5-Land data from the Copernicus Climate Data Store. Both notebooks are identical apart from the area:

| Notebook | Area name | Box (N, W, S, E) | Used for RGI regions |
|---|---|---|---|
| 1.1.1 | `HIGH_MOUNTAIN_ASIA` | 45, 60, 25, 105 | 13–15 |
| 1.1.2 | `EU_US_CANADA` | 82, −170, 42, 40 | 01, 02, 06, 07, 08, 11 |

Each notebook:
1. Downloads monthly means for 1950–2025 of 14 variables: wind, 2 m temperature, albedo, snow variables, latent/sensible heat fluxes, net thermal and downward solar radiation, and precipitation.
2. Downloads the geopotential separately. It is used to get the elevation of each ERA5 cell.
3. Renames `valid_time` → `time`, converts the geopotential longitudes to [−180, 180), and removes the zips and temporary files.

**Outputs** (`ERA5Land/raw/`): `era5_monthly_averaged_data_<AREA>.nc` and `era5_geopotential_pressure_<AREA>.nc`.

**Setup**: you need an ECMWF account, the ERA5-Land licence accepted, and a `~/.cdsapirc` file. The links are in the notebook.

### 1.2 Create_geotiffs

This is the first step of the glacier grids. It runs in the MBM environment. For each region in `RGI_REGIONS` (`scripts/config_TF_Europe.py`):
1. Downloads the preprocessed OGGM glacier directories (OGGM v1.6, `elev_bands_w_data`) and exports each glacier's gridded data (topography, aspect, slope, glacier mask, ...). The output is `OGGM/rgi_region_XX/xr_grids/<RGIId>.zarr`.
2. Exports each glacier's DEM (`topo`) as a GeoTIFF in its projected, metric CRS. The output is `RGI_v6/<region folder>/geotiff/<RGIId>.tif`.

The `geotiff/` folder is emptied before each run. Glaciers are not restricted to those with stakes; that option is commented out.

### 1.3 Create_skyview_factor

This computes terrain visibility variables from the GeoTIFFs with the Relief Visualization Toolbox (RVT). It runs in the separate `rvt` environment, because RVT is not installed in the MBM environment.

For each DEM, it computes on the projected grid:
- `svf`: sky-view factor (16 directions, 10-pixel radius)
- `asvf`: anisotropic SVF
- `opns`: positive openness

It then reprojects these to WGS84 lat/lon.

**Output**: `RGI_v6/<region folder>/svf_nc_latlon/<RGIId>_svf_latlon.nc`. Each region's output folder is emptied before it is recomputed.

Because the MBM config can't be imported in the `rvt` environment, this notebook hard-codes the data root (`cfg_datapath`) and keeps its own copy of `RGI_REGIONS`. Keep that copy in sync with the config. The regions to compute are set in `RERUN_REGIONS`.

### 1.4 Create_grids

This notebook has two independent parts.

**Part 1: masked glacier grids** (run before the regional PMB preprocessing). `run_all_regions` (`scripts/dataset/create_grids.py`) processes each glacier as follows:
1. Masks the OGGM grid to the glacier outline.
2. Coarsens it according to the glacier extent: native resolution below 500 m, 50 m below 5 km, 100 m below 20 km, and 200 m otherwise.
3. Reprojects it to lat/lon.
4. Merges `svf`, `asvf` and `opns`, interpolating them to the grid if needed, and adds glacier-masked versions (`masked_svf`, ...).
5. Saves the result to `RGI_v6/<region folder>/xr_masked_grids/<RGIId>.zarr`.

Regions without OGGM grids or SVF are skipped. The cell is off by default (`RUN = False`). Before it, a cell prints per-glacier grid statistics, cached in `OGGM/grid_stats_cache/`.

**Part 2: monthly glacier grids** (run after the regional PMB preprocessing). These grids let the MBM predict PMB over whole glaciers:
1. Loads the stake datasets of all regions and builds the monthly stake data per region code (`build_monthly_cache`). The results are cached in `WGMS/CrossRegional/monthly_cache/`, and `run_flag_by_code` selects whether to recompute or load them. This step also splits Alaska, Central Asia and Italy/Austria into subregions, and sets the month padding (`months_head_pad`, `months_tail_pad`) shared with model training.
2. Computes a bounding box per region from the grid statistics and clips ERA5-Land to it. The clipped files go to `ERA5Land/clipped_to_RGI/`.
3. For each glacier and hydrological year (October start, 2000–2024), combines the monthly ERA5-Land variables with the static topography (`aspect`, `slope`, `svf`). The output is `RGI_v6/<region folder>/monthly_parquets/<RGIId>/<RGIId>_grid_<year>.parquet`. With `build_ds=True`, a model-ready dataset is also cached in `grid_ds_cache/`, and logs go to `logs/errors/`.

The grids can be run in three ways:
- for glaciers with stakes (`run_for_stake_glaciers`)
- for named glaciers (`run_for_glaciers`)
- for all glaciers larger than 1 km² (`run_all_glacier_grids`)

## Paper figures

Where each figure in `manuscript.tex` is made. Notebooks save to `figures/paperTF/` (appendix figures to `figures/paperTF/appendix/`), under the file name used in the paper unless **Saved as** says otherwise. Cell numbers count all cells (markdown included) from 0; the section is the markdown heading above the cell.

Figures marked *assembled* are combined outside the code from the listed panels or legends. Figures 2 and A1 are schematics that are not made by any code in this folder.

### Main text

| Fig. | Paper file | Notebook | Section (cell) | Saved as |
|---|---|---|---|---|
| 1 | `fig1_study_regions.png` | `2.2 Study_regions` | (a) Alaska (13), (b) Svalbard (11), (c) Norway (7), (d) Iceland (9), (e) Central European Alps (5), (f) Central Asia (15) | *assembled* from `fig1_<a–f>_maps_<region>_gl.pdf`, `fig1_e_maps_CEU_gl_legend.pdf` and `fig1_f_maps_CA_legend_color.pdf` |
| 2 | `fig2_transformer_architecture.pdf` | — | — | schematic, not made in code |
| 3 | `fig3_within_region_transformer_only.png` | `2.3 Within_region` | Fig. 3: Transformer only (26) | |
| 4 | `fig4_zeroshot_comparison.pdf` | `2.4 Cross_region` | Combined figure (30) | *assembled* from `fig_zeroshot_combined.png` and `fig_zeroshot_combined_legend.png` |
| 5 | `fig5_zs_pooled_vs_ss_barplot_a.png` | `3.2 Pooled_model` | Annual (29) | |
| 6 | `fig6_pooled_ft_gain_barplot.png` | `3.2 Pooled_model` | Annual (46) | |
| 7 | `fig7_application.png` | `4.1 Pooled_model_pred` | Specific glaciers (60) | |

### Appendix

| Fig. | Paper file | Notebook | Section (cell) | Saved as |
|---|---|---|---|---|
| A1 | `A1_diff_LSTM_Transformer.pdf` | — | — | schematic, not made in code |
| A2 | `A2_within_region_lstm_vs_transf_mixed.png` | `2.3 Within_region` | Fig. A2: LSTM vs Transformer (24) | |
| A3 | `A3_CA3_per_gl.png` | (a) `2.3 Within_region`; (b) `2.2 Study_regions` | (a) Fig. A3a: Central Asia sub-region 3 by glacier (28); (b) Appendix A3 (b) (17) | *assembled* from (a) `A3_a.png` and `A3_a_legend.png`, (b) `A3_b.png` |
| A4 | `A4_random_vs_smart_split_transformer.pdf` | `2.3 Within_region` | Fig. A4: Sinkhorn-informed vs random splits (31) | `A4_random_vs_smart_split_transformer.png` (PNG, the paper uses a PDF) |
| A5 | `A5_zs_comparison_absolute.png` | `2.4 Cross_region` | Barplot (28) | |
| A6 | `A6_zs_pooled_vs_ss_barplot_w.png` | `3.2 Pooled_model` | Winter (31) | |
| A7 | `A7_zs_pooled_vs_ss_scatter_all.png` | `3.2 Pooled_model` | Scatter (33) | |
| A8 | `A8_sensitivity_FT.png` | `3.2 Pooled_model` | Sensitivity to the fine-tuning fraction (52) | `app_ft_fraction_sensitivity_annual.png` |
| A9 | `A9_FT_loss_curves.png` | `3.2 Pooled_model` | Sensitivity to the fine-tuning fraction (53) | `app_ft_loss_curves_10pct.png` |
| A10 | `A10_ds_pooled_gain.png` | `4.2 Distance` | Annual gain (34) | |
| A11 | `A11_gradients_CH.png` | `4.1 Pooled_model_pred` | Gradients all (70) | |
| A12 | `A12_gradients_CEU.png` | `4.1 Pooled_model_pred` | Gradients all (71) | |
| A13 | `A13_gradients_NOR_SJM.png` | `4.1 Pooled_model_pred` | Gradients all (74) | |
| A14 | `A14_gradients_ISL.png` | `4.1 Pooled_model_pred` | Gradients all (73) | |
| A15 | `A15_gradients_CA.png` | `4.1 Pooled_model_pred` | Gradients all (72) | |
| A16 | `A16_gradients_ALA.png` | `4.1 Pooled_model_pred` | Gradients all (69) | |

## Known issues

- **RGI 02 (Western Canada & USA) is not set up.** It is missing from `RGI_REGIONS` in `scripts/config_TF_Europe.py` and from the copy in 1.3, so no SVF or masked grids exist for it. The SVF step of `RGI_01_02_US_CA/1.1. US_CA-prepro.ipynb` will fail for region 02 until it is added.
- **1.4 "Run all RGIs"** uses `RGI_REGIONS_TEST`, which isn't defined anywhere.
- **The last cell of 1.4 overwrites `RGI_REGIONS`** with a subset. Restart the kernel or re-import the config before running other cells after it.

## `scripts/`

These are the helper modules imported by the notebooks:
- `config_TF_Europe.py`: paths, `RGI_REGIONS`, variable lists
- `oggm/`: OGGM glacier directories and grid export, and sampling of OGGM variables at stakes
- `pmb_processing/`: stake QC, RGI ID assignment and SVF sampling, used by the regional preprocessing notebooks
- `svf_processing/`: SVF computation (needs the `rvt` environment)
- `geodata/`: GeoTIFF export
- `dataset/`: stake loading, monthly data and grid creation
- `models/`, `plotting/`, `utils/`
