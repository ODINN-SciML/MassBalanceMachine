# Transformer_TF — data preparation, models and paper figures

This folder holds everything behind the paper (`manuscript.tex`): the climate and glacier-grid preparation (notebooks `1.x`), the regional stake (PMB) preprocessing (`PMB_sources/`), the model experiments and figures (notebooks `2.x` and `3.x`), and the shared helpers in `scripts/`.

This README gives the order in which to run the notebooks, describes the data preparation notebooks `1.1.1`–`1.4`, and lists where each figure of the paper is made. Each `2.x`/`3.x` notebook explains its own steps in its first cell, and `PMB_sources/README.md` covers the stake preprocessing.

## Installation

Two conda environments are used:

| Environment | Used for | Create with |
|---|---|---|
| `MassBalanceMachine` (Python 3.11) | everything except 1.3 | `conda env create -f environment_gpu.yml` (or `environment_cpu.yml` without a GPU) in the repository root, then `conda activate MassBalanceMachine` and `poetry install` to install the `massbalancemachine` package and its dependencies (`pyproject.toml`) |
| `rvt` (Python 3.11) | `1.3 Create_skyview_factor` only (Relief Visualization Toolbox) | `conda env create -f environment_rvt.yml` in this folder |

See also the [MBM installation instructions](https://massbalancemachine.readthedocs.io/en/latest/install.html). The model notebooks (2.2–3.4) are meant to run on a GPU. Each notebook adds the repository root to `sys.path`, so run it from its own folder. The data root (`cfg.dataPath`) is set in the MBM config (`mbm.EuropeTFConfig()`); 1.3 sets it by hand (`cfg_datapath`).

## Model settings

All model settings of the paper are in `scripts/config_models.py`:

| Setting | Value | Name in `config_models.py` |
|---|---|---|
| Transformer hyperparameters | best configuration of the 3.1 search | `PARAMS_TRANSFORMER` |
| LSTM hyperparameters | | `PARAMS_LSTM` |
| Maximum training epochs (early stopping on the validation loss) | 150 | `N_EPOCHS` |
| Hyperparameter search | random search, 150 configurations sampled (seed `cfg.seed`) from the grid in 3.1, up to 150 epochs each | `GS_N_TRIALS`, `GS_MAX_EPOCHS` |
| Search validation data and selection rule | test glaciers of the within-region splits of 2.3 (about 16–24% of each source region's measurements); lowest pooled validation loss | `GS_SELECTION_METRIC`, `GS_LOG` |
| Fine-tuning data | 10% of the measurement IDs of each target region (winter and annual drawn separately); the rest is the test set | `FT_FRAC` |
| Fine-tuning repeats and seeds | 5 repeats, seeds `cfg.seed + rep` (20–24) | `FT_N_REPEATS` |
| Fine-tuning training | up to 50 epochs, learning rate × 0.1, 80/20 train/validation split of the fine-tuning data, early stopping (patience 10), best-validation checkpoint restored | `PARAMS_FT` |
| Date tag of the trained models | 2026-06-10 | `MODEL_DATE` |

## Order of the steps

```
1.1.1 / 1.1.2  ERA5-Land download ───────────────────────────────┐
                                                                 │
1.2  OGGM grids + DEM GeoTIFFs                                   │
 └─> 1.3  Sky-view factor (rvt env)                              │
      └─> 1.4 part 1: masked glacier grids                       │
           └─> PMB preprocessing (PMB_sources/RGI_*/1.1*)        │
                └─> 1.4 part 2: monthly inputs + glacier grids <─┘
                     │
                     ├─> 2.1 Study regions          (Fig. 1, A3b)
                     ├─> 2.2 Transformer grid search (not used in the paper)
                     ├─> 2.3 Within-region          (Fig. 3, A2, A3a, A4)
                     │    └─> 3.1 Multi-region grid search -> PARAMS_TRANSFORMER
                     ├─> 2.4 Cross-region           (Fig. 4, A5)
                     │    └─> 3.2 Multi-region model (Fig. 5, 6, A6–A9)
                     │         ├─> 3.3 Predictions   (Fig. 7, A11–A16)
                     │         └─> 3.4 Distance      (Fig. A10)
```

**Data preparation**
1. **1.1.1 and 1.1.2**: download the ERA5-Land climate data. These can run at any time before 1.4 part 2.
2. **1.2 → 1.3 → 1.4 part 1**: build one grid per glacier with topography and sky-view factor (SVF).
3. **Regional PMB preprocessing**: each region's `1.1*-prepro` notebook samples the SVF at its stakes from the grids made in step 2 and writes `WGMS/<region>/csv/*_wgms_dataset_all.csv`.
4. **1.4 part 2**: builds the monthly stake inputs of all regions (`WGMS/CrossRegional/monthly_cache/`), which every `2.x`/`3.x` notebook loads, and the monthly inputs over whole glaciers used for the maps in 3.3. It needs the stake datasets from step 3 and the ERA5-Land files from step 1.

**Experiments and figures** (all in the MBM environment; models and datasets are cached in `cache/TF_Europe/` and only recomputed when missing)

| Notebook | Needs | Writes (under `cache/TF_Europe/`) |
|---|---|---|
| `2.1 Study_regions` | stake datasets, RGI outlines | figures only |
| `2.2 Transformer_grid_search` | monthly inputs | `TF_REGION/glacier_splits.json` (pool/holdout glaciers, computed if missing), datasets and grid-search log |
| `2.3 Within_region` | monthly inputs | `WITHIN_REGION/` splits, datasets, models |
| `2.4 Cross_region` | monthly inputs | `CROSS_REGION/` datasets and models (one LSTM and one Transformer per source region) |
| `3.1 Pooled_model_grid_search` | monthly inputs; within-region test glaciers of 2.3 (used as validation glaciers) | `FD_MODEL/GS_results/` log; its best configuration is `PARAMS_TRANSFORMER` in `scripts/config_models.py` |
| `3.2 Pooled_model` | monthly inputs; `PARAMS_TRANSFORMER`; single-region Transformers of 2.4 | `FD_MODEL/` datasets, multi-region model (`models/foundation/`), fine-tuned models (`models/finetuned/`), fine-tuning sensitivity runs |
| `3.3 Pooled_model_pred` | models of 3.2; for new predictions, the grid datasets of 1.4 part 2 (`build_ds=True`) | `pooled_pred/predictions_transformer/` (in `cfg.dataPath/cache/`) |
| `3.4 Distance` | models of 3.2 | `FD_MODEL/domain_shifts_pooled_cache.pkl` |

The model notebooks run on a GPU. With the caches in place, 3.2–3.4 only load models. Missing fine-tuned models are trained; the multi-region and single-region models are only trained with `TRAIN_FLAG = True` in 3.2 (otherwise a missing checkpoint raises an error), so run 3.2 before 3.3 and 3.4.

All paths below are relative to the data root `cfg.dataPath`.

## Notebooks

### 1.1.1 ERA5Land-prepro_HMA / 1.1.2 ERA5Land-prepro_US_CA_EU

These download monthly ERA5-Land data from the Copernicus Climate Data Store. Both notebooks are identical apart from the area:

| Notebook | Area name | Box (N, W, S, E) | Used for RGI regions |
|---|---|---|---|
| 1.1.1 | `HIGH_MOUNTAIN_ASIA` | 45, 60, 25, 105 | 13–15 |
| 1.1.2 | `EU_US_CANADA` | 82, −170, 42, 40 | 01, 06, 07, 08, 11 |

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

The `geotiff/` folder is emptied before each run.

The OGGM step can be restricted to the glaciers with stakes: for the regions in `stakes_only`, only the glaciers in the processed stake datasets of the PMB_sources notebooks are downloaded (`stake_rgi_ids` in `scripts/dataset/data_loader.py`). As set now (`stakes_only = RGI_REGIONS.keys()`), every region is restricted to its stake glaciers. On a fresh setup the stake datasets don't exist yet, so all glaciers of each region are processed; rerun 1.2 after the stake preprocessing to restrict it. The GeoTIFF export converts all OGGM grids that exist.

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
- for all glaciers larger than 1 km² of the regions in `REGIONS_TO_RUN` (`run_all_glacier_grids`; off by default, `RUN = False`)

## Paper figures

Where each figure in `manuscript.tex` is made. Notebooks save to `figures/paperTF/` (appendix figures to `figures/paperTF/appendix/`), under the file name used in the paper unless **Saved as** says otherwise. Cell numbers count all cells (markdown included) from 0; the section is the markdown heading above the cell.

Figures marked *assembled* are combined outside the code from the listed panels or legends. Figures 2 and A1 are schematics that are not made by any code in this folder.

### Main text

| Fig. | Paper file | Notebook | Section (cell) | Saved as |
|---|---|---|---|---|
| 1 | `fig1_study_regions.png` | `2.1 Study_regions` | (a) Alaska (13), (b) Svalbard (11), (c) Norway (7), (d) Iceland (9), (e) Central European Alps (5), (f) Central Asia (15) | *assembled* from `fig1_<a–f>_maps_<region>_gl.pdf`, `fig1_e_maps_CEU_gl_legend.pdf` and `fig1_f_maps_CA_legend_color.pdf` |
| 2 | `fig2_transformer_architecture.pdf` | — | — | schematic, not made in code |
| 3 | `fig3_within_region_transformer_only.png` | `2.3 Within_region` | Fig. 3: Transformer only (26) | |
| 4 | `fig4_zeroshot_comparison.pdf` | `2.4 Cross_region` | Fig. 4: Δ RMSE and example scatter plots (16) | *assembled* from `fig4_zeroshot_comparison_subpart.png` and `fig4_zeroshot_comparison_subpart_legend.png` |
| 5 | `fig5_zs_pooled_vs_ss_barplot_a.png` | `3.2 Pooled_model` | Fig. 5 and A6: RMSE per target region (11) | |
| 6 | `fig6_pooled_ft_gain_barplot.png` | `3.2 Pooled_model` | Fig. 6: fine-tuning gain (20) | |
| 7 | `fig7_application.png` | `3.3 Pooled_model_pred` | Fig. 7: zero-shot vs fine-tuned maps (13) | |

### Appendix

| Fig. | Paper file | Notebook | Section (cell) | Saved as |
|---|---|---|---|---|
| A1 | `A1_diff_LSTM_Transformer.pdf` | — | — | schematic, not made in code |
| A2 | `A2_within_region_lstm_vs_transf_mixed.png` | `2.3 Within_region` | Fig. A2: LSTM vs Transformer (24) | |
| A3 | `A3_CA3_per_gl.png` | (a) `2.3 Within_region`; (b) `2.1 Study_regions` | (a) Fig. A3a: Central Asia sub-region 3 by glacier (28); (b) Appendix A3 (b) (17) | *assembled* from (a) `A3_a.png` and `A3_a_legend.png`, (b) `A3_b.png` |
| A4 | `A4_random_vs_smart_split_transformer.pdf` | `2.3 Within_region` | Fig. A4: Sinkhorn-informed vs random splits (31) | `A4_random_vs_smart_split_transformer.png` (PNG, the paper uses a PDF) |
| A5 | `A5_zs_comparison_absolute.png` | `2.4 Cross_region` | Fig. A5: absolute RMSE (14) | |
| A6 | `A6_zs_pooled_vs_ss_barplot_w.png` | `3.2 Pooled_model` | Fig. 5 and A6: RMSE per target region (12) | |
| A7 | `A7_zs_pooled_vs_ss_scatter_all.png` | `3.2 Pooled_model` | Fig. A7: modeled vs observed PMB (14) | |
| A8 | `A8_sensitivity_FT.png` | `3.2 Pooled_model` | Fig. A8: RMSE versus fine-tuning fraction (25) | |
| A9 | `A9_FT_loss_curves.png` | `3.2 Pooled_model` | Fig. A9: loss curves at 10% fine-tuning data (27) | |
| A10 | `A10_ds_pooled_gain.png` | `3.4 Distance` | Fig. A10: fine-tuning gain versus domain shift (10) | |
| A11 | `A11_gradients_CH.png` | `3.3 Pooled_model_pred` | Fig. A11–A16: elevation gradients of all stake glaciers (15) | |
| A12 | `A12_gradients_CEU.png` | `3.3 Pooled_model_pred` | Fig. A11–A16: elevation gradients of all stake glaciers (15) | |
| A13 | `A13_gradients_NOR_SJM.png` | `3.3 Pooled_model_pred` | Fig. A11–A16: elevation gradients of all stake glaciers (15) | |
| A14 | `A14_gradients_ISL.png` | `3.3 Pooled_model_pred` | Fig. A11–A16: elevation gradients of all stake glaciers (15) | |
| A15 | `A15_gradients_CA.png` | `3.3 Pooled_model_pred` | Fig. A11–A16: elevation gradients of all stake glaciers (15) | |
| A16 | `A16_gradients_ALA.png` | `3.3 Pooled_model_pred` | Fig. A11–A16: elevation gradients of all stake glaciers (15) | |


## `scripts/`

These are the helper modules imported by the notebooks:
- `config_TF_Europe.py`: paths, `RGI_REGIONS`, variable lists
- `oggm/`: OGGM glacier directories and grid export, and sampling of OGGM variables at stakes
- `pmb_processing/`: stake QC, RGI ID assignment and SVF sampling, used by the regional preprocessing notebooks
- `svf_processing/`: SVF computation (needs the `rvt` environment)
- `geodata/`: GeoTIFF export
- `config_models.py`: model hyperparameters, search and fine-tuning settings (see [Model settings](#model-settings))
- `dataset/`: stake loading, monthly data, grid creation, and the datasets and splits of the experiments (e.g. `within_region_splits.py` for 2.3, `pooled_experiment.py` for 3.2–3.4, `pooled_distance.py` for 3.4)
- `models/`: training, evaluation and the model figures of each experiment (`within_region.py`, `cross_region.py`, `transformer_grid_search.py`, `pooled_model.py`, `pooled_figures.py`, `ft_sensitivity.py`)
- `plotting/`: maps and other figures (`study_region_maps.py` for 2.1, `domain_shift.py`, `pooled_pred.py` and `pooled_pred_figures.py` for 3.3)
- `utils/`: glacier and region helpers

The packages are star-imported by the notebooks (`from papers.Transformer_TF.scripts.models import *`, ...).
