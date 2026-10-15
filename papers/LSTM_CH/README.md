# LSTM_CH — Glacier mass balance modelling with an LSTM (Switzerland)

This folder contains the code for the paper:

> van der Meer, M., Zekollari, H., Gossard, A., Sjursen, K. H., Bolibar, J., Huss, M., & Farinotti, D. (2026).
> *Glacier Mass Balance Modeling Using a Long Short-Term Memory Network.*
> https://doi.org/10.1029/2026JH001250

It trains a Long Short-Term Memory (LSTM) network with the [MassBalanceMachine](../../README.md) (MBM) to predict monthly glacier surface mass balance in the Swiss Alps from ERA5-Land climate and topography. The model is trained on seasonal (winter and annual) stake measurements from GLAMOS. The LSTM is compared with an XGBoost and a feed-forward neural network (MLP) baseline, evaluated on held-out glaciers, and compared against GLAMOS gridded and geodetic mass balance. The folder also contains the analyses of seasonal patterns, feature importance and sensitivity.

## Setup

- **Environment:** the MBM environment from the repository root (`environment_cpu.yml` or `environment_gpu.yml`). Notebook `1.4` runs in a separate environment with the [Relief Visualization Toolbox](https://github.com/EarthObservation/RVT_py) (`rvt`) installed.
- **Imports:** run the notebooks from this folder. They add the repository root to `sys.path` and import the helpers as `papers.LSTM_CH.scripts.<module>` (notebook `1.4`, which runs in the `rvt` environment, imports `scripts.svf_processing` directly).
- **Data root:** all data paths are relative to `cfg.dataPath` (`cfg = mbm.SwitzerlandConfig()`). By default this is a `data/` folder next to the repository. The sub-paths are defined in `scripts/config_CH.py`. Notebook `1.4` can't load the MBM config, so it sets the same default path itself (`cfg_datapath` in its setup cell); change it there if your data are elsewhere.

### Input data

| Data | Used for | Path under `cfg.dataPath` |
|---|---|---|
| GLAMOS stake (point) mass balance | Training and test targets | `GLAMOS/point/` |
| GLAMOS glacier-wide and gridded mass balance | Evaluation | `GLAMOS/glacier-wide/`, `GLAMOS/distributed_MB_grids/` |
| GLAMOS geodetic mass balance (`dV_DOI2025_allcomb_prelim.csv`) | Evaluation | `GLAMOS/geodetic/` |
| GLAMOS yearly DEMs and potential clear-sky solar radiation (PCSR) | Model inputs | `GLAMOS/topo/GLAMOS_DEM/`, `GLAMOS/topo/pcsr/` |
| Swiss Glacier Inventory 2016 (outlines, DEM, aspect, slope) | Model inputs | `GLAMOS/topo/SGI2020/` |
| Glacier ID table (short name ↔ SGI ID ↔ RGI ID) | All steps | `GLAMOS/CH_glacier_ids_long.csv` |
| RGI v6 outlines (region 11) | RGI IDs, regional grids | `RGI_v6/RGI_11_CentralEurope/` |
| ERA5-Land monthly data and geopotential | Climate inputs | `ERA5Land/raw/` (downloaded in `1.2`) |
| OGGM glacier directories | Additional topography | downloaded by OGGM in `1.1` / `1.3` |

Trained model weights (`models/`) and hyperparameter-search logs (`logs/`) are not included in this folder. Re-create them with notebooks `2.x`–`3.x`, or place them at the paths listed in `scripts/config_CH.py`.

## Order of the steps

```
1.2  ERA5-Land download ─────────────────────────────┐
1.1  GLAMOS stake data, glacier-wide MB, PCSR ───────┤
1.3  (part 1) DEM GeoTIFFs                           │
 └─> 1.4  Sky-view factor (rvt env)                  │
      └─> 1.3 (part 2) yearly masked grids           │
           └─> 1.3 (part 3) monthly glacier grids <──┘
                │
                ├─> 2.3 → 2.2   MLP baseline
                ├─> 2.1         XGBoost baseline
                └─> 3.1.x → 3.2.x  LSTM search and training
                                   │
                                   ├─> 4.x  seasonal patterns, PFI, sensitivity, 2025
                                   └─> 5.x  paper figures
```

## Notebooks

Each notebook starts with a short description of what it does and what it needs.

| Notebook | Content |
|---|---|
| **Data preparation** | |
| `1.1 GLAMOS-prepro` | Stake dataset (`CH_wgms_dataset_all.csv`): file conversion, quality control, RGI IDs, OGGM/SGI topography. Also glacier-wide MB and monthly PCSR. |
| `1.2 ERA5Land-prepro` | Downloads ERA5-Land monthly data via the CDS API. |
| `1.3 Glacier grids-SGI-prepro` | Yearly masked topography grids from the GLAMOS DEMs and monthly input tables over whole glaciers (used for gridded predictions). |
| `1.4 Skyview_factor-prepro` | Sky-view factor from DEMs (`rvt` environment). |
| **Baselines** | |
| `2.1 Train-XGB` | XGBoost: grid search, test-glacier evaluation, gridded predictions. |
| `2.2 Train-NN` | MLP: training, test-glacier evaluation, gridded predictions. |
| `2.3 Grid search NN model` | MLP hyperparameter search. |
| **LSTM** | |
| `3.1.1 Grid_search_LSTM_OOS` | Hyperparameter search, out-of-sample (test glaciers held out). |
| `3.1.2 Grid_search_LSTM_IS` | Hyperparameter search, in-sample (all glaciers). |
| `3.1.3 Grid_search_LSTM_IS_wo_pcsr` | As 3.1.2, without PCSR as input. |
| `3.2.1 Train-LSTM_OOS_norm_y` | Out-of-sample LSTM (normalised target): training, evaluation, gridded predictions. |
| `3.2.2 Train-LSTM_IS_norm_y` | In-sample LSTM (normalised target). |
| `3.2.3 Train-LSTM_IS_original_y` | In-sample LSTM (original target); its monthly gridded predictions are used in 4.1. |
| **Analysis** | |
| `4.1 Seasonal_patterns` | Monthly MB distributions of LSTM, MLP, XGBoost vs. GLAMOS; permutation feature importance. |
| `4.2.1–4.2.3 Sensitivity_*` | Gradient-based sensitivity of monthly MB to the inputs, per elevation band (Rhone, Aletsch, Gries). |
| `4.3 LSTM_pred_on_2025` | Evaluation on the 2025 measurements. |
| **Figures** | |
| `5.1 Figures paper intro` | Study area and data figures (Figs. 1–2). |
| `5.2 Figures paper LSTM` | Results figures (Figs. 4–7) and appendix figures. |


Figures are written to `figures/paper/` and `figures/paper/appendix/`. Intermediate datasets are cached in `cache/`.

## `scripts/`

Helper modules imported by the notebooks:

- `config_CH.py`: test glaciers, model paths and hyperparameters, input variables, data paths.
- `glamos/`: GLAMOS stake processing and quality control, glacier-wide and gridded MB, PCSR.
- `dataset/`: monthly datasets, train/test and CV splits, LSTM sequence datasets.
- `geo_data/`: coordinate transforms, masked topography grids, gridded model inputs, SVF merging.
- `geodetic/`: geodetic MB and comparison with modelled MB.
- `models/`: MLP architectures, LSTM helpers, prediction loading/scoring, parallel gridded inference.
- `physical/`: permutation feature importance.
- `oggm/`: OGGM glacier directories.
- `svf_processing/`: SVF computation (`rvt` environment).
- `plotting/`: figure functions.
- `utils/`: seeding, dates, glacier areas, LV03 ↔ WGS84 conversion.

## Known issues

- `models/`, `logs/` and `figures/paper/appendix/` are missing in a fresh checkout. The notebooks save to and load from these folders, so create them first.
