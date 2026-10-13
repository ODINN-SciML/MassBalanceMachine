# RGI 11 — Switzerland

This folder builds the Swiss point mass balance (PMB) dataset from the GLAMOS stake data. The notebook and helpers come from `papers/LSTM_CH` (van der Meer et al., 2026); only the parts needed for the stake dataset are kept here.

| RGI region | Output file |
|---|---|
| 11 (Switzerland) | `CH_wgms_dataset_all.csv` in `cfg.dataPath/WGMS/Switzerland/csv/` |

Intermediate files (converted `.dat` files, `df_pmb_50s.csv`, `df_pmb_oggm_intermediate.csv`, `GLAMOS_elev_mismatch.csv`) are written to `cfg.dataPath/GLAMOS/point/csv/`.

## Contents

```
RGI_11_Switzerland/
├── 1.1 GLAMOS-prepro.ipynb   # Preprocessing notebook
└── scripts/
    ├── config_CH.py          # Data paths and Swiss settings
    ├── glamos/               # .dat → .csv conversion, stake assembly, QC, glacier-wide MB, PCSR
    ├── geo_data/             # SGI topography masks, coordinate transforms, SVF merging
    ├── geodetic/             # Geodetic MB helpers (used by ../../4.1 Pooled_model_pred)
    ├── oggm/                 # OGGM glacier directories
    ├── plotting/             # Palettes and LSTM map loading (used by ../../scripts/plotting/pooled_pred.py)
    └── utils/                # Dates, glacier areas, helpers, LV03 ↔ WGS84
```

Unlike the other regions, the Swiss notebook uses its own copies of the stake QC helpers (`glamos/pmb_stakes_qc.py`) rather than `papers/Transformer_TF/scripts/pmb_processing`.

## Required input data

All paths are relative to `cfg.dataPath` (`mbm.SwitzerlandConfig()`):

- `GLAMOS/point/point_raw/`: raw GLAMOS stake files (`.dat`).
- `GLAMOS/topo/GLAMOS_DEM/xr_masked_grids/<glacier>_<year>.zarr`: yearly GLAMOS DEM grids, used to correct stake elevations.
- `GLAMOS/topo/SGI2020/`: SGI 2020 topography, and the SVF in `svf_nc_latlon/`.
- `RGI_v6/RGI_11_CentralEurope/11_rgi60_CentralEurope.shp`: RGI outlines.

The two grid folders are made by notebooks `1.3 Glacier grids-SGI-prepro` and `1.4 Skyview_factor-prepro` in `papers/LSTM_CH`.

## What the notebook does

1. Convert the GLAMOS `.dat` files to CSV and assemble the annual and winter measurements.
2. Assign RGI IDs (with hand-curated corrections), keep years after 1950 and convert to m w.e.
3. Merge stakes closer than 10 m and correct or remove stakes whose elevation differs from the GLAMOS DEM by more than 400 m.
4. Sample OGGM topography, then the SGI topography and SVF (`aspect_sgi`, `slope_sgi`, `topo_sgi`, `svf`, `asvf`, `opns`).
5. Rename stakes by elevation and clean the months (remove July starts, reclassify some annual/winter periods, drop stakes with too little data).
6. Save `CH_wgms_dataset_all.csv`.

The notebook also converts the GLAMOS glacier-wide mass balance files and the potential clear-sky solar radiation (PCSR) grids. These are not needed for the stake dataset.
