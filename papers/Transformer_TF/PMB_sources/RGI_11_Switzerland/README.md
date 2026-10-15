# RGI 11 — Switzerland

This folder builds the Swiss point mass balance (PMB) dataset from the GLAMOS stake data. The notebook and helpers come from `papers/LSTM_CH` (van der Meer et al., 2026); only the parts needed for the stake dataset are kept here.

| RGI region | Output file |
|---|---|
| 11 (Switzerland) | `CH_wgms_dataset_all.csv` in `cfg.dataPath/WGMS/Switzerland/csv/` |

Intermediate files are written to `cfg.dataPath/GLAMOS/point/csv/`: the converted `.dat` files (`annual/`, `winter/`, `winter_clean/`), `df_pmb_50s.csv` (after the elevation check), `df_pmb_oggm_intermediate.csv` (after OGGM), and `GLAMOS_elev_mismatch.csv` (stakes flagged by the elevation check). The two `df_pmb_*` files are restart points: the notebook reads them back, so later sections can be re-run without repeating the earlier ones.

## Contents

```
RGI_11_Switzerland/
├── 1.1 GLAMOS-prepro.ipynb   # Preprocessing notebook
└── scripts/
    ├── config_CH.py          # Data paths (relative to cfg.dataPath)
    ├── glamos/
    │   ├── pmb_stakes_processing.py  # .dat → .csv, annual/winter assembly, RGI IDs, OGGM and SGI sampling
    │   └── pmb_stakes_qc.py          # Merge close stakes, RGI ID fixes, elevation check, stake renaming
    ├── geo_data/             # SGI topography masks (create_sgi_topo_masks), SVF merging, LV03/LV95 → WGS84
    ├── oggm/                 # OGGM glacier directories
    ├── utils/                # Date parsing, emptyfolder, LV03 ↔ WGS84 converter
    └── plotting/             # Not used by the notebook, see below
```

Unlike the other regions, the Swiss notebook uses its own copies of the stake QC and OGGM helpers rather than `papers/Transformer_TF/scripts/pmb_processing` and `scripts/oggm`.

`plotting/palettes.py` (`get_color_maps`), `plotting/spatial_maps.py` (`_load_lstm_ds`) and `geo_data.apply_gaussian_filter` are not used for the stake data. They are imported by `papers/Transformer_TF/3.3 Pooled_model_pred.ipynb` and `papers/Transformer_TF/scripts/plotting/pooled_pred.py`.

## Required input data

All paths are relative to `cfg.dataPath` (`mbm.SwitzerlandConfig()`):

- `GLAMOS/point/point_raw/annual/` and `winter/`: raw GLAMOS stake files (`.dat`).
- `GLAMOS/topo/GLAMOS_DEM/xr_masked_grids/<glacier>_<year>.zarr`: yearly GLAMOS DEM grids, used to check stake elevations.
- `GLAMOS/topo/SGI2020/`: SGI 2020 topography; the SVF is read from `svf_nc_latlon/` and the masked grids are written to `xr_masked_grids/`.
- `GLAMOS/CH_glacier_ids_long.csv`: glacier name ↔ SGI/RGI ID table.
- `RGI_v6/RGI_11_CentralEurope/11_rgi60_CentralEurope.shp`: RGI outlines.

The GLAMOS DEM grids and the SGI SVF are made by notebooks `1.3 Glacier grids-SGI-prepro` and `1.4 Skyview_factor-prepro` in `papers/LSTM_CH`. OGGM glacier directories are downloaded on the first run.

## What the notebook does

1. Convert the GLAMOS `.dat` files to CSV and assemble the annual and winter measurements (for each annual stake, the closest winter measurement).
2. Assign RGI IDs (with hand-curated corrections), keep years after 1950, convert to m w.e., and merge Clariden L and U.
3. Merge stakes closer than 10 m.
4. Compare stake elevations with the yearly GLAMOS DEM: differences above 400 m are replaced by the DEM value on Aletsch and removed elsewhere.
5. Sample OGGM topography and drop stakes outside their RGI outline.
6. Build the SGI masks per glacier and sample `aspect_sgi`, `slope_sgi`, `topo_sgi`, `svf`, `asvf` and `opns`; drop stakes outside the SGI mask.
7. Drop Taelliboden (one measurement) and Plaine Morte (outliers), and rename stakes by elevation.
8. Clean months: remove July starts and annual periods ending in June or November; relabel annual periods ending in May as winter and winter periods ending in August as annual; remove eight stakes with too much missing data.
9. Save `CH_wgms_dataset_all.csv`.
