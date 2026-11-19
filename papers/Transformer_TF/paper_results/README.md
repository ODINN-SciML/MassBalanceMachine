# paper_results — archived result tables

Result tables behind the figures and tables of the paper, so that a re-run can be compared with the published values without retraining. The CSV files listed below are not tracked on the development branch; the versions of the paper are archived in release v0.1.3. All tables are written by the last cell ("Save result tables") of the notebook listed below. RMSE and bias are in m w.e.; `_annual` and `_winter` refer to annual and winter point mass balance.

The trained model weights are archived separately on Zenodo (https://doi.org/10.5281/zenodo.24373039; see [Model weights](#model-weights)).

| File | Notebook | Paper | Content |
|---|---|---|---|
| `within_region_metrics_transformer.csv` | `2.2 Within_region` | Fig. 3, A2, A3a | Test metrics of every within-region Transformer, one row per model `key`: `<region>` or `<region>_GL` (Sinkhorn glacier split), `<region>_ID` (measurement split), `<region>_RAND<i>` (random glacier splits) |
| `within_region_metrics_lstm.csv` | `2.2 Within_region` | Fig. A2 | Same for the LSTM |
| `within_region_split_comparison.csv` | `2.2 Within_region` | Fig. A4 | Sinkhorn split (`smart_val`) versus the mean and standard deviation of the random splits, per model, region and metric |
| `cross_region_metrics_transformer.csv` | `2.3 Cross_region` | Fig. 4, A5 | Zero-shot metrics of the single-region Transformers, one row per source-target pair (`XREG_<source>_TO_<target>`) |
| `cross_region_metrics_lstm.csv` | `2.3 Cross_region` | Fig. 4, A5 | Same for the LSTM |
| `pooled_tf_gs_2026-06-12.csv` | `3.1 Pooled_model_grid_search` | Table A2 | Grid-search log: one row per sampled Transformer configuration, with the validation loss and RMSE. The row with the lowest `valid_loss` is `PARAMS_TRANSFORMER` in `scripts/config_models.py` |
| `pooled_zero_shot_metrics.csv` | `3.2 Pooled_model` | Fig. 5, A6, A7 | Zero-shot metrics per target region of the multi-region model (`pooled_zs`) and of the four single-region models (`CH`, `NOR`, `ISL`, `FR`) |
| `pooled_finetuning_metrics.csv` | `3.2 Pooled_model` | Fig. 6 | Zero-shot (`no_ft`), head-only and full fine-tuning with 10% local data, mean and standard deviation (`_std`) over the 5 repeats, per target region |
| `pooled_finetuning_metrics_per_repeat.csv` | `3.2 Pooled_model` | Fig. 6 | Same, one row per target region, strategy and repeat |
| `ft_fraction_sensitivity_2026-06-10.csv` | `3.2 Pooled_model` | Fig. A8, A9 | Fine-tuning fraction sensitivity: metrics and loss diagnostics per target region, fraction, strategy and repeat |
| `ft_fraction_set_sizes.csv` | `3.2 Pooled_model` | Table A3 | Number of fine-tuning (`n_ft_seqs`) and test (`n_test_seqs`) measurements per target region, fraction and repeat |
| `weights_manifest.csv` | `export_weights.py` | — | File list of the archived model weights, with size and SHA-256 checksum |

Table A1 (stake data per region) is printed by `2.1 Study_regions` and is not saved here; the Icelandic stake list is in `PMB_sources/RGI_06_Iceland/ISL_stake_inventory.csv`.

The zero-shot evaluation of the multi-region model differs slightly between `pooled_zero_shot_metrics.csv` (all measurements of the target region) and the `no_ft` rows of `pooled_finetuning_metrics.csv` (the 90% test set of each repeat), as in the paper.

## Model weights

All models of the paper carry the date tag `MODEL_DATE = "2026-06-10"` (`scripts/config_models.py`) in their file names. The tag identifies the paper's model set; it is not necessarily the date on which each file was written.

`export_weights.py` packs them into two archives, keeping their paths relative to `cfg.dataPath/cache/TF_Europe/` (so they can be extracted there and loaded by the notebooks without retraining):

- `MBM_TF_paper_weights_main_2026-06-10.tar.gz`: within-region models (2.2), single-region models (2.3), multi-region model and its head-only and full fine-tuned versions (3.2–3.4)
- `MBM_TF_paper_weights_ft_fraction_sensitivity_2026-06-10.tar.gz`: fine-tuned models and loss histories of the fraction sensitivity (Fig. A8, A9)

```
python papers/Transformer_TF/paper_results/export_weights.py <output_dir>
```

Use `weights_manifest.csv` to check that the extracted files are the archived ones.
