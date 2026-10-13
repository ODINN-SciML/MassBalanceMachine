"""Models and metrics of the multi-region (pooled) experiments, shared by notebooks
`3.2 Pooled_model`, `3.3 Pooled_model_pred` and `3.4 Distance`.

- `load_pooled_model`: the multi-region Transformer and the scaler of its training data.
- `load_single_source_models`: the single-region Transformers of `2.4 Cross_region`.
- `add_nrmse` / `evaluate_on`: evaluation of a model on a test dataset.
- `zero_shot_comparison_metrics`: zero-shot metrics of all models on all targets.
- `finetune_id_split_models`: fine-tuned models per target region and repeat.
- `ft_metrics_id_split`: metrics of zero-shot and fine-tuned models, averaged over repeats.
"""

import os

import numpy as np
import torch
import massbalancemachine as mbm

from papers.Transformer_TF.scripts.models.lstm_helpers import (
    train_or_load_one_source_model,
)
from papers.Transformer_TF.scripts.models.transformer_helpers import (
    finetune_transformer_on_target,
)


def _fit_scaler_dataset(assets, seed):
    """Copy of the training dataset with scalers fitted on its train split."""
    ds_scaler = mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
        assets["ds_train"]
    )
    ds_scaler.make_loaders(
        train_idx=assets["train_idx"],
        val_idx=assets["val_idx"],
        fit_and_transform=True,
        seed=seed,
        verbose=False,
    )
    return ds_scaler


def load_pooled_model(
    cfg, device, assets_full, params, models_dir, epochs, date, train_flag=False
):
    """
    Load (or with train_flag=True train) the multi-region Transformer
    `transformer_foundation_100pct` from models_dir, and build the scaler dataset
    of its training data (used to standardise every test set).

    Returns (model_pooled, ds_pooled_scaler).
    """
    os.makedirs(models_dir, exist_ok=True)
    model_pooled, model_path, _ = train_or_load_one_source_model(
        cfg=cfg,
        key="foundation_100pct",
        lstm_assets=assets_full,
        best_params=params,
        device=device,
        models_dir=models_dir,
        prefix="transformer_foundation",
        train_flag=train_flag,
        force_retrain=False,
        epochs=epochs,
        date=date,
        model_type="transformer",
        verbose=True,
    )
    print(f"Pooled model saved to: {model_path}")

    ds_pooled_scaler = _fit_scaler_dataset(assets_full, cfg.seed)
    return model_pooled, ds_pooled_scaler


def load_single_source_models(
    cfg,
    device,
    res_xreg_by_source,
    source_regions,
    monthly_cols,
    static_cols,
    months_head_pad,
    months_tail_pad,
    params,
    models_base,
    epochs,
    date,
    train_flag=False,
):
    """
    Load the single-region Transformers trained in `2.4 Cross_region`
    (models_base/<src>/transformer_xreg_<src>_<src>_<date>.pt), with the scaler of
    each source region's training data. Missing models are trained if train_flag.

    Returns (models, assets, scalers), each {source region: ...}.
    """
    # imported here: dataset.fd_model imports the plotting package, which imports
    # this module (circular at import time)
    from papers.Transformer_TF.scripts.dataset.fd_model import (
        build_assets_from_glacier_list,
    )

    models, assets, scalers = {}, {}, {}
    for src in source_regions:
        ckpt = models_base / src / f"transformer_xreg_{src}_{src}_{date}.pt"
        os.makedirs(ckpt.parent, exist_ok=True)

        df_src = (
            res_xreg_by_source[src]["df_train"][["GLACIER"]]
            .drop_duplicates()
            .assign(region=src)
            .rename(columns={"GLACIER": "glacier"})
        )
        assets_src = build_assets_from_glacier_list(
            glaciers=df_src["glacier"].tolist(),
            df_ranked=df_src,
            res_xreg_by_source=res_xreg_by_source,
            monthly_cols=monthly_cols,
            static_cols=static_cols,
            cfg=cfg,
            cache_path=str(models_base / src / f"assets_{src}_100pct.joblib"),
            force_recompute=False,
            months_head_pad=months_head_pad,
            months_tail_pad=months_tail_pad,
            src_regions=[src],
        )
        assets[src] = assets_src
        scalers[src] = _fit_scaler_dataset(assets_src, cfg.seed)

        m = mbm.models.Transformer_MB.build_model_from_params(
            cfg, params, device, verbose=False
        )
        if ckpt.exists():
            m.load_state_dict(torch.load(ckpt, map_location=device))
            print(f"Loaded {src} model from {ckpt}")
        else:
            print(f"No checkpoint found for {src}, training from scratch...")
            m, _, _ = train_or_load_one_source_model(
                cfg=cfg,
                key=src,
                lstm_assets=assets_src,
                best_params=params,
                device=device,
                models_dir=models_base / src,
                prefix="transformer_xreg",
                train_flag=train_flag,
                force_retrain=False,
                epochs=epochs,
                date=date,
                model_type="transformer",
                verbose=False,
            )
            print(f"Trained {src} model, saved to {ckpt}")
        models[src] = m
    return models, assets, scalers


def add_nrmse(metrics, preds_df):
    """Add NRMSE (RMSE normalised by the std of observed PMB) for annual and winter."""
    preds_clean = preds_df.dropna(subset=["target", "pred"])

    std_annual = preds_clean.loc[preds_clean["PERIOD"] == "annual", "target"].std()
    std_winter = preds_clean.loc[preds_clean["PERIOD"] == "winter", "target"].std()

    metrics["NRMSE_annual"] = (
        metrics["RMSE_annual"] / std_annual if std_annual and std_annual > 0 else np.nan
    )
    metrics["NRMSE_winter"] = (
        metrics["RMSE_winter"] / std_winter if std_winter and std_winter > 0 else np.nan
    )
    return metrics


def evaluate_on(cfg, device, model, ds_test, scaler):
    """Metrics and predictions of model on ds_test, standardised with scaler."""
    ds_copy = mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
        ds_test
    )
    test_dl = mbm.data_processing.MBSequenceDataset.make_test_loader(
        ds_copy, scaler, batch_size=128, seed=cfg.seed
    )
    return model.evaluate_with_preds(device, test_dl, ds_copy)


def _value_range(preds_df):
    preds_clean = preds_df.dropna(subset=["target", "pred"])
    return {
        "y_min": float(preds_clean["target"].min()),
        "y_max": float(preds_clean["target"].max()),
        "pred_min": float(preds_clean["pred"].min()),
        "pred_max": float(preds_clean["pred"].max()),
    }


def zero_shot_comparison_metrics(
    cfg,
    device,
    eval_configs,
    model_pooled,
    ds_pooled_scaler,
    single_source_models,
    single_source_scalers,
):
    """
    Zero-shot metrics (with NRMSE) of the multi-region model ('pooled_zs') and of
    each single-region model on the full test set of every target region.

    Returns (df_metrics_comparison, region_pred_target_ranges), both
    {target: {model: ...}}; the second holds the range of observed and predicted PMB.
    """
    df_metrics_comparison, ranges = {}, {}
    for tgt_label, assets in eval_configs:
        if assets is None:
            print(f"No assets for {tgt_label}, skipping.")
            continue
        df_metrics_comparison[tgt_label], ranges[tgt_label] = {}, {}

        m, preds_df = evaluate_on(
            cfg, device, model_pooled, assets["ds_test"], ds_pooled_scaler
        )
        df_metrics_comparison[tgt_label]["pooled_zs"] = add_nrmse(m, preds_df)
        ranges[tgt_label]["pooled_zs"] = _value_range(preds_df)

        for src, model in single_source_models.items():
            m, preds_df = evaluate_on(
                cfg, device, model, assets["ds_test"], single_source_scalers[src]
            )
            df_metrics_comparison[tgt_label][src] = add_nrmse(m, preds_df)
            ranges[tgt_label][src] = _value_range(preds_df)
    return df_metrics_comparison, ranges


def print_metric_summary(df_metrics_comparison, region_pred_target_ranges):
    """Table of RMSE, NRMSE and R² (annual, winter) per target and model."""
    print(f"\n{'='*90}")
    print(
        f"{'Model':<15} {'RMSE_a':>8} {'NRMSE_a':>8} {'RMSE_w':>8} {'NRMSE_w':>8} {'R2_a':>8} {'R2_w':>8}"
    )
    print(f"{'='*90}")
    for tgt_label, results in df_metrics_comparison.items():
        print(f"\n--- {tgt_label} ---")
        for model_label, m in results.items():
            rng = region_pred_target_ranges[tgt_label][model_label]
            print(
                f"  {model_label:<15}  "
                f"RMSE_a={m['RMSE_annual']:.3f}  "
                f"NRMSE_a={m['NRMSE_annual']:.3f}  "
                f"RMSE_w={m['RMSE_winter']:.3f}  "
                f"NRMSE_w={m['NRMSE_winter']:.3f}  "
                f"R2_a={m['R2_annual']:.3f}  "
                f"R2_w={m['R2_winter']:.3f}  "
                f"y∈[{rng['y_min']:.2f}, {rng['y_max']:.2f}]  "
                f"pred∈[{rng['pred_min']:.2f}, {rng['pred_max']:.2f}]"
            )


def finetune_id_split_models(
    cfg,
    device,
    model_pooled,
    ds_pooled_scaler,
    ft_assets_id,
    regions,
    models_dir_ft,
    params,
    ft_params,
    date,
    strategies=("head_only", "full"),
    force_retrain=False,
):
    """
    Fine-tune (or load) the multi-region model on the FT set of each target region
    and repeat, for each strategy ('head_only': output head only; 'full': all
    weights). Models are stored as
    `transformer_ft_<strategy>_<region>_IDsplit_rep<rep>_<date>.pt`.

    Returns {region: [{"rep", "models": {strategy: model}}, ...]}.
    """
    os.makedirs(models_dir_ft, exist_ok=True)
    ft_models_id = {}
    for region in regions:
        ft_models_id[region] = []
        print(f"\n{'='*60}")
        print(
            f"Fine-tuning on: {region} (IDsplit, {len(ft_assets_id[region])} repeats)"
        )

        for rep_data in ft_assets_id[region]:
            rep = rep_data["rep"]
            rep_models = {}
            for strategy in strategies:
                rep_models[strategy] = finetune_transformer_on_target(
                    cfg=cfg,
                    model_pooled=model_pooled,
                    ds_ft=rep_data["ds_ft"],
                    ds_pooled_scaler=ds_pooled_scaler,
                    device=device,
                    best_params=params,
                    model_filename=str(
                        models_dir_ft
                        / f"transformer_ft_{strategy}_{region}_IDsplit_rep{rep}_{date}.pt"
                    ),
                    strategy=strategy,
                    epochs=ft_params["epochs"],
                    lr_factor=ft_params["lr_factor"],
                    force_retrain=force_retrain,
                )
                print(f"  [{region}] rep {rep} - {strategy} done")
            ft_models_id[region].append({"rep": rep, "models": rep_models})
    return ft_models_id


def ft_metrics_id_split(
    cfg,
    device,
    model_pooled,
    ds_pooled_scaler,
    ft_assets_id,
    ft_models_id,
    regions,
    strategies=("head_only", "full"),
):
    """
    Metrics of the zero-shot model ('no_ft') and of each fine-tuned model on the
    test set of each repeat, averaged over the repeats.

    Returns (df_ft_metrics_id, df_ft_metrics_id_raw):
    {region: {label: {metric: mean, metric_std: std}}} and
    {region: {label: [metrics per repeat]}}.
    """
    labels = ["no_ft"] + list(strategies)
    print(f"\n{'='*95}")
    print(
        f"{'Strategy':<25} {'RMSE_a':>8} {'NRMSE_a':>8} {'RMSE_w':>8} {'NRMSE_w':>8} {'R2_a':>8} {'R2_w':>8}"
    )
    print(f"{'='*95}")

    df_ft_metrics_id, df_ft_metrics_id_raw = {}, {}
    for region in regions:
        print(f"\n--- {region} (mean over {len(ft_assets_id[region])} repeats) ---")
        df_ft_metrics_id_raw[region] = {label: [] for label in labels}

        for rep_idx, rep_data in enumerate(ft_assets_id[region]):
            rep_ft_models = ft_models_id[region][rep_idx]["models"]
            all_models = {
                "no_ft": model_pooled,
                **{s: rep_ft_models[s] for s in strategies},
            }
            for label, model in all_models.items():
                m, preds_df = evaluate_on(
                    cfg, device, model, rep_data["ds_test"], ds_pooled_scaler
                )
                df_ft_metrics_id_raw[region][label].append(add_nrmse(m, preds_df))

        # mean and std across repeats
        df_ft_metrics_id[region] = {}
        for label, rep_metrics_list in df_ft_metrics_id_raw[region].items():
            agg = {}
            for metric_key in rep_metrics_list[0].keys():
                vals = [m[metric_key] for m in rep_metrics_list]
                agg[metric_key] = float(np.mean(vals))
                agg[f"{metric_key}_std"] = float(np.std(vals))
            df_ft_metrics_id[region][label] = agg

            print(
                f"  {label:<25}  "
                f"RMSE_a={agg['RMSE_annual']:.3f}±{agg['RMSE_annual_std']:.3f}  "
                f"NRMSE_a={agg['NRMSE_annual']:.3f}±{agg['NRMSE_annual_std']:.3f}  "
                f"RMSE_w={agg['RMSE_winter']:.3f}±{agg['RMSE_winter_std']:.3f}  "
                f"NRMSE_w={agg['NRMSE_winter']:.3f}±{agg['NRMSE_winter_std']:.3f}  "
                f"R2_a={agg['R2_annual']:.3f}±{agg['R2_annual_std']:.3f}  "
                f"R2_w={agg['R2_winter']:.3f}±{agg['R2_winter_std']:.3f}"
            )
    return df_ft_metrics_id, df_ft_metrics_id_raw
