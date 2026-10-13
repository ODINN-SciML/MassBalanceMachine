"""Random grid search of the Transformer hyperparameters on the pooled multi-region
dataset (notebook `2.2 Transformer_grid_search`).

- `generate_transformer_param_sets` / `sample_transformer_param_sets`: valid
  parameter combinations from a grid, and a random sample of them.
- `run_transformer_grid_search`: trains one model per sampled configuration and logs
  its validation loss and RMSE to a CSV file (2.2, random validation sequences).
- `run_pooled_transformer_grid_search`: same with validation on held-out glaciers and
  an RMSE per source region (3.1; gives PARAMS_TRANSFORMER).
- `best_params_from_gs_log` / `summarize_gs_top_k`: best configuration and summary of
  the top configurations from that CSV file.
"""

import os
import csv
import json
import random
import hashlib
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import massbalancemachine as mbm

# Hyperparameter columns of the grid-search log
GS_PARAM_COLS = [
    "d_model",
    "nhead",
    "num_layers",
    "dim_feedforward",
    "dropout",
    "static_layers",
    "static_hidden",
    "static_dropout",
    "head_dropout",
    "lr",
    "weight_decay",
]
# Validation scores logged for each configuration
GS_SCORE_COLS = ["valid_loss", "val_rmse_a", "val_rmse_w"]


def generate_transformer_param_sets(const_params, param_grid):
    """
    Generate all valid transformer parameter combinations.
    Filters out combinations where nhead does not divide d_model, or where
    dim_feedforward < d_model. Without a static MLP (static_layers == 0),
    static_hidden and static_dropout are set to None.
    """
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    param_sets = []
    for combo in product(*values):
        params = dict(zip(keys, combo))

        # nhead must divide d_model
        if params["d_model"] % params["nhead"] != 0:
            continue

        # dim_feedforward should be >= d_model (not strictly required but sensible)
        if params["dim_feedforward"] < params["d_model"]:
            continue

        # no static MLP -> ignore static_hidden & static_dropout
        if params["static_layers"] == 0:
            params["static_hidden"] = None
            params["static_dropout"] = None

        full_params = {**const_params, **params}
        param_sets.append(full_params)

    return param_sets


def sample_transformer_param_sets(const_params, param_grid, n_samples, seed=42):
    """Random sample (without replacement) of n_samples valid parameter combinations."""
    all_params = generate_transformer_param_sets(const_params, param_grid)
    print(f"Total valid combinations: {len(all_params)}")
    if n_samples >= len(all_params):
        print(f"Requested {n_samples}, using all {len(all_params)}.")
        return all_params
    rng = random.Random(seed)
    return rng.sample(all_params, n_samples)


def run_transformer_grid_search(
    cfg,
    device,
    ds_pooled,
    train_idx,
    val_idx,
    sampled_params,
    models_dir,
    logs_dir,
    epochs=150,
):
    """
    Train one Transformer per parameter set on ds_pooled (train_idx), with early
    stopping on the validation split (val_idx). The best validation loss and the
    validation RMSE (annual, winter) of each configuration are appended to
    `<logs_dir>/transformer_gs_pooled_<date>.csv`. Model checkpoints are deleted
    after evaluation to save disk space.

    Returns
    -------
    (best_overall, log_filename), with best_overall = {"val", "row", "params"} of the
    configuration with the lowest validation loss.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    log_filename = (
        logs_dir / f"transformer_gs_pooled_{datetime.now().strftime('%Y-%m-%d')}.csv"
    )

    fieldnames = list(sampled_params[0].keys()) + GS_SCORE_COLS
    with open(log_filename, mode="w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    best_overall = {"val": float("inf"), "row": None, "params": None}

    for i, params in enumerate(sampled_params):
        mbm.utils.seed_all(cfg.seed)

        run_id = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[
            :8
        ]
        model_filename = str(models_dir / f"transformer_pooled_gs_{run_id}.pt")

        if os.path.exists(model_filename):
            os.remove(model_filename)

        print(f"\n--- Config {i+1}/{len(sampled_params)}  [id={run_id}] ---")
        print(params)

        # --- loaders: fit scalers on train split, evaluate on val split ---
        ds_train_copy = (
            mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
                ds_pooled
            )
        )
        # val dataset for metric evaluation (separate clone)
        ds_val_copy = (
            mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
                ds_pooled
            )
        )

        train_dl, val_dl = ds_train_copy.make_loaders(
            train_idx=train_idx,
            val_idx=val_idx,
            batch_size_train=64,
            batch_size_val=128,
            seed=cfg.seed,
            fit_and_transform=True,
            shuffle_train=True,
            use_weighted_sampler=True,
            verbose=False,
        )

        # val loader with train scalers (for RMSE evaluation)
        val_eval_dl = mbm.data_processing.MBSequenceDataset.make_test_loader(
            ds_val_copy, ds_train_copy, batch_size=128, seed=cfg.seed
        )

        # --- build model & train ---
        model = mbm.models.Transformer_MB.build_model_from_params(
            cfg, params, device, verbose=False
        )
        loss_fn = mbm.models.Transformer_MB.resolve_loss_fn(params)

        history, best_val, _ = model.train_loop(
            device=device,
            train_dl=train_dl,
            val_dl=val_dl,
            epochs=epochs,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            clip_val=1,
            sched_factor=0.5,
            sched_patience=6,
            sched_threshold=0.01,
            sched_threshold_mode="rel",
            sched_cooldown=1,
            sched_min_lr=1e-6,
            es_patience=15,
            es_min_delta=1e-4,
            log_every=10,
            verbose=False,
            save_best_path=model_filename,
            loss_fn=loss_fn,
        )

        # --- evaluate on val split for RMSE (not used for selection, just logged) ---
        best_state = torch.load(model_filename, map_location=device)
        model.load_state_dict(best_state)

        val_metrics, _ = model.evaluate_with_preds(device, val_eval_dl, ds_val_copy)
        val_rmse_a = val_metrics["RMSE_annual"]
        val_rmse_w = val_metrics["RMSE_winter"]

        print(
            f"  val_loss={best_val:.4f}  RMSE_a={val_rmse_a:.3f}  RMSE_w={val_rmse_w:.3f}"
        )

        # --- log ---
        row = {
            **params,
            "valid_loss": float(best_val),
            "val_rmse_a": float(val_rmse_a),
            "val_rmse_w": float(val_rmse_w),
        }
        with open(log_filename, mode="a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        if best_val < best_overall["val"]:
            best_overall = {"val": best_val, "row": row, "params": params}

        # clean up model file to save disk space
        if os.path.exists(model_filename):
            os.remove(model_filename)

    print("\n=== Best config by validation loss ===")
    print(best_overall["params"])
    print(f"  val_loss={best_overall['val']:.4f}")
    print(f"  val_rmse_a={best_overall['row']['val_rmse_a']:.3f}")
    print(f"  val_rmse_w={best_overall['row']['val_rmse_w']:.3f}")
    return best_overall, log_filename


def best_params_from_gs_log(logs_gs, verbose=True):
    """
    Parameter dict of the configuration with the lowest validation loss in a
    grid-search log (DataFrame read from the CSV of `run_transformer_grid_search`),
    with the same keys and types as PARAMS_TRANSFORMER in `config_models.py`.
    """
    best_row = logs_gs.sort_values("valid_loss").iloc[0]
    if verbose:
        print("Best config:")
        print(best_row[GS_PARAM_COLS + GS_SCORE_COLS])

    best_params = {
        "Fm": int(best_row["Fm"]),
        "Fs": int(best_row["Fs"]),
        "d_model": int(best_row["d_model"]),
        "nhead": int(best_row["nhead"]),
        "num_layers": int(best_row["num_layers"]),
        "dim_feedforward": int(best_row["dim_feedforward"]),
        "dropout": float(best_row["dropout"]),
        "static_layers": int(best_row["static_layers"]),
        "static_hidden": (
            None
            if pd.isna(best_row["static_hidden"])
            else int(best_row["static_hidden"])
        ),
        "static_dropout": (
            None
            if pd.isna(best_row["static_dropout"])
            else float(best_row["static_dropout"])
        ),
        "head_dropout": float(best_row["head_dropout"]),
        "lr": float(best_row["lr"]),
        "weight_decay": float(best_row["weight_decay"]),
        "two_heads": False,
        "loss_spec": None,
        "T_max": 32,
    }

    if verbose:
        print("\nbest_params_gs:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
    return best_params


def summarize_gs_top_k(logs_gs, k=10):
    """
    Print the k best configurations (by validation loss), how much each
    hyperparameter varies among them, value counts of the architecture
    parameters, and the spread of their validation scores.
    """
    top_k = logs_gs.sort_values("valid_loss").head(k)

    # --- top K table ---
    print(f"=== Top {k} configs by validation loss ===")
    print(top_k[GS_PARAM_COLS + GS_SCORE_COLS].to_string(index=False))

    # --- variability summary ---
    print(f"\n=== Parameter variability across top {k} ===")
    for col in GS_PARAM_COLS:
        vals = top_k[col].dropna()
        unique = sorted(vals.unique())
        if len(unique) == 1:
            stability = "stable"
        elif len(unique) <= 3:
            stability = "low variance"
        else:
            stability = "high variance"
        print(f"  {col:20s}: {unique}  [{stability}]")

    # --- value counts for categorical-like params ---
    print(f"\n=== Value counts in top {k} ===")
    for col in [
        "d_model",
        "nhead",
        "num_layers",
        "dim_feedforward",
        "static_layers",
        "static_hidden",
    ]:
        counts = top_k[col].value_counts().sort_index()
        print(f"  {col:20s}: {counts.to_dict()}")

    # --- loss spread ---
    print(f"\n=== Loss spread across top {k} ===")
    print(
        f"  valid_loss : min={top_k['valid_loss'].min():.4f}  "
        f"max={top_k['valid_loss'].max():.4f}  "
        f"range={top_k['valid_loss'].max() - top_k['valid_loss'].min():.4f}"
    )
    print(
        f"  val_rmse_a : min={top_k['val_rmse_a'].min():.3f}  "
        f"max={top_k['val_rmse_a'].max():.3f}"
    )
    print(
        f"  val_rmse_w : min={top_k['val_rmse_w'].min():.3f}  "
        f"max={top_k['val_rmse_w'].max():.3f}"
    )
    return top_k


def run_pooled_transformer_grid_search(
    cfg,
    device,
    ds_gs_train,
    val_datasets,
    source_regions,
    sampled_params,
    log_filename,
    epochs=150,
):
    """
    Grid search of `3.1 Pooled_model_grid_search`: train one Transformer per parameter
    set on the grid-search training glaciers (ds_gs_train), with early stopping on the
    pooled validation glaciers (val_datasets["pooled"]). The validation loss, the
    pooled validation RMSE (annual, winter) and the annual RMSE per source region
    (val_datasets[region]) are appended to log_filename.

    Unlike `run_transformer_grid_search`, no checkpoint is saved, and the RMSE is
    computed with the model at the end of training.

    Returns best_overall = {"val", "row", "params"} (lowest validation loss).
    """
    fieldnames = (
        list(sampled_params[0].keys())
        + GS_SCORE_COLS
        + [f"val_rmse_a_{r}" for r in source_regions]
    )

    with open(log_filename, mode="w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    best_overall = {"val": float("inf"), "row": None, "params": None}

    for i, params in enumerate(sampled_params):
        mbm.utils.seed_all(cfg.seed)
        run_id = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[
            :8
        ]
        print(f"\n--- Config {i+1}/{len(sampled_params)}  [id={run_id}] ---")

        # Fresh clone of train dataset — fit scalers here
        ds_train_copy = (
            mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
                ds_gs_train
            )
        )
        ds_train_copy.fit_scalers(np.arange(len(ds_train_copy)))
        ds_train_copy.transform_inplace()
        train_dl = DataLoader(ds_train_copy, batch_size=64, shuffle=True)

        # Pooled val loader — apply train scalers
        ds_val_copy = (
            mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
                val_datasets["pooled"]["ds_train"]
            )
        )
        val_dl = mbm.data_processing.MBSequenceDataset.make_test_loader(
            ds_val_copy, ds_train_copy, batch_size=128, seed=cfg.seed
        )

        model = mbm.models.Transformer_MB.build_model_from_params(
            cfg, params, device, verbose=False
        )
        loss_fn = mbm.models.Transformer_MB.resolve_loss_fn(params)

        history, best_val, _ = model.train_loop(
            device=device,
            train_dl=train_dl,
            val_dl=val_dl,
            epochs=epochs,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            clip_val=1,
            sched_factor=0.5,
            sched_patience=6,
            sched_threshold=0.01,
            sched_threshold_mode="rel",
            sched_cooldown=1,
            sched_min_lr=1e-6,
            es_patience=15,
            es_min_delta=1e-4,
            log_every=10,
            verbose=False,
            loss_fn=loss_fn,
        )

        # Pooled val RMSE
        val_metrics, _ = model.evaluate_with_preds(device, val_dl, ds_val_copy)

        # Per-region val RMSE
        per_region_rmse = {}
        for region in source_regions:
            ds_reg_copy = (
                mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
                    val_datasets[region]["ds_train"]
                )
            )
            reg_dl = mbm.data_processing.MBSequenceDataset.make_test_loader(
                ds_reg_copy, ds_train_copy, batch_size=128, seed=cfg.seed
            )
            reg_metrics, _ = model.evaluate_with_preds(device, reg_dl, ds_reg_copy)
            per_region_rmse[f"val_rmse_a_{region}"] = reg_metrics["RMSE_annual"]

        print(
            f"  val_loss={best_val:.4f}  RMSE_a={val_metrics['RMSE_annual']:.3f}  "
            f"RMSE_w={val_metrics['RMSE_winter']:.3f}  "
            + "  ".join(
                f"{r}={per_region_rmse[f'val_rmse_a_{r}']:.3f}" for r in source_regions
            )
        )

        row = {
            **params,
            "valid_loss": float(best_val),
            "val_rmse_a": float(val_metrics["RMSE_annual"]),
            "val_rmse_w": float(val_metrics["RMSE_winter"]),
            **per_region_rmse,
        }
        with open(log_filename, mode="a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        if best_val < best_overall["val"]:
            best_overall = {"val": best_val, "row": row, "params": params}

    print("\n=== Best config ===")
    print(best_overall["params"])
    print(
        f"  val_loss={best_overall['val']:.4f}  val_rmse_a={best_overall['row']['val_rmse_a']:.3f}"
    )
    return best_overall
