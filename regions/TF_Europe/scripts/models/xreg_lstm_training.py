from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np
import random as rd
import torch
import os
from datetime import datetime
import logging

import massbalancemachine as mbm

from regions.TF_Europe.scripts.dataset import make_finetune_loaders_for_exp
from regions.TF_Europe.scripts.plotting import plot_history_lstm


def make_param_groups_lstm_mb(model, lr_lstm, lr_static, lr_head, weight_decay):
    groups = {"lstm": [], "static": [], "head": []}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("lstm."):
            groups["lstm"].append(p)
        elif name.startswith("static_mlp."):
            groups["static"].append(p)
        else:
            groups["head"].append(p)

    param_groups = []
    if groups["head"]:
        param_groups.append(
            {"params": groups["head"], "lr": lr_head, "weight_decay": weight_decay}
        )
    if groups["static"]:
        param_groups.append(
            {"params": groups["static"], "lr": lr_static, "weight_decay": weight_decay}
        )
    if groups["lstm"]:
        param_groups.append(
            {"params": groups["lstm"], "lr": lr_lstm, "weight_decay": weight_decay}
        )

    return param_groups


def freeze_lstm_only(model):
    for name, p in model.named_parameters():
        if name.startswith("lstm."):
            p.requires_grad = False
        else:
            p.requires_grad = True


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def finetune_or_load_one_TL(
    cfg,
    exp_key: str,
    tl_assets_for_key: dict,
    best_params: dict,
    device,
    pretrained_ckpt_path: str,
    models_dir="models",
    prefix="lstm_TL",
    strategy="safe",
    force_retrain=False,
    batch_size_train=64,
    batch_size_val=128,
    epochs_safe=60,
    epochs_full=80,
    stage1_epochs=20,
    stage2_epochs=60,
    lr_safe=1e-4,
    lr_full=1e-5,
    lr_stage1=2e-4,
    lr_stage2=1e-5,
    # ---- NEW knobs (optional) ----
    lr_head=5e-5,
    lr_static=1e-5,
    lr_lstm=5e-6,
    # ---- Adapter knobs ----
    lr_adapter=1e-4,
    train_head_with_adapter=True,
):
    os.makedirs(models_dir, exist_ok=True)
    current_date = datetime.now().strftime("%Y-%m-%d")

    out_name = f"{prefix}_{exp_key}_{strategy}_{current_date}.pt"
    out_path = os.path.join(models_dir, out_name)

    # -------------------------
    # helpers (local, minimal)
    # -------------------------
    def _freeze_all(m):
        for p in m.parameters():
            p.requires_grad = False

    def _unfreeze_module(mod):
        for p in mod.parameters():
            p.requires_grad = True

    def _unfreeze_heads(m):
        if getattr(m, "two_heads", False):
            _unfreeze_module(m.head_w)
            _unfreeze_module(m.head_a)
        else:
            _unfreeze_module(m.head)

    def _unfreeze_adapters(m):
        if not getattr(m, "use_adapter", False):
            return
        if getattr(m, "adapter_domainwise", False):
            _unfreeze_module(m.adapters)
        else:
            _unfreeze_module(m.adapter)

    # load if exists
    if (not force_retrain) and os.path.exists(out_path):
        model = mbm.models.LSTM_MB.build_model_from_params(cfg, best_params, device)
        state = torch.load(out_path, map_location=device)
        model.load_state_dict(state)
        return model, out_path, None

    # loaders (we want domain_vocab *before* building model if adapter_domainwise)
    ds_ft_tl, ft_train_dl, ft_val_dl = make_finetune_loaders_for_exp(
        cfg,
        tl_assets_for_key,
        batch_size_train=batch_size_train,
        batch_size_val=batch_size_val,
        domain_vocab=best_params.get("domain_vocab", None),  # optional
    )

    # -------------------------
    # Ensure adapter params if adapter strategy
    # -------------------------
    is_adapter_strategy = strategy == "adapter"

    params = dict(best_params)  # don't mutate caller dict
    if is_adapter_strategy:
        params["use_adapter"] = True

        # Ensure domainwise adapter has n_domains consistent with the vocab actually used in ds_ft_tl
        # MBSequenceDatasetTL stores the mapping as ds_ft_tl.domain_vocab
        dv = getattr(ds_ft_tl, "domain_vocab", None)
        if params.get("adapter_domainwise", True):
            if dv is None:
                # domain_id won't exist; model will fall back to adapter[0], but you still need n_domains
                # simplest: set n_domains=1 unless user provided something else
                params["n_domains"] = int(params.get("n_domains", 1))
            else:
                params["n_domains"] = len(dv)

    # build model + base loss
    model = mbm.models.LSTM_MB.build_model_from_params(cfg, params, device)
    base_loss_fn = mbm.models.LSTM_MB.resolve_loss_fn(params)

    # load pretrained weights (CH)
    pretrained_state = torch.load(pretrained_ckpt_path, map_location=device)

    if is_adapter_strategy:
        # strict=False: pretrained checkpoint won't have adapter weights
        missing, unexpected = model.load_state_dict(pretrained_state, strict=False)
        if len(unexpected) > 0:
            logging.warning(
                f"[{exp_key}] Unexpected keys when loading pretrained: {unexpected[:10]}"
            )
        # missing will include adapter params; that's expected
    else:
        model.load_state_dict(pretrained_state)

    # anchor for L2-SP (snapshot right after loading pretrained)
    anchor_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # overwrite if retraining
    if os.path.exists(out_path):
        os.remove(out_path)
        logging.info(f"[{exp_key}] Deleted existing TL checkpoint: {out_path}")

    # -------------------------
    # Adapter strategy (single supported variant):
    #   Train adapters + head(s), freeze backbone
    # -------------------------
    if strategy == "adapter":
        _freeze_all(model)
        _unfreeze_adapters(model)
        _unfreeze_heads(model)

        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(params.get("lr_adapter", lr_adapter)),
            weight_decay=params["weight_decay"],
        )
        history, best_val, best_state = model.train_loop(
            device=device,
            train_dl=ft_train_dl,
            val_dl=ft_val_dl,
            epochs=epochs_safe,
            optimizer=opt,
            clip_val=1.0,
            loss_fn=base_loss_fn,
            es_patience=8,
            save_best_path=out_path,
            verbose=False,
        )

    # -------------------------
    # Existing strategies
    # -------------------------
    elif strategy == "safe":
        freeze_lstm_only(model)
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr_safe,
            weight_decay=params["weight_decay"],
        )
        history, best_val, best_state = model.train_loop(
            device=device,
            train_dl=ft_train_dl,
            val_dl=ft_val_dl,
            epochs=epochs_safe,
            optimizer=opt,
            clip_val=1.0,
            loss_fn=base_loss_fn,
            es_patience=8,
            save_best_path=out_path,
            verbose=False,
        )

    elif strategy == "full":
        unfreeze_all(model)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=lr_full,
            weight_decay=params["weight_decay"],
        )
        history, best_val, best_state = model.train_loop(
            device=device,
            train_dl=ft_train_dl,
            val_dl=ft_val_dl,
            epochs=epochs_full,
            optimizer=opt,
            clip_val=1.0,
            loss_fn=base_loss_fn,
            es_patience=10,
            save_best_path=out_path,
            verbose=False,
        )

    elif strategy == "two_stage":
        tmp_stage1 = out_path.replace(".pt", "_stage1_tmp.pt")

        freeze_lstm_only(model)
        opt1 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr_stage1,
            weight_decay=params["weight_decay"],
        )
        model.train_loop(
            device=device,
            train_dl=ft_train_dl,
            val_dl=ft_val_dl,
            epochs=stage1_epochs,
            optimizer=opt1,
            clip_val=1.0,
            loss_fn=base_loss_fn,
            es_patience=5,
            save_best_path=tmp_stage1,
            verbose=False,
        )

        state = torch.load(tmp_stage1, map_location=device)
        model.load_state_dict(state)

        unfreeze_all(model)
        opt2 = torch.optim.AdamW(
            model.parameters(),
            lr=lr_stage2,
            weight_decay=params["weight_decay"],
        )
        history, best_val, best_state = model.train_loop(
            device=device,
            train_dl=ft_train_dl,
            val_dl=ft_val_dl,
            epochs=stage2_epochs,
            optimizer=opt2,
            clip_val=1.0,
            loss_fn=base_loss_fn,
            es_patience=10,
            save_best_path=out_path,
            verbose=False,
        )

        try:
            os.remove(tmp_stage1)
        except OSError:
            pass

    elif strategy == "disc_full":
        unfreeze_all(model)
        opt = torch.optim.AdamW(
            make_param_groups_lstm_mb(
                model,
                lr_lstm=float(params.get("lr_lstm", lr_lstm)),
                lr_static=float(params.get("lr_static", lr_static)),
                lr_head=float(params.get("lr_head", lr_head)),
                weight_decay=params["weight_decay"],
            )
        )
        history, best_val, best_state = model.train_loop(
            device=device,
            train_dl=ft_train_dl,
            val_dl=ft_val_dl,
            epochs=epochs_full,
            optimizer=opt,
            clip_val=1.0,
            loss_fn=base_loss_fn,
            es_patience=10,
            save_best_path=out_path,
            verbose=False,
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # load best
    state = torch.load(out_path, map_location=device)
    model.load_state_dict(state)

    return model, out_path, {"history": history, "best_val": best_val}


def finetune_TL_models_all(
    cfg,
    tl_assets_by_key: dict,  # e.g. tl_assets["TL_CH_to_ISL_5pct"] -> {...}
    best_params: dict,
    device,
    pretrained_ckpt_path: str,
    strategies=("safe", "full", "two_stage", "adapter"),
    train_keys=None,  # optional subset of exp_keys
    force_retrain=False,
    models_dir="models",
    prefix="lstm_TL",
):
    models = {}
    infos = {}

    train_keys_set = set(train_keys) if train_keys else None

    for exp_key in sorted(tl_assets_by_key.keys()):
        if train_keys_set is not None and exp_key not in train_keys_set:
            continue

        assets = tl_assets_by_key[exp_key]
        if assets is None or assets.get("ds_finetune", None) is None:
            logging.warning(f"Skipping {exp_key}: missing finetune dataset.")
            continue

        for strat in strategies:
            run_key = f"{exp_key}__{strat}"
            logging.info(f"\n=== FINETUNE {run_key} ===")

            model, path, info = finetune_or_load_one_TL(
                cfg=cfg,
                exp_key=exp_key,
                tl_assets_for_key=assets,
                best_params=best_params,
                device=device,
                pretrained_ckpt_path=pretrained_ckpt_path,
                models_dir=models_dir,
                prefix=prefix,
                strategy=strat,
                force_retrain=force_retrain,
            )

            models[run_key] = model
            infos[run_key] = {"model_path": path, **(info or {})}

    return models, infos


def train_or_load_CH_baseline(
    cfg,
    tl_assets: dict,  # the whole dict returned by build_transfer_learning_assets
    default_params: dict,
    device,
    models_dir="models",
    prefix="lstm_CH",
    key="BASELINE",
    train_flag=True,
    force_retrain=False,
    epochs=150,
    batch_size_train=64,
    batch_size_val=128,
):
    """
    Trains a CH-only model on ds_pretrain using CH scalers from ds_pretrain_scalers.
    Assumes all tl_assets share the same CH dataset + indices + scaler donor.
    """
    any_key = next(iter(tl_assets.keys()))
    assets0 = tl_assets[any_key]

    ds_train_pristine = assets0["ds_pretrain"]  # pristine CH dataset
    ds_ch_scalers = assets0[
        "ds_pretrain_scalers"
    ]  # scaler donor (fitted on CH train split)
    train_idx = assets0["pretrain_train_idx"]
    val_idx = assets0["pretrain_val_idx"]

    current_date = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{prefix}_{key}_{current_date}.pt")

    # build model + loss
    model = mbm.models.LSTM_MB.build_model_from_params(cfg, default_params, device)
    loss_fn = mbm.models.LSTM_MB.resolve_loss_fn(default_params)

    # load if exists
    if (not train_flag) and os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        return model, model_path, None

    if train_flag and (not force_retrain) and os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        return model, model_path, None

    if (not train_flag) and (not os.path.exists(model_path)):
        raise FileNotFoundError(f"No CH checkpoint found: {model_path}")

    # loaders (DO NOT refit scalers; use ds_ch_scalers)
    mbm.utils.seed_all(cfg.seed)

    ds_train_copy = mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
        ds_train_pristine
    )

    # Apply CH scalers + transform once
    ds_train_copy.set_scalers_from(ds_ch_scalers)
    ds_train_copy.transform_inplace()

    train_dl, val_dl = ds_train_copy.make_loaders(
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size_train=batch_size_train,
        batch_size_val=batch_size_val,
        seed=cfg.seed,
        fit_and_transform=False,  # IMPORTANT: already transformed
        shuffle_train=True,
        use_weighted_sampler=True,
    )

    # fresh checkpoint
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"Deleted existing CH model file: {model_path}")

    history, best_val, best_state = model.train_loop(
        device=device,
        train_dl=train_dl,
        val_dl=val_dl,
        epochs=epochs,
        lr=default_params["lr"],
        weight_decay=default_params["weight_decay"],
        clip_val=1,
        # scheduler
        sched_factor=0.5,
        sched_patience=6,
        sched_threshold=0.01,
        sched_threshold_mode="rel",
        sched_cooldown=1,
        sched_min_lr=1e-6,
        # early stopping
        es_patience=15,
        es_min_delta=1e-4,
        # logging
        log_every=5,
        verbose=False,
        # checkpoint
        save_best_path=model_path,
        loss_fn=loss_fn,
    )

    plot_history_lstm(history)

    # load best
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    return model, model_path, {"history": history, "best_val": best_val}
