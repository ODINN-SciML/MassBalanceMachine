from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np
import random as rd
import torch
import os
from datetime import datetime
import logging

from torch import nn
import os, logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import massbalancemachine as mbm
from tqdm.auto import tqdm

from regions.TF_Europe.scripts.dataset import (
    make_finetune_loaders_for_exp,
    make_dan_loaders_for_exp,
)
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


def _region_from_assets(exp_key, tl_assets_for_key):
    reg = tl_assets_for_key.get("target_code", None)
    if reg is None:
        try:
            reg = exp_key.split("_to_")[1].split("_")[0]
        except Exception:
            reg = None
    return reg


def _tuned_for(region, strategy, best_by_region):
    """
    Returns (tuned_dict_or_None, tuned_ckpt_or_None)
    """
    if best_by_region is None or region is None or region not in best_by_region:
        return None, None

    if strategy == "adapter":
        tuned = best_by_region[region].get("best_adapter", None)
    elif strategy == "dan":
        tuned = best_by_region[region].get("best_dan", None)
    else:
        tuned = None

    ckpt = tuned.get("ckpt", None) if isinstance(tuned, dict) else None
    return tuned, ckpt


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
    # ---- knobs (optional) ----
    lr_head=5e-5,
    lr_static=1e-5,
    lr_lstm=5e-6,
    # ---- Adapter knobs ----
    lr_adapter=1e-4,
    verbose=False,
    # NEW:
    best_by_region=None,
    date=None,
):
    os.makedirs(models_dir, exist_ok=True)
    if date == None:
        date = datetime.now().strftime("%Y-%m-%d")
    out_name = f"{prefix}_{exp_key}_{strategy}_{date}.pt"
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

    # -------------------------
    # tuned overrides (region-specific)
    # -------------------------
    region = _region_from_assets(exp_key, tl_assets_for_key)
    tuned, tuned_ckpt = _tuned_for(region, strategy, best_by_region)

    # Prefer tuned checkpoint for adapter/dan when not retraining
    load_path = None
    if not force_retrain:
        if tuned_ckpt is not None and os.path.exists(tuned_ckpt):
            load_path = tuned_ckpt
        elif os.path.exists(out_path):
            load_path = out_path

    # -------------------------
    # LOAD
    # -------------------------
    if load_path is not None:
        params = dict(best_params)

        if strategy == "adapter":
            params["use_adapter"] = True

            # apply tuned adapter hyperparams (affects module shapes!)
            if tuned is not None:
                params["adapter_bottleneck"] = int(
                    tuned.get(
                        "adapter_bottleneck", params.get("adapter_bottleneck", 32)
                    )
                )
                params["adapter_dropout"] = float(
                    tuned.get("adapter_dropout", params.get("adapter_dropout", 0.0))
                )

            # ensure n_domains is consistent with vocab used during training
            dv = tl_assets_for_key.get("domain_vocab", None)
            if params.get("adapter_domainwise", True):
                params["n_domains"] = (
                    len(dv) if dv is not None else int(params.get("n_domains", 1))
                )

        # DAN stores base weights only -> build base params (no adapter change required)
        model = mbm.models.LSTM_MB.build_model_from_params(
            cfg, params, device, verbose=verbose
        )
        state = torch.load(load_path, map_location=device)
        model.load_state_dict(state, strict=True)
        return model, load_path, {"loaded": True, "region": region, "tuned": tuned}

    # -------------------------
    # TRAIN loaders
    # -------------------------
    ds_ft_tl, ft_train_dl, ft_val_dl = make_finetune_loaders_for_exp(
        cfg,
        tl_assets_for_key,
        batch_size_train=batch_size_train,
        batch_size_val=batch_size_val,
        domain_vocab=best_params.get("domain_vocab", None),
        verbose=verbose,
    )

    params = dict(best_params)

    # -------------------------
    # Strategy: DAN (delegate)
    # -------------------------
    if strategy == "dan":
        dan_alpha = float(tuned.get("dan_alpha", 0.1)) if tuned else 0.1
        mix_ratio_ft = float(tuned.get("mix_ratio_ft", 1.0)) if tuned else 1.0

        return train_dan_one_TL(
            cfg=cfg,
            exp_key=exp_key,
            tl_assets_for_key=tl_assets_for_key,
            best_params=best_params,
            device=device,
            pretrained_ckpt_path=pretrained_ckpt_path,
            models_dir=models_dir,
            prefix=prefix,
            force_retrain=True,
            batch_size_train=batch_size_train,
            batch_size_val=batch_size_val,
            epochs=epochs_safe,
            lr_backbone=lr_full,
            lr_domain=1e-4,
            dan_alpha=dan_alpha,
            mix_ratio_ft=mix_ratio_ft,
            verbose=verbose,
        )

    # -------------------------
    # Build model + loss
    # -------------------------
    is_adapter_strategy = strategy == "adapter"
    if is_adapter_strategy:
        params["use_adapter"] = True

        if tuned is not None:
            params["adapter_bottleneck"] = int(
                tuned.get("adapter_bottleneck", params.get("adapter_bottleneck", 32))
            )
            params["adapter_dropout"] = float(
                tuned.get("adapter_dropout", params.get("adapter_dropout", 0.0))
            )

        dv = getattr(ds_ft_tl, "domain_vocab", None)
        if params.get("adapter_domainwise", True):
            params["n_domains"] = (
                len(dv) if dv is not None else int(params.get("n_domains", 1))
            )

    model = mbm.models.LSTM_MB.build_model_from_params(
        cfg, params, device, verbose=verbose
    )
    base_loss_fn = mbm.models.LSTM_MB.resolve_loss_fn(params)

    # pretrained load
    pretrained_state = torch.load(pretrained_ckpt_path, map_location=device)
    if is_adapter_strategy:
        model.load_state_dict(pretrained_state, strict=False)
    else:
        model.load_state_dict(pretrained_state, strict=True)

    # fresh checkpoint
    if os.path.exists(out_path):
        os.remove(out_path)

    # -------------------------
    # Strategy: adapter (train adapters + heads)
    # -------------------------
    if strategy == "adapter":
        _freeze_all(model)
        _unfreeze_adapters(model)
        _unfreeze_heads(model)

        lr_use = (
            float(tuned.get("lr_adapter", lr_adapter))
            if tuned
            else float(params.get("lr_adapter", lr_adapter))
        )

        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr_use,
            weight_decay=params["weight_decay"],
        )
        history, best_val, _ = model.train_loop(
            device=device,
            train_dl=ft_train_dl,
            val_dl=ft_val_dl,
            epochs=epochs_safe,
            optimizer=opt,
            clip_val=1.0,
            loss_fn=base_loss_fn,
            es_patience=8,
            save_best_path=out_path,
            verbose=verbose,
        )
        state = torch.load(out_path, map_location=device)
        model.load_state_dict(state, strict=True)
        return (
            model,
            out_path,
            {
                "history": history,
                "best_val": best_val,
                "region": region,
                "tuned": tuned,
            },
        )

    # -------------------------
    # Existing strategies (unchanged)
    # -------------------------
    if strategy == "safe":
        freeze_lstm_only(model)
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr_safe,
            weight_decay=params["weight_decay"],
        )
        history, best_val, _ = model.train_loop(
            device=device,
            train_dl=ft_train_dl,
            val_dl=ft_val_dl,
            epochs=epochs_safe,
            optimizer=opt,
            clip_val=1.0,
            loss_fn=base_loss_fn,
            es_patience=8,
            save_best_path=out_path,
            verbose=verbose,
        )

    elif strategy == "full":
        unfreeze_all(model)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=lr_full,
            weight_decay=params["weight_decay"],
        )
        history, best_val, _ = model.train_loop(
            device=device,
            train_dl=ft_train_dl,
            val_dl=ft_val_dl,
            epochs=epochs_full,
            optimizer=opt,
            clip_val=1.0,
            loss_fn=base_loss_fn,
            es_patience=10,
            save_best_path=out_path,
            verbose=verbose,
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
            verbose=verbose,
        )

        state = torch.load(tmp_stage1, map_location=device)
        model.load_state_dict(state)

        unfreeze_all(model)
        opt2 = torch.optim.AdamW(
            model.parameters(),
            lr=lr_stage2,
            weight_decay=params["weight_decay"],
        )
        history, best_val, _ = model.train_loop(
            device=device,
            train_dl=ft_train_dl,
            val_dl=ft_val_dl,
            epochs=stage2_epochs,
            optimizer=opt2,
            clip_val=1.0,
            loss_fn=base_loss_fn,
            es_patience=10,
            save_best_path=out_path,
            verbose=verbose,
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
        history, best_val, _ = model.train_loop(
            device=device,
            train_dl=ft_train_dl,
            val_dl=ft_val_dl,
            epochs=epochs_full,
            optimizer=opt,
            clip_val=1.0,
            loss_fn=base_loss_fn,
            es_patience=10,
            save_best_path=out_path,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    state = torch.load(out_path, map_location=device)
    model.load_state_dict(state, strict=True)
    return (
        model,
        out_path,
        {"history": history, "best_val": best_val, "region": region, "tuned": tuned},
    )


def finetune_TL_models_all(
    cfg,
    tl_assets_by_key: dict,
    best_params: dict,
    device,
    pretrained_ckpt_path: str,
    strategies=("safe", "full", "two_stage"),
    train_keys=None,
    regions_only=None,
    split_name_only=None,
    force_retrain=False,
    models_dir="models",
    prefix="lstm_TL",
    verbose=False,
    best_by_region=None,
    date=None,
):
    models = {}
    infos = {}
    # ---------------------------------------
    # Resolve date once for the whole batch
    # ---------------------------------------
    if date is None:
        run_date = datetime.now().strftime("%Y-%m-%d")
    else:
        run_date = date

    train_keys_set = set(train_keys) if train_keys else None
    regions_set = set(regions_only) if regions_only is not None else None

    if split_name_only is None:
        split_set = None
    elif isinstance(split_name_only, (list, tuple, set)):
        split_set = set(split_name_only)
    else:
        split_set = {split_name_only}

    # ---------------------------------------------------------
    # 1️⃣  Build task list first (after filtering)
    # ---------------------------------------------------------
    tasks = []

    for exp_key in sorted(tl_assets_by_key.keys()):
        assets = tl_assets_by_key.get(exp_key, None)
        if assets is None:
            continue

        if train_keys_set is not None and exp_key not in train_keys_set:
            continue

        if regions_set is not None:
            reg = assets.get("target_code", None)
            if reg not in regions_set:
                continue

        if split_set is not None:
            sp = assets.get("split_name", None)
            if sp not in split_set:
                continue

        if assets.get("ds_finetune", None) is None:
            logging.warning(f"Skipping {exp_key}: missing finetune dataset.")
            continue

        for strat in strategies:
            tasks.append((exp_key, strat))

    # ---------------------------------------------------------
    # 2️⃣  Progress bar over true number of runs
    # ---------------------------------------------------------
    pbar = tqdm(tasks, desc="Finetuning TL models")

    for exp_key, strat in pbar:

        assets = tl_assets_by_key[exp_key]
        run_key = f"{exp_key}__{strat}"

        # show live info
        pbar.set_postfix(
            {"exp": exp_key.split("_seed")[0][-12:], "strat": strat}  # short display
        )

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
            verbose=verbose,
            best_by_region=best_by_region,
            date=run_date,
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
    verbose=False,
    date=None,
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

    if date == None:
        date = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{prefix}_{key}_{date}.pt")

    # build model + loss
    model = mbm.models.LSTM_MB.build_model_from_params(
        cfg, default_params, device, verbose=verbose
    )
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
        verbose=verbose,
        # checkpoint
        save_best_path=model_path,
        loss_fn=loss_fn,
    )

    plot_history_lstm(history)

    # load best
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    return model, model_path, {"history": history, "best_val": best_val}


# -------------------------------- DAN --------------------------------


def train_dan_one_TL(
    cfg,
    exp_key: str,
    tl_assets_for_key: dict,
    best_params: dict,
    device,
    pretrained_ckpt_path: str,
    *,
    models_dir="models",
    prefix="lstm_TL",
    force_retrain=False,
    # loaders
    batch_size_train=64,
    batch_size_val=128,
    mix_ratio_ft=1.0,
    # DAN knobs
    dan_alpha=0.1,
    grl_lambda=1.0,
    pool="mean",  # "mean" or "last"
    disc_hidden=128,
    disc_dropout=0.1,
    # optimization
    epochs=60,
    lr_backbone=1e-5,
    lr_domain=1e-4,
    weight_decay=None,
    clip_val=1.0,
    es_patience=8,
    log_every=5,
    verbose=True,
):

    def _domain_metrics(dom_logits: torch.Tensor, dom_y: torch.Tensor):
        """
        dom_logits: (B, n_domains)
        dom_y: (B,) long
        Returns: (loss_ce, acc)
        """
        loss = F.cross_entropy(dom_logits, dom_y)
        pred = dom_logits.argmax(dim=1)
        acc = (pred == dom_y).float().mean()
        return loss, acc

    os.makedirs(models_dir, exist_ok=True)
    current_date = datetime.now().strftime("%Y-%m-%d")
    strategy = "dan"
    out_name = f"{prefix}_{exp_key}_{strategy}_{current_date}.pt"
    out_path = os.path.join(models_dir, out_name)

    # load if exists
    if (not force_retrain) and os.path.exists(out_path):
        base = mbm.models.LSTM_MB.build_model_from_params(
            cfg, best_params, device, verbose=verbose
        )
        state = torch.load(out_path, map_location=device)
        base.load_state_dict(state)
        return base, out_path, None

    # -------------------------
    # loaders
    # -------------------------
    domain_vocab, train_dl, val_dl = make_dan_loaders_for_exp(
        cfg,
        tl_assets_for_key,
        batch_size_train=batch_size_train,
        batch_size_val=batch_size_val,
        seed=cfg.seed,
        domain_vocab=tl_assets_for_key.get("domain_vocab", None),
        mix_ratio_ft=mix_ratio_ft,
        verbose=verbose,
    )
    n_domains = len(domain_vocab)

    # -------------------------
    # build base model + loss
    # -------------------------
    params = dict(best_params)
    base = mbm.models.LSTM_MB.build_model_from_params(
        cfg, params, device, verbose=verbose
    )
    base_loss_fn = mbm.models.LSTM_MB.resolve_loss_fn(params)

    # pretrained CH weights
    pretrained_state = torch.load(pretrained_ckpt_path, map_location=device)
    base.load_state_dict(pretrained_state)

    # wrap
    dan = mbm.models.LSTM_MB_DAN(
        base=base,
        n_domains=n_domains,
        grl_lambda=grl_lambda,
        dan_alpha=dan_alpha,
        pool=pool,
        disc_hidden=disc_hidden,
        disc_dropout=disc_dropout,
    ).to(device)

    # -------------------------
    # optimizer: two param groups
    # -------------------------
    if weight_decay is None:
        weight_decay = float(params.get("weight_decay", 1e-4))

    opt = torch.optim.AdamW(
        [
            {"params": dan.base.parameters(), "lr": float(lr_backbone)},
            {"params": dan.domain_disc.parameters(), "lr": float(lr_domain)},
        ],
        weight_decay=weight_decay,
    )

    # overwrite if retraining
    if os.path.exists(out_path):
        os.remove(out_path)
        if verbose:
            logging.info(f"[{exp_key}] Deleted existing TL checkpoint: {out_path}")

    # -------------------------
    # training loop
    # -------------------------
    best_val = float("inf")
    wait = 0
    history = {"train_loss": [], "val_loss": [], "dom_loss": [], "dom_acc": []}

    for ep in range(1, epochs + 1):
        # ---- train ----
        dan.train()
        tr_tot, tr_n = 0.0, 0

        dom_tot_loss, dom_tot_acc = 0.0, 0.0
        dom_n = 0

        for batch in train_dl:
            batch = mbm.models.LSTM_MB.to_device(device, batch)

            out = dan(
                batch["x_m"],
                batch["x_s"],
                batch["mv"],
                batch["mw"],
                batch["ma"],
                domain_id=batch.get("domain_id", None),
            )
            # out = (y_m, y_w, y_a, dom_logits)
            y_m, y_w, y_a, dom_logits = out

            loss = dan.dan_loss(out, batch, base_loss_fn)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(dan.parameters(), clip_val)
            opt.step()

            bs = batch["x_m"].shape[0]
            tr_tot += float(loss.item()) * bs
            tr_n += bs

            # ---- domain metrics ----
            dom_y = batch.get("domain_id", None)
            if dom_y is not None:
                if not torch.is_tensor(dom_y):
                    dom_y = torch.tensor(
                        dom_y, device=dom_logits.device, dtype=torch.long
                    )
                dom_y = dom_y.to(dom_logits.device).long()

                dloss, dacc = _domain_metrics(dom_logits.detach(), dom_y)
                dom_tot_loss += float(dloss.item()) * bs
                dom_tot_acc += float(dacc.item()) * bs
                dom_n += bs

        tr_loss = tr_tot / max(tr_n, 1)
        dom_loss = dom_tot_loss / max(dom_n, 1) if dom_n > 0 else float("nan")
        dom_acc = dom_tot_acc / max(dom_n, 1) if dom_n > 0 else float("nan")

        # ---- val (FT only): task loss only ----
        dan.eval()
        va_tot, va_n = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                batch = mbm.models.LSTM_MB.to_device(device, batch)
                y_m, y_w, y_a = dan.base(
                    batch["x_m"],
                    batch["x_s"],
                    batch["mv"],
                    batch["mw"],
                    batch["ma"],
                    domain_id=batch.get("domain_id", None),
                )
                loss_task = base_loss_fn((y_m, y_w, y_a), batch)
                bs = batch["x_m"].shape[0]
                va_tot += float(loss_task.item()) * bs
                va_n += bs

        va_loss = va_tot / max(va_n, 1)

        history["train_loss"].append(float(tr_loss))
        history["val_loss"].append(float(va_loss))
        history["dom_loss"].append(float(dom_loss))
        history["dom_acc"].append(float(dom_acc))

        if verbose and (ep == 1 or ep % log_every == 0):
            print(
                f"[{exp_key}][DAN] ep {ep:03d} | "
                f"train {tr_loss:.4f} | val(task) {va_loss:.4f} | "
                f"dom_loss {dom_loss:.4f} | dom_acc {dom_acc*100:.1f}% | "
                f"best {best_val:.4f} | wait {wait}/{es_patience}"
            )

        # ---- early stopping on val(task) ----
        if va_loss < best_val - 1e-6:
            best_val = va_loss
            wait = 0
            # save base weights only
            best_state = {
                k: v.detach().cpu().clone() for k, v in dan.base.state_dict().items()
            }
            torch.save(best_state, out_path)
        else:
            wait += 1
            if wait >= es_patience:
                if verbose:
                    print(
                        f"[{exp_key}][DAN] Early stopping at ep {ep} (best val(task)={best_val:.6f})"
                    )
                break

    # load best base state
    state = torch.load(out_path, map_location=device)
    base.load_state_dict(state)

    return (
        base,
        out_path,
        {"history": history, "best_val": best_val, "domain_vocab": domain_vocab},
    )
