from torch.utils.data import Subset

import massbalancemachine as mbm
import os
import torch

from regions.TF_Europe.scripts.plotting import plot_history_lstm


def finetune_transformer_on_target(
    cfg,
    model_pooled,
    ds_ft,
    ds_pooled_scaler,
    device,
    best_params,
    model_filename,
    strategy="adapter",
    epochs=50,
    lr_factor=0.1,
    force_retrain=False,
):
    import copy

    assert strategy in ("adapter", "head_only", "full")

    model_ft = copy.deepcopy(model_pooled)

    # inject adapter before any load attempt, so architecture matches checkpoint
    if strategy == "adapter" and not model_ft.use_adapter:
        fused_dim = (
            model_ft.head_w.in_features
            if model_ft.two_heads
            else model_ft.head.in_features
        )
        model_ft.adapter = mbm.models.BottleneckAdapter(
            dim=fused_dim,
            bottleneck=int(best_params.get("adapter_bottleneck", 32)),
            dropout=float(best_params.get("adapter_dropout", 0.0)),
            use_ln=bool(best_params.get("adapter_use_ln", True)),
        ).to(device)
        model_ft.use_adapter = True

    if not force_retrain and os.path.exists(model_filename):
        print(f"Loading [{strategy}] from {model_filename}")
        model_ft.load_state_dict(torch.load(model_filename, map_location=device))
        return model_ft

    # --- split indices ---
    train_idx_ft, val_idx_ft = mbm.data_processing.MBSequenceDataset.split_indices(
        len(ds_ft), val_ratio=0.2, seed=cfg.seed
    )

    ds_ft_copy = mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
        ds_ft
    )
    ds_ft_copy.set_scalers_from(ds_pooled_scaler)
    ds_ft_copy.transform_inplace()
    train_dl, val_dl = ds_ft_copy.make_loaders(
        train_idx=train_idx_ft,
        val_idx=val_idx_ft,
        fit_and_transform=False,
        seed=cfg.seed,
        use_weighted_sampler=True,
        verbose=True,
    )

    # --- freeze/unfreeze ---
    if strategy == "head_only":
        for param in model_ft.parameters():
            param.requires_grad = False
        head_modules = (
            [model_ft.head_w, model_ft.head_a]
            if model_ft.two_heads
            else [model_ft.head]
        )
        for m in head_modules:
            for param in m.parameters():
                param.requires_grad = True

    elif strategy == "adapter":
        for param in model_ft.parameters():
            param.requires_grad = False
        for param in model_ft.adapter.parameters():
            param.requires_grad = True

    # "full" -> all params trainable

    n_trainable = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model_ft.parameters())
    print(f"  [{strategy}] {n_trainable} / {n_total} params trainable")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_ft.parameters()),
        lr=best_params["lr"] * lr_factor,
        weight_decay=best_params["weight_decay"],
    )
    loss_fn = mbm.models.Transformer_MB.resolve_loss_fn(best_params)

    history, best_val, _ = model_ft.train_loop(
        device=device,
        train_dl=train_dl,
        val_dl=val_dl,
        epochs=epochs,
        optimizer=optimizer,
        loss_fn=loss_fn,
        es_patience=10,
        es_min_delta=1e-4,
        sched_factor=0.5,
        sched_patience=4,
        sched_min_lr=1e-7,
        log_every=5,
        verbose=True,
        save_best_path=model_filename,
    )

    plot_history_lstm(history)
    model_ft.load_state_dict(torch.load(model_filename, map_location=device))
    return model_ft
