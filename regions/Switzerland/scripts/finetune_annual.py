#!/usr/bin/env python3
"""
Fine-tune an already-trained mbm.models.LSTM_MB on ANNUAL-only samples.
- Reuses mixed (annual+winter) scalers from ds_all (no refit)
- Freezes winter head; warmup trains heads only, finetune trains the LSTM body + head_a
- L2-SP regularization (light) to protect the body
- Tiny winter distillation to a frozen teacher to avoid forgetting
"""

import os
import torch
import numpy as np
from typing import Optional, Tuple
import config
import massbalancemachine as mbm


# ---------------------------
# Utilities
# ---------------------------


def make_annual_loaders(
    ds_annual: mbm.data_processing.MBSequenceDataset,
    ds_full: mbm.data_processing.MBSequenceDataset,  # carries fitted (mixed) scalers
    val_ratio: float = 0.2,
    seed: int = 42,
    batch_size_train: int = 64,
    batch_size_val: int = 158,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """Split an already-annual-only dataset; apply ds_full scalers without refitting."""
    n = len(ds_annual)
    tr_idx, va_idx = mbm.data_processing.MBSequenceDataset.split_indices(
        n, val_ratio=val_ratio, seed=seed
    )
    ds_annual.set_scalers_from(ds_full)  # copy mixed scalers
    ds_annual.transform_inplace()  # apply them (no refit)

    train_dl, val_dl = ds_annual.make_loaders(
        tr_idx,
        va_idx,
        batch_size_train=batch_size_train,
        batch_size_val=batch_size_val,
        seed=seed,
        fit_and_transform=False,  # IMPORTANT: do not refit here
        shuffle_train=True,
        drop_last_train=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        use_weighted_sampler=False,
    )
    return train_dl, val_dl


def annual_only_loss(outputs, batch) -> torch.Tensor:
    """MSE on ANNUAL only."""
    _, _, y_a_pred = outputs
    y_true = batch["y"]
    ia = batch["ia"]
    if ia.any():
        return torch.mean((y_a_pred[ia] - y_true[ia]) ** 2)
    return torch.tensor(0.0, device=y_true.device)


def _count_trainable(params):
    return sum(p.numel() for p in params if p.requires_grad)


def _parse_train_loop_ret(va):
    if isinstance(va, tuple):
        if len(va) >= 2 and isinstance(va[1], (float, int)):
            logs = va[0] if isinstance(va[0], dict) else None
            return float(va[1]), logs
        if len(va) >= 1 and isinstance(va[0], dict):
            logs = va[0]
            if "val_loss" in logs and len(logs["val_loss"]) > 0:
                return float(logs["val_loss"][-1]), logs
    if isinstance(va, dict) and "val_loss" in va and len(va["val_loss"]) > 0:
        return float(va["val_loss"][-1]), va
    if isinstance(va, (float, int)):
        return float(va), None
    raise TypeError(f"Don't know how to parse train_loop return: {type(va)}")


# ---------------------------
# Loss = annual + L2-SP (+ winter distillation)
# ---------------------------


def make_loss_fn_with_l2sp_and_distill(
    model,
    ref_state: dict,
    *,
    alpha_l2sp: float = 1e-4,
    exclude_last_idx: int,
    teacher: mbm.models.LSTM_MB,
    lam_distill: float = 0.05,
):
    def l2sp_penalty():
        pen = 0.0
        for k, p in model.state_dict().items():
            if not (k.startswith("lstm.") or k.startswith("static_mlp.")):
                continue
            # let the last LSTM layer move freely
            if f"l{exclude_last_idx}" in k or f"l{exclude_last_idx}_reverse" in k:
                continue
            pen = pen + torch.sum((p - ref_state[k].to(p.device)) ** 2)
        return alpha_l2sp * pen

    @torch.no_grad()
    def teacher_forward(batch):
        return teacher(
            batch["x_m"], batch["x_s"], batch["mv"], batch["mw"], batch["ma"]
        )

    def distill_winter(student_out, teacher_out, iw):
        _, y_w_s, _ = student_out
        _, y_w_t, _ = teacher_out
        if iw.any():
            return lam_distill * torch.mean((y_w_s[iw] - y_w_t[iw]) ** 2)
        return torch.tensor(0.0, device=y_w_s.device)

    def loss_fn(outputs, batch):
        loss = annual_only_loss(outputs, batch) + l2sp_penalty()
        t_out = teacher_forward(batch)
        loss = loss + distill_winter(outputs, t_out, batch["iw"])
        return loss

    return loss_fn


# ---------------------------
# Main
# ---------------------------


def fine_tune_annual(
    ds_all: mbm.data_processing.MBSequenceDataset,
    ds_annual: mbm.data_processing.MBSequenceDataset,
    checkpoint_path: str,
    save_best_path: str,
    model_params: dict,
    cfg: config.Config,
    device: Optional[str] = None,
    warmup_epochs: int = 6,
    finetune_epochs: int = 24,
    lr_head_warmup: float = 5e-4,
    lr_body_warmup: float = 1e-4,
    weight_decay: float = 1e-4,
    clip_val: float = 1.0,
    lam_distill: float = 0.05,
    alpha_l2sp: float = 1e-4,
) -> None:
    """One call to model.train_loop for warmup, one for finetune (so you see LR in logs)."""
    # start fresh
    if os.path.exists(save_best_path):
        os.remove(save_best_path)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- Loaders (reuse mixed scalers) ---
    train_dl, val_dl = make_annual_loaders(ds_annual, ds_all, seed=cfg.seed)

    # --- Student model ---
    model = mbm.models.LSTM_MB.build_model_from_params(cfg, model_params, device)
    state = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in state:  # support both formats
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)

    # --- Teacher (frozen copy) ---
    teacher = mbm.models.LSTM_MB.build_model_from_params(cfg, model_params, device)
    teacher.load_state_dict(state, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # --- Reference for L2-SP ---
    ref_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    last_idx = getattr(model.lstm, "num_layers", 1) - 1
    loss_fn = make_loss_fn_with_l2sp_and_distill(
        model,
        ref_state,
        alpha_l2sp=alpha_l2sp,
        exclude_last_idx=last_idx,
        teacher=teacher,
        lam_distill=lam_distill,
    )

    # --- Freeze winter head entirely ---
    if getattr(model, "two_heads", False):
        for p in model.head_w.parameters():
            p.requires_grad = False

    # ========== WARMUP ==========
    # freeze body; train head_a only
    if hasattr(model, "lstm"):
        for p in model.lstm.parameters():
            p.requires_grad = False
    if hasattr(model, "static_mlp"):
        for p in model.static_mlp.parameters():
            p.requires_grad = False
    head_params = list(
        model.head_a.parameters()
        if getattr(model, "two_heads", False)
        else model.head.parameters()
    )

    optim = torch.optim.AdamW(
        [
            {
                "params": [],
                "lr": lr_body_warmup,
            },  # body frozen; group kept for readability
            {"params": head_params, "lr": lr_head_warmup},
        ],
        weight_decay=weight_decay,
    )

    print(
        "Trainable (warmup) — LSTM:",
        _count_trainable(model.lstm.parameters()),
        "| head_a:" if getattr(model, "two_heads", False) else "| head:",
        _count_trainable(head_params),
    )

    (history_w, best_val_w, best_state_w) = model.train_loop(
        device=device,
        train_dl=train_dl,
        val_dl=val_dl,
        epochs=warmup_epochs,
        optimizer=optim,
        lr=model_params.get(
            "lr", lr_head_warmup
        ),  # shown in your logs, optimizer lr takes precedence
        weight_decay=weight_decay,
        clip_val=clip_val,
        loss_fn=loss_fn,
        sched_patience=4,
        sched_factor=0.5,
        save_best_path=None,  # we’ll save best at the very end
        verbose=True,
        return_best_state=True,
    )

    # ========== FINETUNE ==========
    # unfreeze body (all LSTM layers), keep winter head frozen
    for p in model.parameters():
        p.requires_grad = True
    if getattr(model, "two_heads", False):
        for p in model.head_w.parameters():
            p.requires_grad = False

    # let all LSTM params move
    for p in model.lstm.parameters():
        p.requires_grad = True

    # keep static_mlp mostly frozen, allow last Linear if present
    if hasattr(model, "static_mlp"):
        for p in model.static_mlp.parameters():
            p.requires_grad = False
        last_linear = None
        for m in reversed(model.static_mlp):
            if isinstance(m, torch.nn.Linear):
                last_linear = m
                break
        if last_linear is not None:
            for p in last_linear.parameters():
                p.requires_grad = True

    body_params = [p for p in model.lstm.parameters() if p.requires_grad]
    if hasattr(model, "static_mlp"):
        body_params += [p for p in model.static_mlp.parameters() if p.requires_grad]
    head_params = list(
        model.head_a.parameters()
        if getattr(model, "two_heads", False)
        else model.head.parameters()
    )

    optim = torch.optim.AdamW(
        [
            {"params": body_params, "lr": 1e-4},  # bump if still flat: 2e-4
            {"params": head_params, "lr": 2e-4},
        ],
        weight_decay=weight_decay,
    )

    print(
        "Trainable (finetune) — LSTM:",
        _count_trainable(model.lstm.parameters()),
        "| head_a:" if getattr(model, "two_heads", False) else "| head:",
        _count_trainable(head_params),
    )

    (history_f, best_val_f, best_state_f) = model.train_loop(
        device=device,
        train_dl=train_dl,
        val_dl=val_dl,
        epochs=finetune_epochs,
        optimizer=optim,
        lr=model_params.get("lr", 2e-4),  # just for printing in your loop
        weight_decay=weight_decay,
        clip_val=clip_val,
        loss_fn=loss_fn,
        sched_patience=10,  # gentler scheduler
        sched_factor=0.5,
        save_best_path=save_best_path,  # save best final weights here
        verbose=True,
        return_best_state=True,
    )

    # load best from the finetune phase (train_loop already tracks best)
    if best_state_f is not None:
        torch.save(best_state_f, save_best_path)
        print(f"Saved best fine-tuned weights to: {save_best_path}")
    else:
        print("No improvement recorded during finetune; skipping save.")
