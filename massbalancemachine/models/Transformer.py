from typing import List, Union, Tuple, Optional
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from functools import partial
import json
import os
import random as rd

import config

"""
Model diagram:

    Monthly inputs (B×T×Fm)
        ↓  Linear projection
    (B×T×d_model) + pos_emb(T)
        ↓  TransformerEncoder (Pre-LN, src_key_padding_mask from mv)
    out (B×T×d_model)
        ↓
    ┌───────────────────────────────────────────────┐
    │  Static inputs (B×Fs) → Static MLP → repeat  │
    │               ↓                               │
    │    concat([out, s_rep])  (B×T×fused_dim)      │
    │               ↓  Dropout                      │
    │          [Head(s)]  (B×T×1)                   │
    └───────────────────────────────────────────────┘
        ↓ squeeze + mask mv
    Per-month MB predictions (B×T)
        ↓ masks mw, ma, sum
    Winter MB (B)    Annual MB (B)

Notes
-----
- Drop-in replacement for LSTM_MB: identical forward signature, identical
  train_loop / run_epoch / loss / evaluate_with_preds / predict_with_keys /
  predict_monthly_with_keys interface.
- T_max controls the size of the learned positional embedding table.
  Set it >= the longest sequence you'll ever see (default 32 is safe for
  the standard 15-16 month window).
- Pre-LayerNorm (norm_first=True) is used throughout for stability on
  small glaciological datasets.
"""

# ---------------------------------------------------------------------------
# Positional embedding helper
# ---------------------------------------------------------------------------


class LearnedPositionalEmbedding(nn.Module):
    """Simple learned embedding over absolute month positions."""

    def __init__(self, T_max: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(T_max, d_model)

    def forward(self, T: int, device) -> torch.Tensor:
        """Returns (1, T, d_model)."""
        positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        return self.emb(positions)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class Transformer_MB(nn.Module):
    """
    Transformer encoder model for monthly mass-balance prediction with
    static-feature fusion.

    Identical external interface to LSTM_MB:
      - forward(x_m, x_s, mv, mw, ma, domain_id=None, debug=False,
                return_features=False)
      - train_loop(...)
      - run_epoch(...)
      - evaluate_with_preds(...)
      - predict_with_keys(...)
      - predict_monthly_with_keys(...)
      - build_model_from_params(cfg, params, device)
      - resolve_loss_fn(params)
    """

    def __init__(
        self,
        cfg: config.Config,
        Fm: int,
        Fs: int,
        # --- Transformer encoder ---
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        T_max: int = 32,
        # --- Static MLP ---
        static_hidden: Union[int, List[int]] = 64,
        static_layers: int = 2,
        static_dropout: Optional[float] = None,
        # --- Heads ---
        two_heads: bool = False,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.cfg = cfg
        self.two_heads = two_heads
        self.seed_all()

        assert (
            d_model % nhead == 0
        ), f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        # ---- Monthly input projection ----
        self.input_proj = nn.Linear(Fm, d_model)

        # ---- Learned positional embeddings ----
        self.pos_emb = LearnedPositionalEmbedding(T_max, d_model)

        # ---- Transformer encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN: more stable on small datasets
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            # Final layer norm (Post-encoder)
            norm=nn.LayerNorm(d_model),
        )

        if static_dropout is None:
            static_dropout = dropout

        # ---- Static MLP ----
        if static_layers == 0 or static_hidden is None:
            widths = []
        elif isinstance(static_hidden, int):
            widths = [static_hidden] * static_layers
        else:
            widths = list(static_hidden)

        mlp, in_dim = [], Fs
        for w in widths:
            mlp += [nn.Linear(in_dim, w), nn.ReLU(), nn.Dropout(static_dropout)]
            in_dim = w
        self.static_mlp = nn.Sequential(*mlp) if mlp else nn.Identity()
        static_out_dim = in_dim if mlp else Fs

        fused_dim = d_model + static_out_dim
        self.head_pre_dropout = (
            nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        )

        # ---- Heads ----
        if self.two_heads:
            self.head_w = nn.Linear(fused_dim, 1)
            self.head_a = nn.Linear(fused_dim, 1)
        else:
            self.head = nn.Linear(fused_dim, 1)

    # ------------------------------------------------------------------ #
    #  Reproducibility                                                     #
    # ------------------------------------------------------------------ #

    def seed_all(self):
        rd.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x_m,
        x_s,
        mv,
        mw,
        ma,
        domain_id=None,  # kept for API compatibility; unused
        debug: bool = False,
        return_features: bool = False,
    ):
        """
        x_m : (B, T, Fm)
        x_s : (B, Fs)
        mv  : (B, T)   valid-month mask  (1 = valid, 0 = pad)
        mw  : (B, T)   winter loss mask
        ma  : (B, T)   annual loss mask
        domain_id : ignored (kept for LSTM_MB drop-in compatibility)

        Returns: (y_month, y_w, y_a)
          y_month : (B, T)  — annual head if two_heads, shared head otherwise
          y_w     : (B,)
          y_a     : (B,)
        """
        B, T, _ = x_m.shape

        # ---- Project + positional encoding ----
        x = self.input_proj(x_m)  # (B, T, d_model)
        x = x + self.pos_emb(T, x_m.device)  # (B, T, d_model)

        if debug:
            print(f"[After proj+pos] {tuple(x.shape)}")

        # ---- Transformer (mask padded positions) ----
        # src_key_padding_mask: True where we want to IGNORE the position
        pad_mask = mv == 0  # (B, T)  bool
        out = self.transformer(x, src_key_padding_mask=pad_mask)  # (B, T, d_model)

        if debug:
            print(f"[Transformer out] {tuple(out.shape)}")

        # ---- Static path ----
        s = self.static_mlp(x_s)  # (B, static_out_dim)
        s_rep = s.unsqueeze(1).expand(-1, T, -1)  # (B, T, static_out_dim)

        if debug:
            print(f"[Static MLP out] {tuple(s.shape)}")

        # ---- Fusion ----
        z = torch.cat([out, s_rep], dim=-1)  # (B, T, fused_dim)
        z = self.head_pre_dropout(z)

        if debug:
            print(f"[Fusion z] {tuple(z.shape)}")

        # ---- Heads ----
        if self.two_heads:
            y_month_w = self.head_w(z).squeeze(-1) * mv  # (B, T)
            y_month_a = self.head_a(z).squeeze(-1) * mv  # (B, T)

            y_w = (y_month_w * mw).sum(dim=1)  # (B,)
            y_a = (y_month_a * ma).sum(dim=1)  # (B,)

            if debug:
                print(
                    f"[Head W] {tuple(y_month_w.shape)}  [Head A] {tuple(y_month_a.shape)}"
                )

            result = (y_month_a, y_w, y_a)

        else:
            y_month = self.head(z).squeeze(-1) * mv  # (B, T)
            y_w = (y_month * mw).sum(dim=1)  # (B,)
            y_a = (y_month * ma).sum(dim=1)  # (B,)

            if debug:
                print(f"[Head shared] {tuple(y_month.shape)}")

            result = (y_month, y_w, y_a)

        if return_features:
            return result, z
        return result

    # ------------------------------------------------------------------ #
    #  Training infrastructure  (identical to LSTM_MB)                    #
    # ------------------------------------------------------------------ #

    def train_loop(
        self,
        device,
        train_dl,
        val_dl,
        *,
        epochs: int = 100,
        optimizer=None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        clip_val: float = 1.0,
        loss_fn=None,
        sched_factor: float = 0.5,
        sched_patience: int = 6,
        sched_threshold: float = 0.01,
        sched_threshold_mode: str = "rel",
        sched_cooldown: int = 1,
        sched_min_lr: float = 1e-6,
        es_patience: int = 20,
        es_min_delta: float = 1e-4,
        log_every: int = 5,
        verbose: bool = True,
        return_best_state: bool = True,
        save_best_path: Optional[str] = None,
    ):
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=sched_factor,
            patience=sched_patience,
            threshold=sched_threshold,
            threshold_mode=sched_threshold_mode,
            cooldown=sched_cooldown,
            min_lr=sched_min_lr,
            verbose=verbose,
        )

        history = {"train_loss": [], "val_loss": [], "lr": []}
        best_val, best_state = float("inf"), None
        es_best, es_wait = float("inf"), 0

        for ep in range(1, epochs + 1):
            tr = self.run_epoch(
                device,
                optimizer,
                train_dl,
                loss_fn=loss_fn,
                clip_val=clip_val,
                train=True,
            )
            va = self.run_epoch(
                device,
                optimizer,
                val_dl,
                loss_fn=loss_fn,
                clip_val=clip_val,
                train=False,
            )

            scheduler.step(va)
            if va + es_min_delta < best_val:
                best_val = va
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                }
                if save_best_path is not None:
                    torch.save(best_state, save_best_path)

            if va + es_min_delta < es_best:
                es_best, es_wait = va, 0
            else:
                es_wait += 1

            curr_lr = optimizer.param_groups[0]["lr"]
            if verbose and (ep % log_every == 0 or ep == 1):
                print(
                    f"Epoch {ep:03d} | lr {curr_lr:.2e} | train {tr:.4f} | val {va:.4f} | "
                    f"best {best_val:.4f} | wait {es_wait}/{es_patience}"
                )

            history["train_loss"].append(float(tr))
            history["val_loss"].append(float(va))
            history["lr"].append(float(curr_lr))

            if es_wait >= es_patience:
                if verbose:
                    print(f"Early stopping at epoch {ep} (best val {best_val:.6f}).")
                break

        if best_state is not None:
            self.load_state_dict(best_state)
        return (
            (history, best_val, best_state)
            if return_best_state
            else (history, best_val)
        )

    # ------------------------------------------------------------------ #
    #  Loss functions  (identical to LSTM_MB)                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def custom_loss(outputs, batch) -> torch.Tensor:
        _, y_w_pred, y_a_pred = outputs
        y_true = batch["y"]
        iw, ia = batch["iw"], batch["ia"]
        loss, terms = 0.0, 0
        if iw.any():
            loss = loss + torch.mean((y_w_pred[iw] - y_true[iw]) ** 2)
            terms = terms + 1
        if ia.any():
            loss = loss + torch.mean((y_a_pred[ia] - y_true[ia]) ** 2)
            terms = terms + 1
        return torch.tensor(0.0, device=y_true.device) if terms == 0 else loss / terms

    @staticmethod
    def seasonal_mse_weighted(outputs, batch, w_winter=1.0, w_annual=3.33):
        _, y_w_pred, y_a_pred = outputs
        y_true = batch["y"]
        iw, ia = batch["iw"], batch["ia"]
        loss, terms = 0.0, 0
        if iw.any():
            loss = loss + w_winter * torch.mean((y_w_pred[iw] - y_true[iw]) ** 2)
            terms = terms + 1
        if ia.any():
            loss = loss + w_annual * torch.mean((y_a_pred[ia] - y_true[ia]) ** 2)
            terms = terms + 1
        return torch.tensor(0.0, device=y_true.device) if terms == 0 else loss / terms

    @staticmethod
    @torch.no_grad()
    def to_device(device, batch: dict) -> dict:
        return {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }

    def run_epoch(
        self,
        device,
        optimizer,
        dl,
        *,
        loss_fn=None,
        clip_val: float = 1.0,
        train: bool = True,
    ) -> float:
        if loss_fn is None:
            loss_fn = self.custom_loss
        self.train(train)
        tot, n = 0.0, 0
        with torch.set_grad_enabled(train):
            for batch in dl:
                batch = self.to_device(device, batch)
                outputs = self(
                    batch["x_m"],
                    batch["x_s"],
                    batch["mv"],
                    batch["mw"],
                    batch["ma"],
                    domain_id=batch.get("domain_id", None),
                )
                loss = loss_fn(outputs, batch)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), clip_val)
                    optimizer.step()
                bs = batch["x_m"].shape[0]
                tot += loss.item() * bs
                n += bs
        return tot / max(n, 1)

    # ------------------------------------------------------------------ #
    #  Evaluation / prediction  (identical to LSTM_MB)                    #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def evaluate_with_preds(self, device, dl, ds) -> Tuple[dict, pd.DataFrame]:
        self.eval()
        rows = []
        all_keys = ds.keys
        i = 0
        for batch in dl:
            bs = batch["x_m"].shape[0]
            batch_keys = all_keys[i : i + bs]
            i += bs
            batch = self.to_device(device, batch)
            _, y_w, y_a = self(
                batch["x_m"],
                batch["x_s"],
                batch["mv"],
                batch["mw"],
                batch["ma"],
                domain_id=batch.get("domain_id", None),
            )
            y_true = batch["y"] * ds.y_std.to(device) + ds.y_mean.to(device)
            y_w = y_w * ds.y_std.to(device) + ds.y_mean.to(device)
            y_a = y_a * ds.y_std.to(device) + ds.y_mean.to(device)

            for j in range(bs):
                g, yr, mid, per = batch_keys[j]
                target = float(y_true[j].cpu())
                pred = float((y_w if per == "winter" else y_a)[j].cpu())
                rows.append(
                    {
                        "target": target,
                        "ID": mid,
                        "pred": pred,
                        "PERIOD": per,
                        "GLACIER": g,
                        "YEAR": yr,
                    }
                )

        df = pd.DataFrame(rows)

        def _subset(period):
            d = df[df["PERIOD"] == period]
            return d["pred"].to_numpy(), d["target"].to_numpy(), len(d)

        def rmse(period):
            p, t, n = _subset(period)
            return float(np.sqrt(np.mean((p - t) ** 2))) if n > 0 else float("nan")

        def bias(period):
            p, t, n = _subset(period)
            return float(np.mean(p - t)) if n > 0 else float("nan")

        def r2(period):
            p, t, n = _subset(period)
            if n == 0:
                return float("nan")
            ss_res = np.sum((t - p) ** 2)
            ss_tot = np.sum((t - np.mean(t)) ** 2)
            return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        metrics = {
            "RMSE_winter": rmse("winter"),
            "RMSE_annual": rmse("annual"),
            "Bias_winter": bias("winter"),
            "Bias_annual": bias("annual"),
            "R2_winter": r2("winter"),
            "R2_annual": r2("annual"),
        }
        return metrics, df

    @torch.no_grad()
    def predict_with_keys(self, device, dl, ds, denorm=True) -> pd.DataFrame:
        self.eval()
        rows = []
        all_keys = ds.keys
        i = 0
        for batch in dl:
            bs = batch["x_m"].shape[0]
            keys = all_keys[i : i + bs]
            i += bs
            batch = self.to_device(device, batch)
            _, y_w, y_a = self(
                batch["x_m"],
                batch["x_s"],
                batch["mv"],
                batch["mw"],
                batch["ma"],
                domain_id=batch.get("domain_id", None),
            )
            if denorm:
                y_w = y_w * ds.y_std.to(device) + ds.y_mean.to(device)
                y_a = y_a * ds.y_std.to(device) + ds.y_mean.to(device)

            for j in range(bs):
                g, yr, mid, per = keys[j]
                pred = float((y_w if per == "winter" else y_a)[j].cpu())
                rows.append(
                    {"ID": mid, "pred": pred, "PERIOD": per, "GLACIER": g, "YEAR": yr}
                )
        return pd.DataFrame(rows)

    @torch.no_grad()
    def predict_monthly_with_keys(
        self,
        device,
        dl,
        ds,
        month_names=None,
        denorm=True,
        consistent_denorm=True,
    ) -> pd.DataFrame:
        self.eval()
        if month_names is None:
            month_names = [
                "aug_",
                "sep_",
                "oct",
                "nov",
                "dec",
                "jan",
                "feb",
                "mar",
                "apr",
                "may",
                "jun",
                "jul",
                "aug",
                "sep",
                "oct_",
            ]

        rows = []
        all_keys = ds.keys
        i = 0
        for batch in dl:
            bs = batch["x_m"].shape[0]
            keys = all_keys[i : i + bs]
            i += bs
            batch = self.to_device(device, batch)
            y_month_a, y_w, y_a = self(
                batch["x_m"],
                batch["x_s"],
                batch["mv"],
                batch["mw"],
                batch["ma"],
                domain_id=batch.get("domain_id", None),
            )
            mv = batch["mv"].to(device)
            ma = batch["ma"].to(device)

            if denorm:
                y_std, y_mean = ds.y_std.to(device), ds.y_mean.to(device)
                y_month_norm = y_month_a * mv * ma
                y_a_phys = y_a * y_std + y_mean
                if consistent_denorm:
                    n_active = ma.sum(dim=1, keepdim=True).clamp_min(1.0)
                    bias_corr = y_mean * (1 - n_active)
                    y_month_phys = y_month_norm * y_std + y_mean + bias_corr / n_active
                else:
                    y_month_phys = y_month_a * y_std + y_mean
            else:
                y_month_phys = y_month_a
                y_a_phys = y_a

            y_np = y_month_phys.cpu().numpy()
            y_total = y_a_phys.cpu().numpy()
            mw_np = batch["mw"].cpu().numpy()
            ma_np = ma.cpu().numpy()
            mv_np = mv.cpu().numpy()

            for j in range(bs):
                g, yr, mid, per = keys[j]
                for t_idx in range(y_np.shape[1]):
                    if not mv_np[j, t_idx]:
                        continue
                    month = (
                        month_names[t_idx]
                        if t_idx < len(month_names)
                        else f"m{t_idx:02d}"
                    )
                    rows.append(
                        {
                            "GLACIER": g,
                            "YEAR": yr,
                            "ID": mid,
                            "PERIOD": per,
                            "MONTH_IDX": t_idx,
                            "MONTH": month,
                            "pred_raw": float(y_month_a[j, t_idx].cpu()),
                            "pred_consistent": float(y_np[j, t_idx]),
                            "mw": int(mw_np[j, t_idx]),
                            "ma": int(ma_np[j, t_idx]),
                            "pred_total": float(y_total[j]),
                        }
                    )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  Loss-spec helpers  (identical to LSTM_MB)                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_na(x):
        if x is None:
            return True
        if isinstance(x, str) and x.strip() == "":
            return True
        try:
            return bool(pd.isna(x))
        except Exception:
            return False

    @classmethod
    def _coerce_loss_spec(cls, val):
        if cls._is_na(val):
            return None
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return None
            try:
                val = json.loads(s)
            except Exception:
                return None
        if isinstance(val, (list, tuple)) and len(val) == 2:
            kind, kw = val
            if isinstance(kw, str):
                try:
                    kw = json.loads(kw)
                except Exception:
                    kw = {}
            if kw is None:
                kw = {}
            return (kind, kw)
        return None

    @classmethod
    def resolve_loss_fn(cls, params, verbose=False):
        spec = cls._coerce_loss_spec(params.get("loss_spec"))
        if spec is None:
            if verbose:
                print("[Model Init] Using loss function: default loss")
            return cls.custom_loss
        kind, kw = spec
        if kind == "weighted":
            if verbose:
                print(
                    f"[Model Init] Using loss function: seasonal_mse_weighted with params {kw}"
                )
            return partial(cls.seasonal_mse_weighted, **kw)
        if verbose:
            print("[Model Init] Using loss function: default loss (fallback)")
        return cls.custom_loss

    # ------------------------------------------------------------------ #
    #  Factory  (mirrors LSTM_MB.build_model_from_params)                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def build_model_from_params(cls, cfg, params, device, verbose=True):
        """
        Construct Transformer_MB from a flat params dict.

        Transformer-specific keys (with defaults):
          d_model          int   128
          nhead            int   4
          num_layers       int   2
          dim_feedforward  int   256
          T_max            int   32

        Shared keys (same as LSTM_MB):
          Fm, Fs, dropout, static_layers, static_hidden, static_dropout,
          two_heads, head_dropout, loss_spec
        """
        static_layers = int(params.get("static_layers", 0) or 0)
        static_hidden = params.get("static_hidden", None)
        static_dropout = params.get("static_dropout", None)
        if static_layers == 0:
            static_hidden = None
            static_dropout = None

        init_params = {
            "Fm": int(params["Fm"]),
            "Fs": int(params["Fs"]),
            "d_model": int(params.get("d_model", 128)),
            "nhead": int(params.get("nhead", 4)),
            "num_layers": int(params.get("num_layers", 2)),
            "dim_feedforward": int(params.get("dim_feedforward", 256)),
            "dropout": float(params.get("dropout", 0.1)),
            "T_max": int(params.get("T_max", 32)),
            "static_layers": static_layers,
            "static_hidden": static_hidden,
            "static_dropout": static_dropout,
            "two_heads": bool(params.get("two_heads", False)),
            "head_dropout": float(params.get("head_dropout", 0.0)),
        }

        if verbose:
            print("\n[Model Init] Building Transformer_MB with parameters:")
            for k, v in init_params.items():
                print(f"  {k}: {v}")

        return cls(cfg=cfg, **init_params).to(device)
