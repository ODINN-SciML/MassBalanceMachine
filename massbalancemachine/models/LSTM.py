# lstm_mb.py
from typing import List, Union, Tuple, Optional
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from functools import partial  # put at top of file if not already
import json, math  # put at top of file if not already
import pandas as pd
import config
import random as rd
import os

"""
Model diagram:
    Monthly inputs (B×15×Fm) ──► LSTM ──────────┐
                                                │
    Static inputs (B×Fs) ──► Static MLP ──► repeat ─► concat ─► Dropout ─► [Head(s)]
                                                │
                                                ▼
                                    Per-month MB predictions (B×15)

            ▼ masks mv, mw, ma
    Winter MB (B)    Annual MB (B)
"""


class LSTM_MB(nn.Module):
    """
    LSTM model for monthly mass-balance prediction with static-feature fusion.
    Supports a single shared head OR two specialized heads (winter/annual).
    """

    def __init__(
        self,
        cfg: config.Config,
        Fm: int,
        Fs: int,
        hidden_size: int = 158,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
        static_hidden: Union[int, List[int]] = 64,
        static_layers: int = 2,
        static_dropout: Optional[float] = None,
        *,
        two_heads: bool = False,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.two_heads = two_heads
        self.cfg = cfg
        self.seed_all()

        # ---- LSTM block ----
        self.lstm = nn.LSTM(
            input_size=Fm,  # number of dynamic features per month
            hidden_size=hidden_size,  # capacity of the recurrent units
            num_layers=num_layers,  # stacked LSTMs for depth
            batch_first=True,
            bidirectional=bidirectional,  # whether to process the sequence forwards and backwards
            dropout=dropout if num_layers > 1 else 0.0,  # applied between LSTM layers
        )

        # Output shape of LSTM block: (B, 15, H) where H = hidden_size × (2 if bidirectional else 1)
        H = hidden_size * (2 if bidirectional else 1)

        if static_dropout is None:
            static_dropout = dropout

        # ---- static MLP ----
        if static_layers == 0 or static_hidden is None:
            widths = []  # identity case where static features go through no MLP
        elif isinstance(static_hidden, int):
            widths = [static_hidden] * static_layers
        else:
            widths = list(static_hidden)

        mlp, in_dim = [], Fs
        # if static_layers > 0, build an MLP with ReLU activations + dropout
        for w in widths:
            mlp += [nn.Linear(in_dim, w), nn.ReLU(), nn.Dropout(static_dropout)]
            in_dim = w
        self.static_mlp = nn.Sequential(*mlp) if mlp else nn.Identity()
        static_out_dim = in_dim if mlp else Fs

        fused_dim = H + static_out_dim
        self.head_pre_dropout = (
            nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        )

        # ---- Two options depending on two_heads ----
        if self.two_heads:
            # Two specialized heads (winter vs annual)
            # Idea: the glacier processes that govern winter MB (accumulation) and annual MB (net balance) differ, so the model learns two separate mappings.
            # Each predicts a per-month series, which is then aggregated differently.
            self.head_w = nn.Linear(fused_dim, 1)  # winter per-month
            self.head_a = nn.Linear(fused_dim, 1)  # annual per-month
        else:
            # Shared head (one output head for all)
            # Produces one per-month MB prediction
            self.head = nn.Linear(fused_dim, 1)  # shared per-month

    def seed_all(self):
        """Sets the random seed everywhere for reproducibility."""
        # Python built-in random
        rd.seed(self.cfg.seed)

        # NumPy random
        np.random.seed(self.cfg.seed)

        # PyTorch seed
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)  # If using multiple GPUs

        # Ensuring deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.use_deterministic_algorithms(True, warn_only=True)
        # Setting CUBLAS environment variable (helps in newer versions)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

    # ----------------
    #  Forward
    # ----------------
    def forward(self, x_m, x_s, mv, mw, ma):
        """
        x_m: (B,15,Fm) | x_s: (B,Fs) | mv,mw,ma: (B,15)
        Returns: y_month, y_w, y_a
        """
        out, _ = self.lstm(x_m)  # (B,15,H or 2H)
        s = self.static_mlp(x_s)  # (B,static_out_dim)

        # ---- Fusion layer ----
        s_rep = s.unsqueeze(1).expand(
            -1, out.size(1), -1
        )  # repeat static along time dimension
        z = torch.cat([out, s_rep], dim=-1)  # concat dynamic + static
        # Output shape: (B, 15, H + static_out_dim)
        z = self.head_pre_dropout(z)

        if self.two_heads:
            y_month_w = self.head_w(z).squeeze(-1)  # (B,15)
            y_month_a = self.head_a(z).squeeze(-1)  # (B,15)

            # mask valid months
            y_month_w = y_month_w * mv
            y_month_a = y_month_a * mv

            # Then it computes seasonal sums (y_w, y_a) depending on which months matter.
            # For a winter meas, this is the sum of the months in mw (e.g. Nov-Apr),
            # And y_a will also be over just winter  months
            # but does not matter because not taken into account in loss
            y_w = (y_month_w * mw).sum(dim=1)  # (B,)

            # For an annual meas, this is the sum of the months in ma (e.g. Oct-Sep).
            # and mw will just be 0 everywhere but again does not matter
            # just ignored in loss
            y_a = (y_month_a * ma).sum(dim=1)  # (B,)

            # keep API: return one per-month series (use annual one for convenience)
            return y_month_a, y_w, y_a
        else:
            y_month = self.head(z).squeeze(-1)  # (B,15)
            y_month = y_month * mv  # mask valid months
            y_w = (y_month * mw).sum(dim=1)  # (B,)
            y_a = (y_month * ma).sum(dim=1)  # (B,)
            return y_month, y_w, y_a

    # ----------------
    #  Train loop / losses
    # ----------------
    def train_loop(
        self,
        device,
        train_dl,
        val_dl,
        *,
        epochs: int = 100,
        optimizer: Optional[torch.optim.Optimizer] = None,
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
                    f"Epoch {ep:03d} | lr {curr_lr:.2e} | train {tr:.4f} | val {va:.4f} | best {best_val:.4f} | wait {es_wait}/{es_patience}"
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

    @staticmethod
    def custom_loss(outputs, batch) -> torch.Tensor:
        _, y_w_pred, y_a_pred = outputs
        y_true = batch["y"]
        iw, ia = batch["iw"], batch["ia"]
        loss, terms = 0.0, 0
        if iw.any():
            loss, terms = loss + torch.mean((y_w_pred[iw] - y_true[iw]) ** 2), terms + 1
        if ia.any():
            loss, terms = loss + torch.mean((y_a_pred[ia] - y_true[ia]) ** 2), terms + 1
        return torch.tensor(0.0, device=y_true.device) if terms == 0 else loss / terms

    @staticmethod
    def seasonal_mse_weighted(outputs, batch, w_winter=1.0, w_annual=3.33):
        _, y_w_pred, y_a_pred = outputs
        y_true = batch["y"]
        iw, ia = batch["iw"], batch["ia"]
        loss, terms = 0.0, 0
        if iw.any():
            loss, terms = (
                loss + w_winter * torch.mean((y_w_pred[iw] - y_true[iw]) ** 2),
                terms + 1,
            )
        if ia.any():
            loss, terms = (
                loss + w_annual * torch.mean((y_a_pred[ia] - y_true[ia]) ** 2),
                terms + 1,
            )
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
                y_m, y_w, y_a = self(
                    batch["x_m"], batch["x_s"], batch["mv"], batch["mw"], batch["ma"]
                )
                loss = loss_fn((y_m, y_w, y_a), batch)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), clip_val)
                    optimizer.step()
                bs = batch["x_m"].shape[0]
                tot += loss.item() * bs
                n += bs
        return tot / max(n, 1)

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
                batch["x_m"], batch["x_s"], batch["mv"], batch["mw"], batch["ma"]
            )
            # invert scaling
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
            # Bias = mean(pred - target); positive = overprediction
            return float(np.mean(p - t)) if n > 0 else float("nan")

        def r2(period):
            p, t, n = _subset(period)
            if n == 0:
                return float("nan")
            ss_res = np.sum((t - p) ** 2)
            ss_tot = np.sum((t - np.mean(t)) ** 2)
            # If variance is ~0, define R^2 as NaN (or 0.0). Using NaN is safer.
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
        """
        Predict seasonal MB for each sequence (no targets required).
        Returns: ID | pred | PERIOD | GLACIER | YEAR
        """
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
                batch["x_m"], batch["x_s"], batch["mv"], batch["mw"], batch["ma"]
            )
            if denorm:
                y_w = y_w * ds.y_std.to(device) + ds.y_mean.to(device)
                y_a = y_a * ds.y_std.to(device) + ds.y_mean.to(device)

            for j in range(bs):
                g, yr, mid, per = keys[j]
                pred = float((y_w if per == "winter" else y_a)[j].cpu())
                rows.append(
                    {
                        "ID": mid,
                        "pred": pred,
                        "PERIOD": per,
                        "GLACIER": g,
                        "YEAR": yr,
                    }
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
        """
        Predict per-month MB for each sequence (masked by mv).
        Returns a long DataFrame with columns:
        ['GLACIER','YEAR','ID','PERIOD','MONTH_IDX','MONTH',
        'pred_raw','pred_consistent','mw','ma','pred_total']
        """
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
                batch["x_m"], batch["x_s"], batch["mv"], batch["mw"], batch["ma"]
            )

            mv = batch["mv"].to(device)
            ma = batch["ma"].to(device)

            # --- denormalization handling ---
            if denorm:
                y_std, y_mean = ds.y_std.to(device), ds.y_mean.to(device)

                # (1) Normalized per-month predictions
                y_month_norm = y_month_a * mv * ma  # apply valid+annual masks

                # (2) Annual prediction from normalized space (already matches model)
                y_a_phys = y_a * y_std + y_mean  # (B,)

                # (3) Compute consistent per-month predictions
                if consistent_denorm:
                    n_active = ma.sum(dim=1, keepdim=True).clamp_min(1.0)
                    bias_corr = y_mean * (1 - n_active)  # scalar offset correction
                    # Apply affine transform with correction so total equals y_a_phys
                    y_month_phys = y_month_norm * y_std + y_mean + bias_corr / n_active
                else:
                    # regular (raw) denormalization
                    y_month_phys = y_month_a * y_std + y_mean
            else:
                y_month_phys = y_month_a
                y_a_phys = y_a

            # --- collect results ---
            y_np = y_month_phys.cpu().numpy()
            y_total = y_a_phys.cpu().numpy()
            mw = batch["mw"].cpu().numpy()
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
                            "pred_raw": float(
                                y_month_a[j, t_idx].cpu()
                            ),  # normalized or denorm raw
                            "pred_consistent": float(
                                y_np[j, t_idx]
                            ),  # physically consistent
                            "mw": int(mw[j, t_idx]),
                            "ma": int(ma_np[j, t_idx]),
                            "pred_total": float(y_total[j]),
                        }
                    )

        return pd.DataFrame(rows)

    @staticmethod
    def _is_na(x):
        # Treat None / '' / NaN (float or numpy/pandas) as missing
        if x is None:
            return True
        if isinstance(x, str) and x.strip() == "":
            return True
        try:
            # works for float('nan'), np.nan, pd.NA, etc.
            return bool(pd.isna(x))
        except Exception:
            return False

    @classmethod
    def _coerce_loss_spec(cls, val):
        """
        Accepts:
        - None / NaN / ''  -> returns None
        - '["weighted", {"w_winter":1.0,"w_annual":2.5}]' (JSON string)
        - ("weighted", {"w_winter":1.0,"w_annual":2.5})
        - ["weighted", {"w_winter":1.0,"w_annual":2.5}]
        Returns:
        - None  or  (kind, kwargs_dict)
        """
        if cls._is_na(val):
            return None

        # If it's a non-empty string, try JSON
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return None
            try:
                val = json.loads(s)
            except Exception:
                return None

        # If it's list/tuple of length 2, (kind, kwargs)
        if isinstance(val, (list, tuple)) and len(val) == 2:
            kind, kw = val
            # kw can be a JSON string as well
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
        """
        Returns a callable loss function based on params['loss_spec'].
        Fallback is cls.custom_loss.
        """
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

    @classmethod
    def build_model_from_params(cls, cfg, params, device, verbose=True):
        """
        Construct LSTM_MB from a flat params dict.
        Also normalizes the static-MLP identity case.
        """
        # Normalize identity static block:
        static_layers = int(params.get("static_layers", 0) or 0)
        static_hidden = params.get("static_hidden", None)
        static_dropout = params.get("static_dropout", None)
        if static_layers == 0:
            static_hidden = None
            static_dropout = None

        # Collect normalized init params for printing
        init_params = {
            "Fm": int(params["Fm"]),
            "Fs": int(params["Fs"]),
            "hidden_size": int(params["hidden_size"]),
            "num_layers": int(params["num_layers"]),
            "bidirectional": bool(params["bidirectional"]),
            "dropout": float(params.get("dropout", 0.0)),
            "static_layers": static_layers,
            "static_hidden": static_hidden,
            "static_dropout": static_dropout,
            "two_heads": bool(params.get("two_heads", False)),
            "head_dropout": float(params.get("head_dropout", 0.0)),
        }

        if verbose:
            print("\n[Model Init] Building model with parameters:")
            for k, v in init_params.items():
                print(f"  {k}: {v}")
        return cls(cfg=cfg, **init_params).to(device)
