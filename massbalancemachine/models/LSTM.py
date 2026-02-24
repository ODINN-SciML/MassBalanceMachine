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
    Monthly inputs (B×16×Fm) ──► LSTM ──────────┐
                                                │
    Static inputs (B×Fs) ──► Static MLP ──► repeat ─► concat ─► Dropout ─► [Head(s)]
                                                │
                                                ▼
                                    Per-month MB predictions (B×16)

            ▼ masks mv, mw, ma
    Winter MB (B)    Annual MB (B)

    And inside the LSTM:
        Input x_t for t in [0, ..., T]
        ↓
        [LSTM Layer 1]
        ↓     produces h_t^(1), c_t^(1)
        [LSTM Layer 2]
        ↓     produces h_t^(2), c_t^(2)

        out = [h_1^(2), h_2^(2), ..., h_T^(2)]   → shape (B, T, H)
        h_n = [h_T^(1), h_T^(2)]                 → shape (2, B, H)
        c_n = [c_T^(1), c_T^(2)]                 → shape (2, B, H)
"""


class BottleneckAdapter(nn.Module):
    """
    Residual bottleneck adapter: x + W_up(ReLU(W_down(LN(x))))
    Works on (..., D) tensors.
    """

    def __init__(
        self,
        dim: int,
        bottleneck: int = 32,
        dropout: float = 0.0,
        use_ln: bool = True,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_ln else nn.Identity()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.up = nn.Linear(bottleneck, dim)

        # Optional: start near-identity by making adapter initially small
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.drop(self.act(self.down(self.norm(x)))))


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
        use_adapter: bool = False,
        adapter_bottleneck: int = 32,
        adapter_dropout: float = 0.0,
        adapter_domainwise: bool = True,  # one adapter per domain
        n_domains: Optional[int] = None,  # required if domainwise=True
        adapter_use_ln: bool = True,
    ):
        super().__init__()
        self.two_heads = two_heads
        self.cfg = cfg
        self.seed_all()

        # ---- LSTM block ----
        self.lstm = nn.LSTM(
            input_size=Fm,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output shape: (B, 16, H) where H = hidden_size × (2 if bidirectional else 1)
        H = hidden_size * (2 if bidirectional else 1)

        if static_dropout is None:
            static_dropout = dropout

        # ---- static MLP ----
        if static_layers == 0 or static_hidden is None:
            widths = []  # identity
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

        fused_dim = H + static_out_dim
        self.head_pre_dropout = (
            nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        )

        # ---- Adapters ----
        self.use_adapter = use_adapter
        self.adapter_domainwise = adapter_domainwise

        if self.use_adapter:
            if self.adapter_domainwise:
                if n_domains is None:
                    raise ValueError(
                        "n_domains must be provided when adapter_domainwise=True."
                    )
                self.n_domains = int(n_domains)
                self.adapters = nn.ModuleList(
                    [
                        BottleneckAdapter(
                            dim=fused_dim,
                            bottleneck=adapter_bottleneck,
                            dropout=adapter_dropout,
                            use_ln=adapter_use_ln,
                        )
                        for _ in range(self.n_domains)
                    ]
                )
            else:
                self.adapter = BottleneckAdapter(
                    dim=fused_dim,
                    bottleneck=adapter_bottleneck,
                    dropout=adapter_dropout,
                    use_ln=adapter_use_ln,
                )

        # ---- Heads ----
        if self.two_heads:
            self.head_w = nn.Linear(fused_dim, 1)
            self.head_a = nn.Linear(fused_dim, 1)
        else:
            self.head = nn.Linear(fused_dim, 1)

    def seed_all(self):
        """Sets the random seed everywhere for reproducibility."""
        rd.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

    # ----------------
    #  Forward
    # ----------------
    def forward(
        self,
        x_m,
        x_s,
        mv,
        mw,
        ma,
        domain_id=None,
        debug=False,
        return_features: bool = False,
    ):
        """
        x_m: (B, 16, Fm) | x_s: (B, Fs) | mv, mw, ma: (B, 16)
        domain_id: None, scalar int/tensor, or tensor of shape (B,)
        Returns: y_month, y_w, y_a
        """

        # ---- Dynamic path ----
        out, (h_n, c_n) = self.lstm(x_m)  # (B, 16, H or 2H)
        if debug:
            print(f"[LSTM out] {tuple(out.shape)}  (H={out.shape[-1]})")
            print(f"[LSTM h_n] {tuple(h_n.shape)}")
            print(f"[LSTM c_n] {tuple(c_n.shape)}")

        # ---- Static path ----
        s = self.static_mlp(x_s)  # (B, static_out_dim)
        if debug:
            print(f"[Static MLP out] {tuple(s.shape)}  (static_out_dim={s.shape[-1]})")

        # ---- Fusion ----
        s_rep = s.unsqueeze(1).expand(-1, out.size(1), -1)  # (B, 16, static_out_dim)
        if debug:
            print(f"[Static repeated] {tuple(s_rep.shape)}")

        z = torch.cat([out, s_rep], dim=-1)  # (B, 16, fused_dim)
        if debug:
            print(f"[Fusion z] {tuple(z.shape)}  (fused_dim={z.shape[-1]})")

        z = self.head_pre_dropout(z)

        # ---- Adapter (OPTION A with safe fallback) ----
        if self.use_adapter:
            if self.adapter_domainwise:
                # If domain_id is missing, fall back to adapter[0]
                if domain_id is None:
                    z = self.adapters[0](z)
                else:
                    if not torch.is_tensor(domain_id):
                        domain_id = torch.tensor(domain_id, device=z.device)
                    domain_id = domain_id.to(z.device)

                    # scalar -> one adapter for whole batch
                    if domain_id.ndim == 0:
                        z = self.adapters[int(domain_id.item())](z)
                    else:
                        # (B,) -> potentially mixed domains
                        z_out = z
                        for d in torch.unique(domain_id).tolist():
                            mask = domain_id == d
                            if mask.any():
                                z_out[mask] = self.adapters[int(d)](z[mask])
                        z = z_out
            else:
                z = self.adapter(z)

        # ---- Heads ----
        if self.two_heads:
            y_month_w = self.head_w(z).squeeze(-1)  # (B, 16)
            y_month_a = self.head_a(z).squeeze(-1)  # (B, 16)

            if debug:
                print(f"[Head W out] {tuple(y_month_w.shape)}")
                print(f"[Head A out] {tuple(y_month_a.shape)}")

            # Mask valid months
            y_month_w = y_month_w * mv
            y_month_a = y_month_a * mv

            y_w = (y_month_w * mw).sum(dim=1)  # (B,)
            y_a = (y_month_a * ma).sum(dim=1)  # (B,)

            if debug:
                print(
                    f"[Seasonal outputs] y_w={tuple(y_w.shape)} | y_a={tuple(y_a.shape)}"
                )

            out = (y_month_a, y_w, y_a)
            if return_features:
                return out, z
            return out

        else:
            y_month = self.head(z).squeeze(-1)  # (B, 16)
            if debug:
                print(f"[Head shared out] {tuple(y_month.shape)}")

            y_month = y_month * mv
            y_w = (y_month * mw).sum(dim=1)
            y_a = (y_month * ma).sum(dim=1)

            if debug:
                print(
                    f"[Seasonal outputs] y_w={tuple(y_w.shape)} | y_a={tuple(y_a.shape)}"
                )

            out = (y_month, y_w, y_a)
            if return_features:
                return out, z
            return out

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
                    batch["x_m"],
                    batch["x_s"],
                    batch["mv"],
                    batch["mw"],
                    batch["ma"],
                    domain_id=batch.get("domain_id", None),
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
                batch["x_m"],
                batch["x_s"],
                batch["mv"],
                batch["mw"],
                batch["ma"],
                domain_id=batch.get("domain_id", None),
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
                batch["x_m"],
                batch["x_s"],
                batch["mv"],
                batch["mw"],
                batch["ma"],
                domain_id=batch.get("domain_id", None),
            )

            mv = batch["mv"].to(device)
            ma = batch["ma"].to(device)

            # --- denormalization handling ---
            if denorm:
                y_std, y_mean = ds.y_std.to(device), ds.y_mean.to(device)

                # (1) Normalized per-month predictions
                y_month_norm = y_month_a * mv * ma

                # (2) Annual prediction from normalized space
                y_a_phys = y_a * y_std + y_mean

                # (3) Compute consistent per-month predictions
                if consistent_denorm:
                    n_active = ma.sum(dim=1, keepdim=True).clamp_min(1.0)
                    bias_corr = y_mean * (1 - n_active)
                    y_month_phys = y_month_norm * y_std + y_mean + bias_corr / n_active
                else:
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
                            "pred_raw": float(y_month_a[j, t_idx].cpu()),
                            "pred_consistent": float(y_np[j, t_idx]),
                            "mw": int(mw[j, t_idx]),
                            "ma": int(ma_np[j, t_idx]),
                            "pred_total": float(y_total[j]),
                        }
                    )

        return pd.DataFrame(rows)

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

    @classmethod
    def build_model_from_params(cls, cfg, params, device, verbose=True):
        """
        Construct LSTM_MB from a flat params dict.
        Also normalizes the static-MLP identity case.
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
            "hidden_size": int(params["hidden_size"]),
            "num_layers": int(params["num_layers"]),
            "bidirectional": bool(params["bidirectional"]),
            "dropout": float(params.get("dropout", 0.0)),
            "static_layers": static_layers,
            "static_hidden": static_hidden,
            "static_dropout": static_dropout,
            "two_heads": bool(params.get("two_heads", False)),
            "head_dropout": float(params.get("head_dropout", 0.0)),
            # Adapter params (optional)
            "use_adapter": bool(params.get("use_adapter", False)),
            "adapter_bottleneck": int(params.get("adapter_bottleneck", 32)),
            "adapter_dropout": float(params.get("adapter_dropout", 0.0)),
            "adapter_domainwise": bool(params.get("adapter_domainwise", True)),
            "n_domains": (
                None
                if params.get("n_domains", None) is None
                else int(params["n_domains"])
            ),
            "adapter_use_ln": bool(params.get("adapter_use_ln", True)),
        }

        if verbose:
            print("\n[Model Init] Building model with parameters:")
            for k, v in init_params.items():
                print(f"  {k}: {v}")

        return cls(cfg=cfg, **init_params).to(device)
