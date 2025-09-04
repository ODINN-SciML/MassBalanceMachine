# lstm_mb.py
from typing import List, Union, Tuple, Optional
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

class LSTM_MB(nn.Module):
    """
    LSTM model for monthly mass-balance prediction with static-feature fusion.
    Provides built-in training/evaluation helpers.
    """

    def __init__(
            self,
            Fm: int,  # monthly feature count
            Fs: int,  # static feature count
            hidden_size: int = 128,  # LSTM hidden size
            num_layers: int = 1,  # LSTM layers
            bidirectional: bool = True,
            dropout: float = 0.1,  # LSTM inter-layer dropout (if num_layers > 1)
            static_hidden: Union[int,
                                 List[int]] = 64,  # static MLP hidden size(s)
            static_layers: int = 2,  # only used if static_hidden is int
            static_dropout: Optional[
                float] = None,  # dropout inside static MLP; default = dropout
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=Fm,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        H = hidden_size * (2 if bidirectional else 1)

        if static_dropout is None:
            static_dropout = dropout

        # ---- build static MLP ----
        if isinstance(static_hidden, int):
            widths = [static_hidden] * max(0, static_layers)
        else:
            widths = list(static_hidden)

        mlp = []
        in_dim = Fs
        for w in widths:
            mlp += [
                nn.Linear(in_dim, w),
                nn.ReLU(),
                nn.Dropout(static_dropout)
            ]
            in_dim = w
        self.static_mlp = nn.Sequential(*mlp) if mlp else nn.Identity()
        static_out_dim = in_dim if mlp else Fs  # if Identity, pass-through Fs

        # ---- monthly head ----
        self.head = nn.Linear(H + static_out_dim, 1)  # per-month scalar

    # ----------------
    #  Forward / loss
    # ----------------
    def forward(self, x_m, x_s, mv, mw, ma):
        """
        x_m: (B, 12, Fm)
        x_s: (B, Fs)
        mv:  (B, 12)   valid-month mask
        mw:  (B, 12)   winter mask (Oct..Apr)
        ma:  (B, 12)   annual mask (Oct..Sep)
        """
        out, _ = self.lstm(x_m)  # (B, 12, H or 2H)
        s = self.static_mlp(x_s)  # (B, static_out_dim)
        s_rep = s.unsqueeze(1).expand(-1, out.size(1), -1)
        z = torch.cat([out, s_rep], dim=-1)  # (B, 12, H+static_out_dim)

        y_month = self.head(z).squeeze(-1)  # (B, 12)
        y_month = y_month * mv  # zero out invalid months

        y_w = (y_month * mw).sum(dim=1)  # (B,)
        y_a = (y_month * ma).sum(dim=1)  # (B,)
        return y_month, y_w, y_a
    
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
        # ReduceLROnPlateau params:
        sched_factor: float = 0.5,
        sched_patience: int = 6,
        sched_threshold: float = 0.01,
        sched_threshold_mode: str = "rel",
        sched_cooldown: int = 1,
        sched_min_lr: float = 1e-6,
        # Early stopping:
        es_patience: int = 20,
        es_min_delta: float = 1e-4,
        # Logging:
        log_every: int = 5,
        verbose: bool = True,
        # Checkpoint:
        return_best_state: bool = True,
        save_best_path: Optional[str] = None,
    ):
        """
        Train the model with ReduceLROnPlateau + EarlyStopping.

        Returns
        -------
        history : dict with lists 'train_loss', 'val_loss', 'lr'
        best_val : float
        best_state : dict (only if return_best_state=True)
        """
        # Optimizer
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        # Scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=sched_factor,
            patience=sched_patience,
            threshold=sched_threshold,
            threshold_mode=sched_threshold_mode,
            cooldown=sched_cooldown,
            min_lr=sched_min_lr,
            verbose=verbose,
        )

        history = {"train_loss": [], "val_loss": [], "lr": []}
        best_val = float('inf')
        best_state = None

        # Early stopping state
        es_best = float('inf')
        es_wait = 0

        for ep in range(1, epochs + 1):
            tr = self.run_epoch(device, optimizer, train_dl, loss_fn=loss_fn, clip_val=clip_val, train=True)
            va = self.run_epoch(device, optimizer, val_dl,   loss_fn=loss_fn, clip_val=clip_val, train=False)

            scheduler.step(va)

            # Save best checkpoint (by val loss)
            if va + es_min_delta < best_val:
                best_val = va
                best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                if save_best_path is not None:
                    torch.save(best_state, save_best_path)

            # Early stopping bookkeeping
            if va + es_min_delta < es_best:
                es_best = va
                es_wait = 0
            else:
                es_wait += 1

            curr_lr = optimizer.param_groups[0]['lr']
            history["train_loss"].append(float(tr))
            history["val_loss"].append(float(va))
            history["lr"].append(float(curr_lr))

            if verbose and (ep % log_every == 0 or ep == 1):
                print(f"Epoch {ep:03d} | lr {curr_lr:.2e} | train {tr:.4f} | val {va:.4f} | best {best_val:.4f} | wait {es_wait}/{es_patience}")

            if es_wait >= es_patience:
                if verbose:
                    print(f"Early stopping at epoch {ep} (best val {best_val:.6f}).")
                break

        # Restore best weights in-memory (common pattern)
        if best_state is not None:
            self.load_state_dict(best_state)

        if return_best_state:
            return history, best_val, best_state
        else:
            return history, best_val

    @staticmethod
    def custom_loss(outputs, batch) -> torch.Tensor:
        """
        Compute MSE on seasonal sums in *scaled* target space.
        Expects batch keys: 'y', 'iw', 'ia'.
        """
        _, y_w_pred, y_a_pred = outputs
        y_true = batch['y']  # already scaled

        iw, ia = batch['iw'], batch['ia']
        loss = 0.0
        terms = 0

        if iw.any():
            loss = loss + torch.mean((y_w_pred[iw] - y_true[iw])**2)
            terms += 1
        if ia.any():
            loss = loss + torch.mean((y_a_pred[ia] - y_true[ia])**2)
            terms += 1

        if terms == 0:
            return torch.tensor(0.0, device=y_true.device)
        return loss / terms

    # ----------------
    #  Utils / helpers
    # ----------------
    @staticmethod
    @torch.no_grad()
    def to_device(device, batch: dict) -> dict:
        """Move a batch dict to a device."""
        return {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

    def run_epoch(
        self,
        device,
        optimizer: torch.optim.Optimizer,
        dl,
        *,
        loss_fn=None,
        clip_val: float = 1.0,
        train: bool = True,
    ) -> float:
        """
        One training/validation epoch. If loss_fn is None, uses custom_loss.
        """
        if loss_fn is None:
            loss_fn = self.custom_loss

        self.train(train)
        tot, n = 0.0, 0
        with torch.set_grad_enabled(train):
            for batch in dl:
                batch = self.to_device(device, batch)
                y_m, y_w, y_a = self(batch['x_m'], batch['x_s'], batch['mv'],
                                     batch['mw'], batch['ma'])
                loss = loss_fn((y_m, y_w, y_a), batch)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), clip_val)
                    optimizer.step()
                bs = batch['x_m'].shape[0]
                tot += loss.item() * bs
                n += bs
        return tot / max(n, 1)

    @torch.no_grad()
    def evaluate_with_preds(self, device, dl, ds) -> Tuple[dict, pd.DataFrame]:
        """
        Evaluate on a dataloader, returning:
          - metrics dict (RMSE_winter, RMSE_annual) in ORIGINAL units
          - DataFrame with columns: target | ID | pred | PERIOD | GLACIER | YEAR

        Assumes ds.keys aligns with dl order and ds.y_mean/std are set
        (i.e., ds was scaled using train statistics).
        """
        self.eval()
        rows = []
        all_keys = ds.keys  # list of (GLACIER, YEAR, ID, PERIOD)

        i = 0  # running index to map back to keys
        for batch in dl:
            batch_size = batch['x_m'].shape[0]
            batch_keys = all_keys[i:i + batch_size]
            i += batch_size

            batch = self.to_device(device, batch)
            _, y_w, y_a = self(batch['x_m'], batch['x_s'], batch['mv'],
                               batch['mw'], batch['ma'])

            # invert scaling
            y_true = batch['y'] * ds.y_std.to(device) + ds.y_mean.to(device)
            y_w = y_w * ds.y_std.to(device) + ds.y_mean.to(device)
            y_a = y_a * ds.y_std.to(device) + ds.y_mean.to(device)

            for j in range(batch_size):
                g, yr, mid, per = batch_keys[j]
                target = float(y_true[j].cpu())
                if per == "winter":
                    pred = float(y_w[j].cpu())
                elif per == "annual":
                    pred = float(y_a[j].cpu())
                else:
                    raise ValueError(f"Unexpected PERIOD: {per}")
                rows.append({
                    "target": target,
                    "ID": mid,
                    "pred": pred,
                    "PERIOD": per,
                    "GLACIER": g,
                    "YEAR": yr
                })

        df_preds = pd.DataFrame(rows)

        def rmse(df, period):
            d = df[df["PERIOD"] == period]
            if len(d) == 0:
                return float("nan")
            return float(np.sqrt(((d["pred"] - d["target"])**2).mean()))

        metrics = {
            "RMSE_winter": rmse(df_preds, "winter"),
            "RMSE_annual": rmse(df_preds, "annual"),
        }
        return metrics, df_preds
