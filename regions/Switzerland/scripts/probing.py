# --- Standard library ---
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# --- Third-party ---
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, Ridge
from sklearn.metrics import r2_score, mean_squared_error


@torch.no_grad()
def _cell_sequence(model, x_m: torch.Tensor) -> torch.Tensor:
    """
    Return sequence of top-layer cell states c_t at each time step.
    Works for uni/bi-directional, multi-layer LSTM.
    x_m: (B, T, Fm) on correct device
    """
    B, T, Fm = x_m.shape
    num_layers = model.lstm.num_layers * (2 if model.lstm.bidirectional else 1)
    H = model.lstm.hidden_size
    h = torch.zeros(num_layers, B, H, device=x_m.device)
    c = torch.zeros_like(h)
    seq = []
    for t in range(T):
        out_t, (h, c) = model.lstm(x_m[:, t : t + 1, :], (h, c))
        # take top layer(s)
        if model.lstm.bidirectional or model.lstm.num_layers > 1:
            ct = (
                c[-(2 if model.lstm.bidirectional else 1) :]
                .transpose(0, 1)
                .reshape(B, -1)
            )
        else:
            ct = c.squeeze(0)  # (B, H)
        seq.append(ct.unsqueeze(1))
    return torch.cat(seq, dim=1)  # (B, T, H' or 2H)


@torch.no_grad()
def extract_probe_dataframe(
    model,
    dataloader,
    targets: List[str],  # e.g. ["sd","smlt","sf"] ; each present in batch as (B,T)
    monthly_cols: List[str],  # exact input columns order used to build x_m
    *,
    rep: str = "c",  # 'c' = cell states, 'h' = hidden outputs
    drop_input: Optional[str] = None,
    split_name: str = "SPLIT",
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Build a long DataFrame with one row per VALID month (mv==1):
      ['GLACIER','YEAR','ID','PERIOD','MONTH_IDX','SPLIT', rep_*, x_<col>..., <targets>...]

    Assumes the probing DataLoader was created with shuffle=False and no custom sampler,
    so the iteration order matches the Subset indices (if any).
    """
    device = next(model.parameters()).device
    rows = []

    # ---- Resolve parent keys & optional subset indices (Option B) ----
    ds = dataloader.dataset
    parent_keys = None
    subset_indices = None

    if hasattr(ds, "keys"):
        # base MBSequenceDataset
        parent_keys = ds.keys
    elif isinstance(ds, Subset) and hasattr(ds.dataset, "keys"):
        # Subset of MBSequenceDataset
        parent_keys = ds.dataset.keys
        subset_indices = np.asarray(ds.indices)
    else:
        parent_keys = None  # fallback to dummy keys if truly unavailable

    # Which input channel to zero (ablation)?
    drop_idx = None
    if drop_input is not None:
        if drop_input not in monthly_cols:
            # no-op; silently ignore (useful if you pass a name not in inputs)
            drop_idx = None
        else:
            drop_idx = monthly_cols.index(drop_input)

    # Iterate batches with a progress bar
    iterator = dataloader
    if show_progress:
        iterator = tqdm(
            dataloader,
            desc=f"Extract {split_name} ({rep}, drop={drop_input or 'None'})",
            total=len(dataloader),
            leave=False,
        )

    # Cursor maps batch positions to parent keys or subset indices
    cursor = 0

    for batch in iterator:
        x_m = batch["x_m"].clone().to(device)  # (B, T, Fm)
        mv = batch["mv"].to(device).bool()  # (B, T)
        B, T, Fm = x_m.shape

        # Optional ablation: zero a specific input channel (post-scaling)
        if drop_idx is not None:
            x_m[:, :, drop_idx] = 0.0

        # Representation sequence
        if rep.lower() == "h":
            out, _ = model.lstm(x_m)  # (B, T, H')
            rep_seq = out
        else:
            rep_seq = _cell_sequence(model, x_m)  # (B, T, H' or 2H)

        # Collect monthly targets from batch (expected shape: (B, T))
        tgt_np = {}
        for k in targets:
            if k not in batch:
                raise KeyError(
                    f"Batch missing required target '{k}'. Found: {list(batch.keys())}"
                )
            tgt_np[k] = batch[k].detach().cpu().numpy()

        # For baseline features
        x_np = x_m.detach().cpu().numpy()  # (B, T, Fm)
        rep_np = rep_seq.detach().cpu().numpy()  # (B, T, Hr)
        mv_np = mv.detach().cpu().numpy()  # (B, T)

        # Resolve batch keys robustly:
        if parent_keys is not None:
            if subset_indices is None:
                # Base dataset: slice by cursor
                batch_keys = parent_keys[cursor : cursor + B]
            else:
                # Subset: map slice cursor..cursor+B to the parent indices
                idx_slice = subset_indices[cursor : cursor + B]
                batch_keys = [parent_keys[j] for j in idx_slice]
        else:
            # Fallback (should not happen if you probe on ordered loaders)
            batch_keys = [("UNK", -1, f"id_{cursor+i}", "period") for i in range(B)]
        cursor += B

        # Emit one row per VALID month
        for i in range(B):
            g, yr, mid, per = batch_keys[i]
            for t_idx in range(T):
                if not mv_np[i, t_idx]:
                    continue
                row = {
                    "GLACIER": g,
                    "YEAR": int(yr),
                    "ID": mid,
                    "PERIOD": per,
                    "MONTH_IDX": int(t_idx),
                    "SPLIT": split_name,
                }
                # representation features
                Hr = rep_np.shape[-1]
                for j in range(Hr):
                    row[f"rep_{j}"] = float(rep_np[i, t_idx, j])
                # raw inputs for baseline
                for j, name in enumerate(monthly_cols):
                    row[f"x_{name}"] = float(x_np[i, t_idx, j])
                # targets
                for k in targets:
                    row[k] = float(tgt_np[k][i, t_idx])
                rows.append(row)

    return pd.DataFrame(rows)


def _rep_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("rep_")]


def _raw_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("x_")]


def _Xy(df: pd.DataFrame, cols: List[str], target: str):
    X = df[cols].to_numpy()
    y = df[target].to_numpy()
    return X, y


def _fit_enet(
    X,
    y,
    alpha_grid=np.logspace(-4, 2, 30),
    l1_ratio_grid=np.linspace(0.05, 0.95, 10),
    n_jobs_enet=-1,
) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "enet",
                ElasticNetCV(
                    alphas=alpha_grid,
                    l1_ratio=l1_ratio_grid,
                    fit_intercept=True,
                    max_iter=20000,
                    cv=5,
                    n_jobs=n_jobs_enet,
                    random_state=0,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe


def _metrics(y, yhat):
    return {
        "R2": float(r2_score(y, yhat)),
        "RMSE": float(np.sqrt(mean_squared_error(y, yhat))),
    }


def eval_train_test(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target: str,
    alpha_grid: List[float],
    l1_ratio_grid: List[float],
    n_jobs_enet: int,
    use_raw: bool = False,
) -> Dict[str, float]:
    cols = _raw_cols(df_train) if use_raw else _rep_cols(df_train)
    Xtr, ytr = _Xy(df_train, cols, target)
    Xte, yte = _Xy(df_test, cols, target)
    mdl = _fit_enet(Xtr, ytr, alpha_grid, l1_ratio_grid, n_jobs_enet)
    yhat_tr = mdl.predict(Xtr)
    yhat_te = mdl.predict(Xte)
    return {
        "train_R2": _metrics(ytr, yhat_tr)["R2"],
        "train_RMSE": _metrics(ytr, yhat_tr)["RMSE"],
        "test_R2": _metrics(yte, yhat_te)["R2"],
        "test_RMSE": _metrics(yte, yhat_te)["RMSE"],
    }


# ========= Plotting =========
def _rep_cols(df: pd.DataFrame):
    return [c for c in df.columns if c.startswith("rep_")]


def _raw_cols(df: pd.DataFrame):
    return [c for c in df.columns if c.startswith("x_")]


def _Xy(df: pd.DataFrame, cols, target: str):
    X = df[cols].to_numpy()
    y = df[target].to_numpy()
    return X, y


def add_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Map (YEAR, MONTH_IDX) to dates for the 15-month window Aug..Oct."""
    month_order = [
        "aug",
        "sep",
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
        "oct",
    ]
    month_to_num = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }

    def _row_to_date(row):
        yr = int(row["YEAR"])
        mi = int(row["MONTH_IDX"])
        token = month_order[mi]
        m = month_to_num[token]
        year_for_month = yr - 1 if token in ["aug", "sep", "oct", "nov", "dec"] else yr
        return pd.Timestamp(year_for_month, m, 1)

    df = df.copy()
    df["DATE"] = df.apply(_row_to_date, axis=1)
    return df


def fit_probe_representation_enet(
    df_fit,
    target: str,
    *,
    alphas=None,
    l1_ratios=None,
    max_iter: int = 20000,
    cv: int = 5,
    n_jobs: int = -1,
    random_state: int = 0,
):
    cols = _rep_cols(df_fit)
    if not cols:
        raise ValueError("No rep_* columns found for representation probe.")
    Xtr, ytr = _Xy(df_fit, cols, target)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "enet",
                ElasticNetCV(
                    alphas=alphas if alphas is not None else alphas,
                    l1_ratio=l1_ratios if l1_ratios is not None else l1_ratios,
                    fit_intercept=True,
                    max_iter=max_iter,
                    cv=cv,
                    n_jobs=n_jobs,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipe.fit(Xtr, ytr)
    return pipe


def fit_baseline_raw_enet(
    df_fit,
    target: str,
    *,
    alphas=None,
    l1_ratios=None,
    max_iter: int = 20000,
    cv: int = 5,
    n_jobs: int = -1,
    random_state: int = 0,
):
    cols = _raw_cols(df_fit)
    if not cols:
        raise ValueError("No x_* columns found for baseline.")
    Xtr, ytr = _Xy(df_fit, cols, target)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "enet",
                ElasticNetCV(
                    alphas=alphas if alphas is not None else alphas,
                    l1_ratio=l1_ratios if l1_ratios is not None else l1_ratios,
                    fit_intercept=True,
                    max_iter=max_iter,
                    cv=cv,
                    n_jobs=n_jobs,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipe.fit(Xtr, ytr)
    return pipe


# ---------- 1) Global fitting (all glaciers) ----------
def fit_global_probes(
    df_fit_all: pd.DataFrame,
    target: str,
    *,
    alphas=None,
    l1_ratios=None,
    max_iter: int = 20000,
    cv: int = 5,
    n_jobs: int = -1,
    random_state: int = 0,
):
    """Fit probe models on ALL glaciers (global). Returns (rep_probe, raw_probe)."""
    rep_probe = fit_probe_representation_enet(
        df_fit_all,
        target=target,
        alphas=alphas,
        l1_ratios=l1_ratios,
        max_iter=max_iter,
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    raw_probe = fit_baseline_raw_enet(
        df_fit_all,
        target=target,
        alphas=alphas,
        l1_ratios=l1_ratios,
        max_iter=max_iter,
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    return rep_probe, raw_probe


# ---------- 2) Glacier-specific prediction + plot ----------
def plot_probe_vs_baseline_for_glacier(
    df_te: pd.DataFrame,
    df_fit_all: pd.DataFrame,  # global pool used ONLY for training probes
    target: str,
    glacier_name: str,
    split: str = "TEST",
    *,
    years_plot: Optional[Tuple[int, int]] = None,
    color_rep: str = "tab:blue",
    color_raw: str = "tab:orange",
    color_true: str = "0.6",
    # If your df_* target is standardized, pass TRAIN scalers:
    target_mean: Optional[float] = None,
    target_std: Optional[float] = None,
    # Optionally pass pre-fit global probes to avoid re-fitting each call:
    rep_probe: Optional[Pipeline] = None,
    raw_probe: Optional[Pipeline] = None,
    agg_func: str = "mean",
):
    # ---------- GLOBAL (all glaciers): fit probes if not provided ----------
    if rep_probe is None or raw_probe is None:
        rep_probe, raw_probe = fit_global_probes(df_fit_all, target=target)

    # ---------- choose split source (still multi-glacier at this point) ----------
    split = split.upper()
    split_map = {
        "TRAIN": (
            df_fit_all.query("SPLIT == 'TRAIN'")
            if "SPLIT" in df_fit_all
            else df_fit_all
        ),
        "VAL": (
            df_fit_all.query("SPLIT == 'VAL'") if "SPLIT" in df_fit_all else df_fit_all
        ),
        "TEST": df_te,
    }
    if split not in split_map:
        raise ValueError(f"split must be one of TRAIN/VAL/TEST, got {split}.")
    src = split_map[split].copy()
    # ---------- GLACIER-SPECIFIC filtering + time indexing ----------
    dfg = src[src["GLACIER"] == glacier_name].copy()
    dfg = add_datetime_index(dfg)
    if years_plot is not None:
        y0, y1 = years_plot
        dfg = dfg[(dfg["YEAR"] >= y0) & (dfg["YEAR"] <= y1)]
    dfg = dfg.sort_values(["DATE", "ID", "MONTH_IDX"])

    # ---------- predict on glacier rows only ----------
    rep_cols = _rep_cols(dfg)
    raw_cols = _raw_cols(dfg)
    if not rep_cols or not raw_cols:
        raise ValueError("Missing rep_* or x_* columns in data to plot.")

    mask_rep = ~np.any(dfg[rep_cols].isna().to_numpy(), axis=1)
    mask_raw = ~np.any(dfg[raw_cols].isna().to_numpy(), axis=1)

    dfg["pred_rep"] = np.nan
    dfg.loc[mask_rep, "pred_rep"] = rep_probe.predict(
        dfg.loc[mask_rep, rep_cols].to_numpy()
    )
    dfg["pred_raw"] = np.nan
    dfg.loc[mask_raw, "pred_raw"] = raw_probe.predict(
        dfg.loc[mask_raw, raw_cols].to_numpy()
    )
    dfg["true"] = dfg[target].astype(float)

    # ---------- optional inverse-transform of TARGET only ----------
    if (target_mean is not None) and (target_std is not None):
        dfg["pred_rep"] = dfg["pred_rep"] * target_std + target_mean
        dfg["pred_raw"] = dfg["pred_raw"] * target_std + target_mean
        dfg["true"] = dfg["true"] * target_std + target_mean

    # ---------- aggregate by DATE (mean) ----------
    if agg_func == "mean":
        agg = (
            dfg[["DATE", "true", "pred_rep", "pred_raw"]]
            .groupby("DATE", as_index=True)
            .mean(numeric_only=True)
            .sort_index()
        )
    elif agg_func == "sum":
        agg = (
            dfg[["DATE", "true", "pred_rep", "pred_raw"]]
            .groupby("DATE", as_index=True)
            .sum(numeric_only=True)
            .sort_index()
        )
    else:
        raise ValueError(f"Unsupported agg_func '{agg_func}'")

    # ---------- metrics on aggregated series (matches what we plot) ----------
    # Drop any dates that are NaN in either series before computing metrics
    def _valid(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return a[m], b[m]

    y_true_rep, y_pred_rep = _valid(agg["true"].values, agg["pred_rep"].values)
    y_true_raw, y_pred_raw = _valid(agg["true"].values, agg["pred_raw"].values)

    rmse_rep = (
        float(np.sqrt(mean_squared_error(y_true_rep, y_pred_rep)))
        if len(y_true_rep)
        else np.nan
    )
    r2_rep = float(r2_score(y_true_rep, y_pred_rep)) if len(y_true_rep) else np.nan
    rmse_raw = (
        float(np.sqrt(mean_squared_error(y_true_raw, y_pred_raw)))
        if len(y_true_raw)
        else np.nan
    )
    r2_raw = float(r2_score(y_true_raw, y_pred_raw)) if len(y_true_raw) else np.nan

    # ---------- plot ----------
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        agg.index, agg["true"], "-", color=color_true, lw=1.5, label=f"Target {target}"
    )
    ax.plot(
        agg.index,
        agg["pred_rep"],
        "--",
        color=color_rep,
        lw=2.0,
        alpha=0.6,
        label=f"Probe (rep)",
    )
    ax.plot(
        agg.index,
        agg["pred_raw"],
        ":",
        color=color_raw,
        lw=1.8,
        alpha=0.7,
        label=f"Baseline (raw)",
    )

    # Annotate metrics
    text = (
        f"Probe vs Truth:  RMSE={rmse_rep:.3f}, R²={r2_rep:.3f}\n"
        f"Baseline vs Truth: RMSE={rmse_raw:.3f}, R²={r2_raw:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, lw=0.5),
    )

    title_bits = [glacier_name, f"{target}", f"({split})"]
    if years_plot is not None:
        title_bits.append(f"{years_plot[0]}–{years_plot[1]}")
    ax.set_title(" — ".join(title_bits))
    ax.set_xlabel("Date")
    ax.set_ylabel(target)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()

    ax.legend(loc="lower right")
    # Return metrics along with fig/ax/data
    metrics = {
        "probe_rmse": rmse_rep,
        "probe_r2": r2_rep,
        "baseline_rmse": rmse_raw,
        "baseline_r2": r2_raw,
    }
    return fig, ax, dfg, metrics
