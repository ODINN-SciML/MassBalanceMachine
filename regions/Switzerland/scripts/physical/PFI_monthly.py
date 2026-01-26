from joblib import Parallel, delayed
from tqdm.auto import tqdm
import numpy as np
import torch
import pandas as pd


import massbalancemachine as mbm


@torch.no_grad()
def eval_rmse_pfi_monthly(model, device, dl, ds):
    """
    Compute denormalized RMSE for winter and annual samples (period-aware).

    Parameters
    ----------
    model : torch.nn.Module
        Model returning (y_hat, y_winter, y_annual).
    device : torch.device
        Device used for evaluation.
    dl : torch.utils.data.DataLoader
        DataLoader over `ds` (must be shuffle=False to align with ds.keys).
    ds : Dataset
        Dataset providing `keys`, `y_mean`, and `y_std`.

    Returns
    -------
    rmse_winter : float
        RMSE computed on samples whose key period is 'winter'.
    rmse_annual : float
        RMSE computed on samples whose key period is 'annual'.
    """
    model.eval()
    y_true_w, y_pred_w = [], []
    y_true_a, y_pred_a = [], []

    all_keys = ds.keys
    i = 0
    y_std = ds.y_std.to(device)
    y_mean = ds.y_mean.to(device)

    for batch in dl:
        bs = batch["x_m"].shape[0]
        batch_keys = all_keys[i : i + bs]
        i += bs

        x_m = batch["x_m"].to(device)
        x_s = batch["x_s"].to(device)
        mv = batch["mv"].to(device)
        mw = batch["mw"].to(device)
        ma = batch["ma"].to(device)
        y = batch["y"].to(device)

        _, y_w, y_a = model(x_m, x_s, mv, mw, ma)

        y_true = y * y_std + y_mean
        y_w = y_w * y_std + y_mean
        y_a = y_a * y_std + y_mean

        for j in range(bs):
            *_, per = batch_keys[j]
            if per == "winter":
                y_true_w.append(float(y_true[j].cpu()))
                y_pred_w.append(float(y_w[j].cpu()))
            else:
                y_true_a.append(float(y_true[j].cpu()))
                y_pred_a.append(float(y_a[j].cpu()))

    def rmse(t, p):
        if len(t) == 0:
            return float("nan")
        return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(t)) ** 2)))

    return rmse(y_true_w, y_pred_w), rmse(y_true_a, y_pred_a)


def _pfi_month_worker(task, model, device, ds_test, base_w, base_a, n_repeats, seed):
    """
    Worker function for month-specific permutation feature importance (PFI).

    Permutes a single feature either for a specific month slice (monthly features)
    or as a static feature, re-evaluates RMSE_winter and RMSE_annual, and returns
    mean/std of ΔRMSE across repeats.

    Parameters
    ----------
    task : tuple
        (kind, k, t) where kind is 'monthly' or 'static', k is feature index,
        and t is month index.
    model : torch.nn.Module
        Trained model.
    device : torch.device
        Device used for inference.
    ds_test : Dataset
        Test dataset clone used for permutation.
    base_w, base_a : float
        Baseline winter and annual RMSE.
    n_repeats : int
        Number of permutations to evaluate.
    seed : int
        Base random seed.

    Returns
    -------
    tuple
        (kind, k, t, mean_dw, std_dw, mean_da, std_da) where dw/da are ΔRMSE
        for winter/annual respectively.
    """
    kind, k, t = task
    rng = np.random.default_rng(seed + k * 1000 + t)

    Xm0 = ds_test.Xm.clone()
    Xs0 = ds_test.Xs.clone()

    dw, da = [], []

    for _ in range(n_repeats):
        perm = rng.permutation(len(ds_test))

        if kind == "monthly":
            ds_test.Xm[:, t, k] = Xm0[perm, t, k]
        else:  # static
            ds_test.Xs[:, k] = Xs0[perm, k]

        dl = torch.utils.data.DataLoader(ds_test, batch_size=128, shuffle=False)
        w, a = eval_rmse_pfi_monthly(model, device, dl, ds_test)

        dw.append(w - base_w)
        da.append(a - base_a)

    return kind, k, t, np.mean(dw), np.std(dw), np.mean(da), np.std(da)


def PFI_LSTM_monthly_parallel(
    model,
    device,
    ds_test,
    monthly_cols,
    static_cols,
    month_names,
    n_repeats=5,
    n_jobs=12,
    seed=0,
):
    """
    Compute month-resolved permutation feature importance (PFI) for an LSTM
    mass-balance model using absolute ΔRMSE, evaluated separately for winter
    and annual periods.

    For each (feature, month) combination, the function permutes the input
    feature across samples, re-evaluates the model, and measures the increase
    in RMSE relative to the baseline. Computation is performed in parallel
    using joblib.

    Permutation strategy:
      - Monthly features are permuted within a single month slice
        (Xm[:, t, k]), preserving inter-month structure.
      - Static features are permuted across samples (Xs[:, k]) and evaluated
        per month task.

    Absolute importance is reported for:
      - winter RMSE (ΔRMSE_winter),
      - annual RMSE (ΔRMSE_annual),
      - a sample-weighted global ΔRMSE combining winter and annual samples.

    Parameters
    ----------
    model : torch.nn.Module
        Trained LSTM mass-balance model.
    device : torch.device
        Device used for inference.
    ds_test : Dataset
        Test dataset providing tensors Xm, Xs, mv, keys, iw, ia, y_mean, and y_std.
    monthly_cols : list of str
        Names of monthly input features (channels of Xm).
    static_cols : list of str
        Names of static input features (columns of Xs).
    month_names : list of str
        Month labels corresponding to the time dimension of Xm/mv.
    n_repeats : int, optional
        Number of random permutations per (feature, month) task.
    n_jobs : int, optional
        Number of parallel workers used by joblib.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Long-format table with one row per (feature, month) containing:
          - 'feature' : feature name,
          - 'month' : month label,
          - 'mean_delta_winter', 'std_delta_winter' : absolute ΔRMSE (winter),
          - 'mean_delta_annual', 'std_delta_annual' : absolute ΔRMSE (annual),
          - 'baseline_winter', 'baseline_annual' : baseline RMSE values,
          - 'mean_delta_global' : sample-weighted global absolute ΔRMSE.

    """
    print(
        "\n▶️ Running monthly permutation feature importance (PFI – absolute ΔRMSE)..."
    )

    base_dl = torch.utils.data.DataLoader(ds_test, batch_size=128, shuffle=False)
    base_w, base_a = eval_rmse_pfi_monthly(model, device, base_dl, ds_test)

    print(f"[Baseline RMSE] winter={base_w:.3f} | annual={base_a:.3f}")

    tasks = []

    # Monthly
    for k in range(len(monthly_cols)):
        for t in range(len(month_names)):
            if ds_test.mv[:, t].sum() > 0:
                tasks.append(("monthly", k, t))

    # Static per month
    for j in range(len(static_cols)):
        for t in range(len(month_names)):
            if ds_test.mv[:, t].sum() > 0:
                tasks.append(("static", j, t))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_pfi_month_worker)(
            task,
            model,
            device,
            mbm.data_processing.MBSequenceDataset._clone_for_permutation(ds_test),
            base_w,
            base_a,
            n_repeats,
            seed,
        )
        for task in tqdm(tasks, desc="Monthly permutation importance (absolute)")
    )

    rows = []
    for kind, k, t, mw, sw, ma, sa in results:
        fname = monthly_cols[k] if kind == "monthly" else static_cols[k]

        rows.append(
            dict(
                feature=fname,
                month=month_names[t],
                mean_delta_winter=mw,
                std_delta_winter=sw,
                mean_delta_annual=ma,
                std_delta_annual=sa,
                baseline_winter=base_w,
                baseline_annual=base_a,
            )
        )

    df = pd.DataFrame(rows)

    # ----- absolute sample-weighted global ΔRMSE -----
    Nw = int(ds_test.iw.sum())
    Na = int(ds_test.ia.sum())

    def compute_global_abs(row):
        parts, weights = [], []
        if np.isfinite(row.mean_delta_winter):
            parts.append(row.mean_delta_winter)
            weights.append(Nw)
        if np.isfinite(row.mean_delta_annual):
            parts.append(row.mean_delta_annual)
            weights.append(Na)
        if len(parts) == 0:
            return np.nan
        return np.average(parts, weights=weights)

    df["mean_delta_global"] = df.apply(compute_global_abs, axis=1)

    return df
