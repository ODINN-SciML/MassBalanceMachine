import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


@torch.no_grad()
def eval_rmse_pfi_full(model, device, dl, ds):
    """
    Evaluate winter and annual RMSE for a trained LSTM model in a
    period-aware and denormalized manner.

    This function mirrors the behavior of the standard evaluation routine
    used during training:
      - Samples are routed to winter or annual based on ``ds.keys``.
      - Predictions and targets are denormalized using ``ds.y_mean`` and
        ``ds.y_std``.
      - RMSE is computed separately for winter and annual samples.

    Parameters
    ----------
    model : torch.nn.Module
        Trained LSTM model returning (y_hat, y_winter, y_annual).
    device : torch.device
        Device on which evaluation is performed.
    dl : torch.utils.data.DataLoader
        DataLoader iterating over ``ds`` with ``shuffle=False``.
    ds : Dataset
        Dataset providing attributes ``keys``, ``y_mean``, and ``y_std``.

    Returns
    -------
    rmse_winter : float
        Root mean squared error for winter samples.
    rmse_annual : float
        Root mean squared error for annual samples.
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

        # denormalize (matches evaluate_with_preds)
        y_true = y * y_std + y_mean
        y_w = y_w * y_std + y_mean
        y_a = y_a * y_std + y_mean

        for j in range(bs):
            *_, per = batch_keys[j]
            if per == "winter":
                y_true_w.append(float(y_true[j].cpu()))
                y_pred_w.append(float(y_w[j].cpu()))
            else:  # annual
                y_true_a.append(float(y_true[j].cpu()))
                y_pred_a.append(float(y_a[j].cpu()))

    def rmse(t, p):
        t = np.asarray(t, dtype=float)
        p = np.asarray(p, dtype=float)
        if len(t) == 0:
            return float("nan")
        return float(np.sqrt(np.mean((p - t) ** 2)))

    return rmse(y_true_w, y_pred_w), rmse(y_true_a, y_pred_a)


def PFI_LSTM_full(
    model,
    device,
    ds_test,
    monthly_cols,
    static_cols,
    n_repeats=5,
    seed=0,
    batch_size=128,
):
    """
    Compute permutation feature importance (PFI) for an LSTM mass-balance model
    with explicit separation of winter and annual performance.

    Feature importance is assessed by permuting input features and measuring
    the resulting increase in RMSE relative to a baseline evaluation.
    The method is period-aware and fully denormalized.

    Permutation strategy:
      - Monthly features: permuted jointly across all months
        (``Xm[:, :, k]``) to preserve seasonal structure.
      - Static features: permuted across samples (``Xs[:, j]``).

    For each feature, the function reports:
      - Absolute RMSE increase (ΔRMSE) for winter and annual,
      - Relative RMSE increase (ΔRMSE / baseline),
      - A sample-weighted global relative importance combining winter and
        annual contributions.

    Parameters
    ----------
    model : torch.nn.Module
        Trained LSTM mass-balance model.
    device : torch.device
        Device on which inference is performed.
    ds_test : Dataset
        Test dataset exposing tensors ``Xm`` (monthly inputs),
        ``Xs`` (static inputs), and attributes ``iw`` / ``ia`` indicating
        winter and annual samples.
    monthly_cols : list of str
        Names of monthly input features corresponding to ``Xm`` channels.
    static_cols : list of str
        Names of static input features corresponding to ``Xs`` columns.
    n_repeats : int, optional
        Number of random permutations per feature.
    seed : int, optional
        Random seed for reproducibility.
    batch_size : int, optional
        Batch size used during evaluation.

    Returns
    -------
    pandas.DataFrame
        Table of permutation feature importance scores with columns including:
          - feature name and type (monthly/static),
          - baseline winter and annual RMSE,
          - mean and std of absolute ΔRMSE (winter & annual),
          - mean and std of relative ΔRMSE,
          - sample-weighted global relative importance.

    """
    rng = np.random.default_rng(seed)

    print("\n▶️ Running aggregated permutation feature importance (PFI)...")

    base_dl = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    base_w, base_a = eval_rmse_pfi_full(model, device, base_dl, ds_test)

    # counts for sample-weighted global
    Nw = int(ds_test.iw.sum())
    Na = int(ds_test.ia.sum())

    print(
        f"[Baseline RMSE] winter={base_w:.3f} | annual={base_a:.3f} (Nw={Nw}, Na={Na})"
    )

    Xm0 = ds_test.Xm.clone()
    Xs0 = ds_test.Xs.clone()

    rows = []
    total_steps = (len(monthly_cols) + len(static_cols)) * n_repeats
    pbar = tqdm(total=total_steps, desc="Aggregated permutation importance")

    def _record(fname, ftype, dw, da):
        dw = np.asarray(dw, dtype=float)
        da = np.asarray(da, dtype=float)

        # relative deltas
        dw_rel = dw / base_w
        da_rel = da / base_a

        # sample-weighted global (relative)
        global_rel = (Nw * dw_rel + Na * da_rel) / max(Nw + Na, 1)

        rows.append(
            dict(
                feature=fname,
                type=ftype,
                baseline_winter=float(base_w),
                baseline_annual=float(base_a),
                mean_delta_winter=float(dw.mean()),
                std_delta_winter=float(dw.std(ddof=0)),
                mean_delta_annual=float(da.mean()),
                std_delta_annual=float(da.std(ddof=0)),
                mean_delta_winter_rel=float(dw_rel.mean()),
                std_delta_winter_rel=float(dw_rel.std(ddof=0)),
                mean_delta_annual_rel=float(da_rel.mean()),
                std_delta_annual_rel=float(da_rel.std(ddof=0)),
                mean_delta_global_rel=float(global_rel.mean()),
                std_delta_global_rel=float(global_rel.std(ddof=0)),
            )
        )

    # ---------- Monthly features ----------
    for k, fname in enumerate(monthly_cols):
        dw, da = [], []

        for _ in range(n_repeats):
            perm = rng.permutation(len(ds_test))
            ds_test.Xm[:, :, k] = Xm0[perm, :, k]

            dl = torch.utils.data.DataLoader(
                ds_test, batch_size=batch_size, shuffle=False
            )
            w, a = eval_rmse_pfi_full(model, device, dl, ds_test)

            dw.append(w - base_w)
            da.append(a - base_a)
            pbar.update(1)

        _record(fname, "monthly", dw, da)
        ds_test.Xm[:] = Xm0  # restore

    # ---------- Static features ----------
    for j, fname in enumerate(static_cols):
        dw, da = [], []

        for _ in range(n_repeats):
            perm = rng.permutation(len(ds_test))
            ds_test.Xs[:, j] = Xs0[perm, j]

            dl = torch.utils.data.DataLoader(
                ds_test, batch_size=batch_size, shuffle=False
            )
            w, a = eval_rmse_pfi_full(model, device, dl, ds_test)

            dw.append(w - base_w)
            da.append(a - base_a)
            pbar.update(1)

        _record(fname, "static", dw, da)
        ds_test.Xs[:] = Xs0  # restore

    pbar.close()

    out = pd.DataFrame(rows)

    # Useful default sorting: by global_rel, then annual_rel, then winter_rel
    out = out.sort_values(
        ["mean_delta_global_rel", "mean_delta_annual_rel", "mean_delta_winter_rel"],
        ascending=False,
    ).reset_index(drop=True)

    return out
