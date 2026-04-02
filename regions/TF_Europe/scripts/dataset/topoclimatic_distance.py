import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler


def compute_wasserstein_per_var(df_src, df_tgt, cols, id_col="ID"):
    result = {}
    for col in cols:
        # CURRENT: x = df_src.groupby(id_col)[col].mean().dropna().values
        # FIX:
        x = df_src[col].dropna().values
        y = df_tgt[col].dropna().values
        mu, sd = np.concatenate([x, y]).mean(), np.concatenate([x, y]).std()
        result[col] = wasserstein_distance((x - mu) / sd, (y - mu) / sd)
    return result


def energy_distance(
    X: np.ndarray,
    Y: np.ndarray,
    max_samples: int = 5000,
    seed: int = 0,
) -> float:
    """
    Energy distance between two multivariate samples.

    Returns sqrt(ed2), where:
        ed2 = 2 E||X-Y|| - E||X-X'|| - E||Y-Y'||
    """
    if X.size == 0 or Y.size == 0:
        return np.nan

    rng = np.random.default_rng(seed)

    def _subsample(A: np.ndarray) -> np.ndarray:
        if A.shape[0] <= max_samples:
            return A
        idx = rng.choice(A.shape[0], size=max_samples, replace=False)
        return A[idx]

    Xs = _subsample(X)
    Ys = _subsample(Y)

    def _mean_pairwise_dist(A: np.ndarray, B: np.ndarray) -> float:
        a2 = np.sum(A * A, axis=1, keepdims=True)
        b2 = np.sum(B * B, axis=1, keepdims=True).T
        d2 = a2 + b2 - 2.0 * (A @ B.T)
        d2 = np.maximum(d2, 0.0)
        d = np.sqrt(d2)
        return float(d.mean())

    EXY = _mean_pairwise_dist(Xs, Ys)
    EXX = _mean_pairwise_dist(Xs, Xs)
    EYY = _mean_pairwise_dist(Ys, Ys)

    ed2 = 2.0 * EXY - EXX - EYY
    return float(np.sqrt(max(ed2, 0.0)))


def mmd_squared_unbiased(
    X: np.ndarray,
    Y: np.ndarray,
    bandwidths: list[float] | None = None,
    max_samples: int = 5000,
    seed: int = 0,
) -> float:
    """
    Unbiased estimator of MMD^2 using a mixture of RBF kernels (MK-MMD).

    MMD^2(P,Q) = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]

    If bandwidths is None, uses the median heuristic on the pooled sample
    and adds two extra scales (0.5x and 2x) for robustness.
    """
    if X.size == 0 or Y.size == 0:
        return np.nan

    rng = np.random.default_rng(seed)

    def _subsample(A: np.ndarray) -> np.ndarray:
        if A.shape[0] <= max_samples:
            return A
        return A[rng.choice(A.shape[0], max_samples, replace=False)]

    X, Y = _subsample(X), _subsample(Y)
    m, n = X.shape[0], Y.shape[0]

    if bandwidths is None:
        # Median heuristic on pooled sample
        Z = np.vstack([X, Y])
        if Z.shape[0] > 2000:
            Z = Z[rng.choice(Z.shape[0], 2000, replace=False)]
        z2 = np.sum(Z * Z, axis=1, keepdims=True)
        D2 = np.maximum(z2 + z2.T - 2.0 * (Z @ Z.T), 0.0)
        np.fill_diagonal(D2, 0.0)
        median_d2 = np.median(D2[D2 > 0])
        sig = float(np.sqrt(0.5 * median_d2))
        bandwidths = [sig * 0.5, sig, sig * 2.0]

    def _rbf(A: np.ndarray, B: np.ndarray, sigma: float) -> np.ndarray:
        a2 = np.sum(A * A, axis=1, keepdims=True)
        b2 = np.sum(B * B, axis=1, keepdims=True).T
        return np.exp(-np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0) / (2.0 * sigma**2))

    mmd2_per_kernel = []
    for sigma in bandwidths:
        Kxx = _rbf(X, X, sigma)
        np.fill_diagonal(Kxx, 0.0)
        Kyy = _rbf(Y, Y, sigma)
        np.fill_diagonal(Kyy, 0.0)
        Kxy = _rbf(X, Y, sigma)
        mmd2_per_kernel.append(
            Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1)) - 2.0 * Kxy.mean()
        )

    return float(np.mean(mmd2_per_kernel))


def compute_domain_shift(
    df_src: pd.DataFrame,
    df_tgt: pd.DataFrame,
    monthly_cols: list[str],
    static_cols: list[str],
    glacier_col: str = "GLACIER",
    seed: int = 0,
    compute_marginals: bool = False,
    scaler_m: StandardScaler | None = None,
    scaler_s: StandardScaler | None = None,
) -> dict:
    """
    Assess input-space domain shift between source and target.

    MMD² values are clipped to zero — the unbiased estimator can return small
    negative values when distributions are very similar or sample sizes are
    small (especially for topo variables at glacier level). Negative MMD²
    is a statistical zero and clipping avoids propagating sign artifacts
    into joint distances and per-variable summaries.
    Energy distance is always non-negative by construction.
    """

    def _clip_mmd2(val: float) -> float:
        """Clip unbiased MMD² estimate to zero."""
        return float(max(val, 0.0))

    # --- climate: row-level ---
    Xm_src = df_src[monthly_cols].to_numpy(dtype=np.float64)
    Xm_tgt = df_tgt[monthly_cols].to_numpy(dtype=np.float64)

    # --- topo: glacier-level (deduplicated) ---
    Xs_src = df_src.groupby(glacier_col)[static_cols].first().to_numpy(dtype=np.float64)
    Xs_tgt = df_tgt.groupby(glacier_col)[static_cols].first().to_numpy(dtype=np.float64)

    # --- scalers (fit if not provided) ---
    if scaler_m is None:
        scaler_m = StandardScaler().fit(np.vstack([Xm_src, Xm_tgt]))
    if scaler_s is None:
        scaler_s = StandardScaler().fit(np.vstack([Xs_src, Xs_tgt]))

    Xm_src_z = scaler_m.transform(Xm_src)
    Xm_tgt_z = scaler_m.transform(Xm_tgt)
    Xs_src_z = scaler_s.transform(Xs_src)
    Xs_tgt_z = scaler_s.transform(Xs_tgt)

    # --- distances ---
    D_mmd2_climate = _clip_mmd2(mmd_squared_unbiased(Xm_src_z, Xm_tgt_z, seed=seed + 2))
    D_energy_climate = energy_distance(Xm_src_z, Xm_tgt_z, seed=seed + 3)
    D_mmd2_topo = _clip_mmd2(mmd_squared_unbiased(Xs_src_z, Xs_tgt_z, seed=seed + 4))
    D_energy_topo = energy_distance(Xs_src_z, Xs_tgt_z, seed=seed + 5)

    out = {
        "n_src_rows": len(Xm_src),
        "n_tgt_rows": len(Xm_tgt),
        "n_src_glaciers": len(Xs_src),
        "n_tgt_glaciers": len(Xs_tgt),
        # --- joint ---
        "D_mmd2_joint": 0.5 * D_mmd2_climate + 0.5 * D_mmd2_topo,
        "D_energy_joint": 0.5 * D_energy_climate + 0.5 * D_energy_topo,
        # --- climate only ---
        "D_mmd2_climate": D_mmd2_climate,
        "D_energy_climate": D_energy_climate,
        # --- topo only ---
        "D_mmd2_topo": D_mmd2_topo,
        "D_energy_topo": D_energy_topo,
    }

    # --- optional per-variable marginal distances ---
    if compute_marginals:
        # climate vars: row-level
        for j, col in enumerate(monthly_cols):
            out[f"D_mmd2_{col}"] = _clip_mmd2(
                mmd_squared_unbiased(
                    Xm_src_z[:, j : j + 1],
                    Xm_tgt_z[:, j : j + 1],
                    seed=seed + 10 + j,
                )
            )
            out[f"D_energy_{col}"] = float(
                energy_distance(
                    Xm_src_z[:, j : j + 1],
                    Xm_tgt_z[:, j : j + 1],
                    seed=seed + 100 + j,
                )
            )

        # topo vars: glacier-level
        offset = len(monthly_cols)
        for j, col in enumerate(static_cols):
            out[f"D_mmd2_{col}"] = _clip_mmd2(
                mmd_squared_unbiased(
                    Xs_src_z[:, j : j + 1],
                    Xs_tgt_z[:, j : j + 1],
                    seed=seed + 10 + offset + j,
                )
            )
            out[f"D_energy_{col}"] = float(
                energy_distance(
                    Xs_src_z[:, j : j + 1],
                    Xs_tgt_z[:, j : j + 1],
                    seed=seed + 100 + offset + j,
                )
            )

    return out


def split_pool_holdout(
    df_region: pd.DataFrame,
    monthly_cols: list[str],
    static_cols: list[str],
    glacier_col: str = "GLACIER",
    id_col: str = "ID",
    holdout_frac: float = 0.2,
    seed: int = 0,
) -> dict:

    def _glacier_features(df):
        meas_m = df.groupby(id_col)[monthly_cols].mean()
        meas_s = df.groupby(id_col)[[glacier_col] + static_cols].first()
        meas = meas_m.join(meas_s)
        grp = meas.groupby(glacier_col)
        X = np.hstack(
            [
                grp[monthly_cols].mean().to_numpy(dtype=np.float64),
                grp[static_cols].mean().to_numpy(dtype=np.float64),
            ]
        )
        names = grp[monthly_cols].mean().index.tolist()
        n_meas = grp[monthly_cols].count().iloc[:, 0].to_numpy(dtype=int)
        return X, names, n_meas

    X, glaciers, n_meas = _glacier_features(df_region)
    n_total_meas = n_meas.sum()
    target_holdout_meas = int(np.round(holdout_frac * n_total_meas))

    scaler = StandardScaler().fit(X)
    X_z = scaler.transform(X)

    print(f"  Total measurements   : {n_total_meas}")
    print(f"  Target holdout meas  : {target_holdout_meas} ({holdout_frac:.0%})")
    print(f"  Total glaciers       : {len(glaciers)}")

    # Return 0 for empty sets (not inf) — empty side is neutral, not broken
    def _mmd2(idxs):
        if len(idxs) < 2:
            return 0.0  # can't estimate MMD² with <2 samples, treat as 0
        return float(mmd_squared_unbiased(X_z[idxs], X_z, seed=seed))

    holdout_idxs = []
    pool_idxs = []
    holdout_meas_count = 0

    rng = np.random.default_rng(seed)
    order = list(range(len(glaciers)))
    rng.shuffle(order)

    for glacier_idx in order:
        glacier_meas = n_meas[glacier_idx]
        holdout_full = holdout_meas_count + glacier_meas > target_holdout_meas * 1.5

        if holdout_full:
            pool_idxs.append(glacier_idx)
            continue

        trial_holdout = holdout_idxs + [glacier_idx]
        trial_pool = pool_idxs + [glacier_idx]

        # Both sides evaluated independently vs Region_all — no cross-contamination
        combined_if_holdout = _mmd2(trial_holdout) + _mmd2(pool_idxs)
        combined_if_pool = _mmd2(holdout_idxs) + _mmd2(trial_pool)

        if combined_if_holdout <= combined_if_pool:
            holdout_idxs.append(glacier_idx)
            holdout_meas_count += glacier_meas
        else:
            pool_idxs.append(glacier_idx)

    mmd2_holdout = _mmd2(holdout_idxs)
    mmd2_pool = _mmd2(pool_idxs)
    actual_frac = holdout_meas_count / n_total_meas

    print(
        f"\n  Holdout : {len(holdout_idxs)} glaciers, "
        f"{holdout_meas_count} measurements ({actual_frac:.1%})"
    )
    print(
        f"  Pool    : {len(pool_idxs)} glaciers, "
        f"{n_total_meas - holdout_meas_count} measurements ({1-actual_frac:.1%})"
    )
    print(f"  MMD²(holdout, Region_all) = {mmd2_holdout:.4f}")
    print(f"  MMD²(pool,    Region_all) = {mmd2_pool:.4f}")

    return {
        "holdout_glaciers": [glaciers[i] for i in holdout_idxs],
        "pool_glaciers": [glaciers[i] for i in pool_idxs],
        "n_meas_holdout": int(holdout_meas_count),
        "n_meas_pool": int(n_total_meas - holdout_meas_count),
        "actual_holdout_frac": float(actual_frac),
        "mmd2_holdout_vs_region": float(mmd2_holdout),
        "mmd2_pool_vs_region": float(mmd2_pool),
    }
