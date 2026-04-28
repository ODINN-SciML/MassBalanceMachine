import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler
import torch
from geomloss import SamplesLoss
from tqdm import tqdm


def _estimate_blur(
    X: np.ndarray,
    Y: np.ndarray,
    blur_quantile_multiplier: float = 0.1,
    max_points: int = 4000,
    seed: int = 0,
) -> float:
    """Estimate blur from median pairwise squared distance on pooled sample."""
    rng = np.random.default_rng(seed)
    Z = np.vstack([X, Y]).astype(np.float32)
    if len(Z) > max_points:
        Z = Z[rng.choice(len(Z), size=max_points, replace=False)]

    n = len(Z)
    n_pairs = min(20000, n * (n - 1) // 2)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]

    if len(i) == 0:
        return 0.5

    sq_dists = np.sum((Z[i] - Z[j]) ** 2, axis=1)
    median_sq_dist = max(float(np.median(sq_dists)), 1e-8)
    return max(float(np.sqrt(blur_quantile_multiplier * median_sq_dist)), 1e-4)


def _sinkhorn_distance(
    X: np.ndarray,
    Y: np.ndarray,
    blur: float = 0.5,
    max_samples: int = 5000,
    device: str = "cpu",
    seed: int = 0,
) -> float:
    """Sinkhorn divergence between two sets of samples."""
    rng = np.random.default_rng(seed)

    def _subsample(A):
        if len(A) <= max_samples:
            return A
        return A[rng.choice(len(A), size=max_samples, replace=False)]

    X = _subsample(X).astype(np.float32)
    Y = _subsample(Y).astype(np.float32)

    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=blur, scaling=0.9, debias=True)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)
    a = torch.ones(len(Xt), device=device) / len(Xt)
    b = torch.ones(len(Yt), device=device) / len(Yt)

    with torch.no_grad():
        return float(loss_fn(a, Xt, b, Yt).item())


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


# def compute_domain_shift(
#     df_src: pd.DataFrame,
#     df_tgt: pd.DataFrame,
#     monthly_cols: list[str],
#     static_cols: list[str],
#     elev_diff_col: str = "ELEVATION_DIFFERENCE",
#     id_col: str = "ID",
#     glacier_col: str = "GLACIER",
#     seed: int = 0,
#     scaler_m: StandardScaler | None = None,
#     scaler_s: StandardScaler | None = None,
#     blur_m: float | None = None,
#     blur_s: float | None = None,
#     blur_joint: float | None = None,
#     bandwidths_m: list[float] | None = None,
#     bandwidths_s: list[float] | None = None,
#     device: str = "cpu",
# ) -> dict:
#     """
#     Assess input-space domain shift between source and target.

#     Climate features are compared row-level (each monthly observation is a
#     genuine data point). Topographic features are compared at ID level
#     (.first() for static cols, .mean() for ELEVATION_DIFFERENCE).

#     Computes MMD², energy distance, and Sinkhorn divergence for climate,
#     topo, a 50/50 averaged joint, and a true joint (climate + topo stacked
#     into one feature space, single OT problem).

#     For cross-pair comparability, pass fixed global blur_m/blur_s/blur_joint
#     and bandwidths_m/bandwidths_s estimated from the global training
#     distribution (via estimate_global_bandwidths).
#     """

#     def _clip_mmd2(val: float) -> float:
#         return float(max(val, 0.0))

#     pure_static = [c for c in static_cols if c != elev_diff_col]

#     def _stake_topo(df: pd.DataFrame) -> np.ndarray:
#         parts = [df.groupby(id_col)[pure_static].first()]
#         if elev_diff_col in static_cols:
#             parts.append(df.groupby(id_col)[[elev_diff_col]].mean())
#         return pd.concat(parts, axis=1)[static_cols].to_numpy(dtype=np.float64)

#     # --- raw features ---
#     Xm_src = df_src[monthly_cols].to_numpy(dtype=np.float64)
#     Xm_tgt = df_tgt[monthly_cols].to_numpy(dtype=np.float64)
#     Xs_src = _stake_topo(df_src)
#     Xs_tgt = _stake_topo(df_tgt)

#     # --- scalers ---
#     if scaler_m is None:
#         scaler_m = StandardScaler().fit(np.vstack([Xm_src, Xm_tgt]))
#     if scaler_s is None:
#         scaler_s = StandardScaler().fit(np.vstack([Xs_src, Xs_tgt]))

#     Xm_src_z = scaler_m.transform(Xm_src)
#     Xm_tgt_z = scaler_m.transform(Xm_tgt)
#     Xs_src_z = scaler_s.transform(Xs_src)
#     Xs_tgt_z = scaler_s.transform(Xs_tgt)

#     # --- blur for Sinkhorn ---
#     blur_m_ = (
#         blur_m if blur_m is not None else _estimate_blur(Xm_src_z, Xm_tgt_z, seed=seed)
#     )
#     blur_s_ = (
#         blur_s
#         if blur_s is not None
#         else _estimate_blur(Xs_src_z, Xs_tgt_z, seed=seed + 1)
#     )

#     # --- distances ---
#     D_mmd2_climate = _clip_mmd2(
#         mmd_squared_unbiased(Xm_src_z, Xm_tgt_z, bandwidths=bandwidths_m, seed=seed + 2)
#     )
#     D_energy_climate = energy_distance(Xm_src_z, Xm_tgt_z, seed=seed + 3)
#     D_sinkhorn_climate = _sinkhorn_distance(
#         Xm_src_z, Xm_tgt_z, blur=blur_m_, device=device, seed=seed + 4
#     )

#     D_mmd2_topo = _clip_mmd2(
#         mmd_squared_unbiased(Xs_src_z, Xs_tgt_z, bandwidths=bandwidths_s, seed=seed + 5)
#     )
#     D_energy_topo = energy_distance(Xs_src_z, Xs_tgt_z, seed=seed + 6)
#     D_sinkhorn_topo = _sinkhorn_distance(
#         Xs_src_z, Xs_tgt_z, blur=blur_s_, device=device, seed=seed + 7
#     )

#     # --- true joint Sinkhorn: stack climate and topo per monthly row ---
#     # Topo is ID-level — broadcast to monthly rows via ID index
#     topo_only_idx = [j for j, c in enumerate(static_cols) if c not in set(monthly_cols)]
#     if topo_only_idx:
#         topo_src_per_row = scaler_s.transform(_stake_topo(df_src))[
#             pd.Categorical(df_src[id_col]).codes
#         ][:, topo_only_idx]
#         topo_tgt_per_row = scaler_s.transform(_stake_topo(df_tgt))[
#             pd.Categorical(df_tgt[id_col]).codes
#         ][:, topo_only_idx]
#         Xjoint_src = np.hstack([Xm_src_z, topo_src_per_row])
#         Xjoint_tgt = np.hstack([Xm_tgt_z, topo_tgt_per_row])
#     else:
#         Xjoint_src = Xm_src_z
#         Xjoint_tgt = Xm_tgt_z

#     blur_joint_ = blur_joint if blur_joint is not None else 0.5 * (blur_m_ + blur_s_)
#     D_sinkhorn_joint = _sinkhorn_distance(
#         Xjoint_src, Xjoint_tgt, blur=blur_joint_, device=device, seed=seed + 8
#     )

#     out = {
#         "n_src_rows": len(Xm_src),
#         "n_tgt_rows": len(Xm_tgt),
#         "n_src_glaciers": df_src[glacier_col].nunique(),
#         "n_tgt_glaciers": df_tgt[glacier_col].nunique(),
#         "n_src_ids": df_src[id_col].nunique(),
#         "n_tgt_ids": df_tgt[id_col].nunique(),
#         # --- averaged joint (50/50 mean of climate and topo) ---
#         "D_mmd2_joint": 0.5 * D_mmd2_climate + 0.5 * D_mmd2_topo,
#         "D_energy_joint": 0.5 * D_energy_climate + 0.5 * D_energy_topo,
#         "D_sinkhorn_joint": D_sinkhorn_joint,
#         # --- climate ---
#         "D_mmd2_climate": D_mmd2_climate,
#         "D_energy_climate": D_energy_climate,
#         "D_sinkhorn_climate": D_sinkhorn_climate,
#         # --- topo ---
#         "D_mmd2_topo": D_mmd2_topo,
#         "D_energy_topo": D_energy_topo,
#         "D_sinkhorn_topo": D_sinkhorn_topo,
#     }
#     return out


def compute_domain_shift(
    df_src: pd.DataFrame,
    df_tgt: pd.DataFrame,
    monthly_cols: list[str],
    static_cols: list[str],
    elev_diff_col: str = "ELEVATION_DIFFERENCE",
    id_col: str = "ID",
    glacier_col: str = "GLACIER",
    seed: int = 0,
    scaler_m: StandardScaler | None = None,
    scaler_s: StandardScaler | None = None,
    scaler_joint: StandardScaler | None = None,  # NEW: joint scaler for grid data
    blur_m: float | None = None,
    blur_s: float | None = None,
    blur_joint: float | None = None,
    bandwidths_m: list[float] | None = None,
    bandwidths_s: list[float] | None = None,
    device: str = "cpu",
) -> dict:
    def _clip_mmd2(val: float) -> float:
        return float(max(val, 0.0))

    pure_static = [c for c in static_cols if c != elev_diff_col]

    def _stake_topo(df: pd.DataFrame) -> np.ndarray:
        parts = [df.groupby(id_col)[pure_static].first()]
        if elev_diff_col in static_cols:
            parts.append(df.groupby(id_col)[[elev_diff_col]].mean())
        return pd.concat(parts, axis=1)[static_cols].to_numpy(dtype=np.float64)

    # --- raw features ---
    Xm_src = df_src[monthly_cols].to_numpy(dtype=np.float64)
    Xm_tgt = df_tgt[monthly_cols].to_numpy(dtype=np.float64)
    Xs_src = _stake_topo(df_src)
    Xs_tgt = _stake_topo(df_tgt)

    # --- scalers ---
    if scaler_m is None:
        scaler_m = StandardScaler().fit(np.vstack([Xm_src, Xm_tgt]))
    if scaler_s is None:
        scaler_s = StandardScaler().fit(np.vstack([Xs_src, Xs_tgt]))

    Xm_src_z = scaler_m.transform(Xm_src)
    Xm_tgt_z = scaler_m.transform(Xm_tgt)
    Xs_src_z = scaler_s.transform(Xs_src)
    Xs_tgt_z = scaler_s.transform(Xs_tgt)

    # --- blur ---
    blur_m_ = (
        blur_m if blur_m is not None else _estimate_blur(Xm_src_z, Xm_tgt_z, seed=seed)
    )
    blur_s_ = (
        blur_s
        if blur_s is not None
        else _estimate_blur(Xs_src_z, Xs_tgt_z, seed=seed + 1)
    )

    # --- climate distances ---
    D_mmd2_climate = _clip_mmd2(
        mmd_squared_unbiased(Xm_src_z, Xm_tgt_z, bandwidths=bandwidths_m, seed=seed + 2)
    )
    D_energy_climate = energy_distance(Xm_src_z, Xm_tgt_z, seed=seed + 3)
    D_sinkhorn_climate = _sinkhorn_distance(
        Xm_src_z, Xm_tgt_z, blur=blur_m_, device=device, seed=seed + 4
    )

    # --- topo distances ---
    D_mmd2_topo = _clip_mmd2(
        mmd_squared_unbiased(Xs_src_z, Xs_tgt_z, bandwidths=bandwidths_s, seed=seed + 5)
    )
    D_energy_topo = energy_distance(Xs_src_z, Xs_tgt_z, seed=seed + 6)
    D_sinkhorn_topo = _sinkhorn_distance(
        Xs_src_z, Xs_tgt_z, blur=blur_s_, device=device, seed=seed + 7
    )

    # --- true joint Sinkhorn ---
    blur_joint_ = blur_joint if blur_joint is not None else 0.5 * (blur_m_ + blur_s_)

    if scaler_joint is not None:
        # Grid data path: scale all cols directly, no ID-level aggregation
        all_cols = monthly_cols + static_cols
        Xjoint_src = scaler_joint.transform(df_src[all_cols].to_numpy(dtype=np.float64))
        Xjoint_tgt = scaler_joint.transform(df_tgt[all_cols].to_numpy(dtype=np.float64))
    else:
        # Stake data path: broadcast ID-level topo to monthly rows
        topo_only_idx = [
            j for j, c in enumerate(static_cols) if c not in set(monthly_cols)
        ]
        if topo_only_idx:
            topo_src_per_row = scaler_s.transform(_stake_topo(df_src))[
                pd.Categorical(df_src[id_col]).codes
            ][:, topo_only_idx]
            topo_tgt_per_row = scaler_s.transform(_stake_topo(df_tgt))[
                pd.Categorical(df_tgt[id_col]).codes
            ][:, topo_only_idx]
            Xjoint_src = np.hstack([Xm_src_z, topo_src_per_row])
            Xjoint_tgt = np.hstack([Xm_tgt_z, topo_tgt_per_row])
        else:
            Xjoint_src = Xm_src_z
            Xjoint_tgt = Xm_tgt_z

    D_sinkhorn_joint = _sinkhorn_distance(
        Xjoint_src, Xjoint_tgt, blur=blur_joint_, device=device, seed=seed + 8
    )

    return {
        "n_src_rows": len(Xm_src),
        "n_tgt_rows": len(Xm_tgt),
        "n_src_glaciers": df_src[glacier_col].nunique(),
        "n_tgt_glaciers": df_tgt[glacier_col].nunique(),
        "n_src_ids": df_src[id_col].nunique(),
        "n_tgt_ids": df_tgt[id_col].nunique(),
        "D_mmd2_joint": 0.5 * D_mmd2_climate + 0.5 * D_mmd2_topo,
        "D_energy_joint": 0.5 * D_energy_climate + 0.5 * D_energy_topo,
        "D_sinkhorn_joint": D_sinkhorn_joint,
        "D_mmd2_climate": D_mmd2_climate,
        "D_energy_climate": D_energy_climate,
        "D_sinkhorn_climate": D_sinkhorn_climate,
        "D_mmd2_topo": D_mmd2_topo,
        "D_energy_topo": D_energy_topo,
        "D_sinkhorn_topo": D_sinkhorn_topo,
    }


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


def compute_glacier_shift_vs_source(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    monthly_cols: list[str],
    static_cols: list[str],
    scaler_m,
    scaler_s,
    glacier_col: str = "GLACIER",
    min_ids: int = 3,
    max_train_samples: int = 5000,
    blur_quantile_multiplier: float = 0.1,
    device: str = "cpu",
    backend: str = "tensorized",  # "tensorized" avoids KeOps CUDA issues
    seed: int = 0,
) -> pd.DataFrame:
    pure_static = [c for c in static_cols if c != "ELEVATION_DIFFERENCE"]

    def _stake_topo(df):
        parts = [df.groupby("ID")[pure_static].first()]
        if "ELEVATION_DIFFERENCE" in static_cols:
            parts.append(df.groupby("ID")[["ELEVATION_DIFFERENCE"]].mean())
        return pd.concat(parts, axis=1)[static_cols].to_numpy(dtype=np.float64)

    # Scale once
    Xm_train_full = scaler_m.transform(
        df_train[monthly_cols].to_numpy(dtype=np.float64)
    )
    Xs_train_full = scaler_s.transform(_stake_topo(df_train))

    # Subsample train for blur estimation
    rng = np.random.default_rng(seed)

    def _subsample(X):
        if len(X) <= max_train_samples:
            return X
        return X[rng.choice(len(X), size=max_train_samples, replace=False)]

    Xm_train = _subsample(Xm_train_full)
    Xs_train = _subsample(Xs_train_full)

    # Estimate blur ONCE on the subsampled train distribution
    # This matches what SinkhornGroupSplit does in _prepare_data
    blur_m = _estimate_blur(
        Xm_train, Xm_train, blur_quantile_multiplier=blur_quantile_multiplier, seed=seed
    )
    blur_s = _estimate_blur(
        Xs_train,
        Xs_train,
        blur_quantile_multiplier=blur_quantile_multiplier,
        seed=seed + 1,
    )

    # Build loss functions once, reuse for all glaciers
    loss_m = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=blur_m,
        scaling=0.9,
        debias=True,
        backend="tensorized",
    )  # avoid KeOps
    loss_s = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=blur_s,
        scaling=0.9,
        debias=True,
        backend="tensorized",
    )

    Xm_train_t = torch.as_tensor(Xm_train, dtype=torch.float32, device=device)
    Xs_train_t = torch.as_tensor(Xs_train, dtype=torch.float32, device=device)

    records = []
    for glacier, df_gl in tqdm(df_test.groupby(glacier_col), desc="glaciers"):
        if df_gl["ID"].nunique() < min_ids:
            continue

        Xm_gl = torch.as_tensor(
            scaler_m.transform(df_gl[monthly_cols].to_numpy(dtype=np.float64)),
            dtype=torch.float32,
            device=device,
        )
        Xs_gl = torch.as_tensor(
            scaler_s.transform(_stake_topo(df_gl)), dtype=torch.float32, device=device
        )

        # Uniform weights
        a_m = torch.ones(len(Xm_train_t), device=device) / len(Xm_train_t)
        b_m = torch.ones(len(Xm_gl), device=device) / len(Xm_gl)
        a_s = torch.ones(len(Xs_train_t), device=device) / len(Xs_train_t)
        b_s = torch.ones(len(Xs_gl), device=device) / len(Xs_gl)

        with torch.no_grad():
            D_climate = float(loss_m(a_m, Xm_train_t, b_m, Xm_gl).item())
            D_topo = float(loss_s(a_s, Xs_train_t, b_s, Xs_gl).item())

        records.append(
            {
                glacier_col: glacier,
                "n_ids": df_gl["ID"].nunique(),
                "n_rows": len(df_gl),
                "D_sinkhorn_joint": 0.5 * D_climate + 0.5 * D_topo,
                "D_sinkhorn_climate": D_climate,
                "D_sinkhorn_topo": D_topo,
            }
        )

    return pd.DataFrame(records).set_index(glacier_col)


# def estimate_global_bandwidths(
#     res_xreg_by_source: dict,
#     monthly_cols: list[str],
#     static_cols: list[str],
#     scaler_m: StandardScaler,
#     scaler_s: StandardScaler,
#     elev_diff_col: str = "ELEVATION_DIFFERENCE",
#     id_col: str = "ID",
#     blur_quantile_multiplier: float = 0.1,
#     seed: int = 0,
# ) -> tuple[float, float, float]:
#     """
#     Estimate fixed blur/bandwidth from the global training distribution.
#     Pools all df_train rows across sources, scales them, then estimates
#     blur from the within-distribution pairwise distances.
#     These fixed values make Sinkhorn and MMD² comparable across all pairs.

#     Returns
#     -------
#     blur_m : float
#         Blur for climate feature space.
#     blur_s : float
#         Blur for topographic feature space.
#     blur_joint : float
#         Blur for joint (climate + topo stacked) feature space.
#     """
#     pure_static = [c for c in static_cols if c != elev_diff_col]

#     def _stake_topo(df):
#         parts = [df.groupby(id_col)[pure_static].first()]
#         if elev_diff_col in static_cols:
#             parts.append(df.groupby(id_col)[[elev_diff_col]].mean())
#         return pd.concat(parts, axis=1)[static_cols].to_numpy(dtype=np.float64)

#     df_train_all = pd.concat(
#         [res_xreg["df_train"] for res_xreg in res_xreg_by_source.values()],
#         ignore_index=True,
#     ).drop_duplicates(subset=[id_col])

#     df_train_full = pd.concat(
#         [res_xreg["df_train"] for res_xreg in res_xreg_by_source.values()],
#         ignore_index=True,
#     )

#     Xm = scaler_m.transform(df_train_all[monthly_cols].to_numpy(dtype=np.float64))
#     Xs = scaler_s.transform(_stake_topo(df_train_full))

#     blur_m = _estimate_blur(
#         Xm, Xm, blur_quantile_multiplier=blur_quantile_multiplier, seed=seed
#     )
#     blur_s = _estimate_blur(
#         Xs, Xs, blur_quantile_multiplier=blur_quantile_multiplier, seed=seed + 1
#     )

#     # Joint: broadcast topo to monthly rows then stack
#     Xs_full = scaler_s.transform(_stake_topo(df_train_full))
#     topo_per_row = Xs_full[pd.Categorical(df_train_full[id_col]).codes]
#     Xjoint = np.hstack([Xm, topo_per_row[: len(Xm)]])

#     blur_joint = _estimate_blur(
#         Xjoint,
#         Xjoint,
#         blur_quantile_multiplier=blur_quantile_multiplier,
#         seed=seed + 2,
#     )

#     return blur_m, blur_s, blur_joint

# def build_global_scalers_multi_source(
#     res_xreg_by_source: dict,
#     monthly_cols: list[str],
#     static_cols: list[str],
#     elev_diff_col: str = "ELEVATION_DIFFERENCE",
#     id_col: str = "ID",
# ) -> tuple[StandardScaler, StandardScaler]:
#     """
#     Build global StandardScalers fitted on the full dataset.

#     Pools df_train and df_test across all sources, deduplicates by ID,
#     then fits:
#       - scaler_m: on monthly climate rows, deduplicated by ID so each
#         stake-year contributes exactly once
#       - scaler_s: on ID-level topographic aggregates — static cols via
#         .first() per ID, ELEVATION_DIFFERENCE via .mean() per ID
#     """
#     pure_static = [c for c in static_cols if c != elev_diff_col]

#     def _stake_topo(df: pd.DataFrame) -> np.ndarray:
#         parts = [df.groupby(id_col)[pure_static].first()]
#         if elev_diff_col in static_cols:
#             parts.append(df.groupby(id_col)[[elev_diff_col]].mean())
#         return pd.concat(parts, axis=1)[static_cols].to_numpy(dtype=np.float64)

#     df_all = pd.concat(
#         [
#             res_xreg[split]
#             for res_xreg in res_xreg_by_source.values()
#             for split in ["df_train", "df_test"]
#         ],
#         ignore_index=True,
#     )

#     # Climate: deduplicate by ID so each stake-year contributes once
#     df_dedup = df_all.drop_duplicates(subset=[id_col])
#     scaler_m = StandardScaler().fit(df_dedup[monthly_cols].to_numpy(dtype=np.float64))

#     # Topo: groupby handles deduplication correctly for mixed static/monthly cols
#     scaler_s = StandardScaler().fit(_stake_topo(df_all))

#     return scaler_m, scaler_s


def build_global_scalers_multi_source_simple(
    res_xreg_by_source: dict,
    monthly_cols: list[str],
    static_cols: list[str],
) -> tuple[StandardScaler, StandardScaler, StandardScaler]:
    """
    Simple version of build_global_scalers_multi_source.
    Pools df_train and df_test across all sources and fits scalers directly
    without ID deduplication or topo aggregation.

    Fits:
      - scaler_m: on monthly climate columns
      - scaler_s: on static/topographic columns
      - scaler_joint: on all columns combined
    """
    df_all = pd.concat(
        [
            res_xreg[split]
            for res_xreg in res_xreg_by_source.values()
            for split in ["df_train", "df_test"]
        ],
        ignore_index=True,
    )

    scaler_m = StandardScaler().fit(df_all[monthly_cols].to_numpy(dtype=np.float64))
    scaler_s = StandardScaler().fit(df_all[static_cols].to_numpy(dtype=np.float64))
    scaler_joint = StandardScaler().fit(
        df_all[monthly_cols + static_cols].to_numpy(dtype=np.float64)
    )

    return scaler_m, scaler_s, scaler_joint


def estimate_global_bandwidths_simple(
    res_xreg_by_source: dict,
    monthly_cols: list[str],
    static_cols: list[str],
    scaler_m: StandardScaler,
    scaler_s: StandardScaler,
    blur_quantile_multiplier: float = 0.1,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    Simple version of estimate_global_bandwidths.
    Pools df_train across all sources and estimates blurs directly
    without ID deduplication or topo aggregation.
    """
    df_all = pd.concat(
        [res_xreg["df_train"] for res_xreg in res_xreg_by_source.values()],
        ignore_index=True,
    )

    Xm = scaler_m.transform(df_all[monthly_cols].to_numpy(dtype=np.float64))
    Xs = scaler_s.transform(df_all[static_cols].to_numpy(dtype=np.float64))
    Xjoint = np.hstack([Xm, Xs])

    blur_m = _estimate_blur(
        Xm, Xm, blur_quantile_multiplier=blur_quantile_multiplier, seed=seed
    )
    blur_s = _estimate_blur(
        Xs, Xs, blur_quantile_multiplier=blur_quantile_multiplier, seed=seed + 1
    )
    blur_joint = _estimate_blur(
        Xjoint, Xjoint, blur_quantile_multiplier=blur_quantile_multiplier, seed=seed + 2
    )

    return blur_m, blur_s, blur_joint


def build_global_scalers_from_dfs(
    dfs: list[pd.DataFrame] | dict[str, pd.DataFrame],
    monthly_cols: list[str],
    static_cols: list[str],
) -> tuple[StandardScaler, StandardScaler, StandardScaler]:
    """
    Build global StandardScalers fitted on a collection of dataframes.
    Accepts either a list or a dict of dataframes (values only are used).

    Fits:
      - scaler_m: on monthly climate columns
      - scaler_s: on static/topographic columns
      - scaler_all: on all columns combined
    """
    if isinstance(dfs, dict):
        dfs = list(dfs.values())

    df_all = pd.concat(dfs, ignore_index=True)

    scaler_m = StandardScaler().fit(df_all[monthly_cols].to_numpy(dtype=np.float64))
    scaler_s = StandardScaler().fit(df_all[static_cols].to_numpy(dtype=np.float64))
    scaler_all = StandardScaler().fit(
        df_all[monthly_cols + static_cols].to_numpy(dtype=np.float64)
    )

    return scaler_m, scaler_s, scaler_all


def estimate_global_bandwidths_from_dfs(
    dfs: list[pd.DataFrame] | dict[str, pd.DataFrame],
    monthly_cols: list[str],
    static_cols: list[str],
    scaler_m: StandardScaler,
    scaler_s: StandardScaler,
    blur_quantile_multiplier: float = 0.1,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    Estimate fixed blur/bandwidth from the pooled distribution of all dfs.
    Simple version for grid data — no ID deduplication or topo aggregation.
    """
    if isinstance(dfs, dict):
        dfs = list(dfs.values())

    df_all = pd.concat(dfs, ignore_index=True)

    Xm = scaler_m.transform(df_all[monthly_cols].to_numpy(dtype=np.float64))
    Xs = scaler_s.transform(df_all[static_cols].to_numpy(dtype=np.float64))
    Xjoint = np.hstack([Xm, Xs])

    blur_m = _estimate_blur(
        Xm, Xm, blur_quantile_multiplier=blur_quantile_multiplier, seed=seed
    )
    blur_s = _estimate_blur(
        Xs, Xs, blur_quantile_multiplier=blur_quantile_multiplier, seed=seed + 1
    )
    blur_joint = _estimate_blur(
        Xjoint, Xjoint, blur_quantile_multiplier=blur_quantile_multiplier, seed=seed + 2
    )

    return blur_m, blur_s, blur_joint
