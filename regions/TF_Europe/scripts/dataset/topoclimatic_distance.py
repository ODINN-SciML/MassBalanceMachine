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


def compute_domain_shift(
    df_src: pd.DataFrame,
    df_tgt: pd.DataFrame,
    monthly_cols: list[str],
    static_cols: list[str],
    geo_cols: list[str] | None = None,
    elev_diff_col: str = "ELEVATION_DIFFERENCE",
    id_col: str = "ID",
    glacier_col: str = "GLACIER",
    seed: int = 0,
    scaler_m: StandardScaler | None = None,
    scaler_s: StandardScaler | None = None,
    scaler_geo: StandardScaler | None = None,
    scaler_joint: StandardScaler | None = None,  # joint scaler for grid data
    blur_m: float | None = None,
    blur_s: float | None = None,
    blur_geo: float | None = None,
    blur_joint: float | None = None,
    bandwidths_m: list[float] | None = None,
    bandwidths_s: list[float] | None = None,
    bandwidths_geo: list[float] | None = None,
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

    def _stake_geo(df: pd.DataFrame) -> np.ndarray:
        # geo coords are per-stake, like topo — one row per ID
        return df.groupby(id_col)[geo_cols].first().to_numpy(dtype=np.float64)

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

    # --- geo distances (purely spatial, e.g. unit-sphere Cartesian coords) ---
    if geo_cols is not None:
        Xg_src = _stake_geo(df_src)
        Xg_tgt = _stake_geo(df_tgt)

        if scaler_geo is None:
            scaler_geo = StandardScaler().fit(np.vstack([Xg_src, Xg_tgt]))

        Xg_src_z = scaler_geo.transform(Xg_src)
        Xg_tgt_z = scaler_geo.transform(Xg_tgt)

        blur_geo_ = (
            blur_geo
            if blur_geo is not None
            else _estimate_blur(Xg_src_z, Xg_tgt_z, seed=seed + 9)
        )

        D_mmd2_geo = _clip_mmd2(
            mmd_squared_unbiased(
                Xg_src_z, Xg_tgt_z, bandwidths=bandwidths_geo, seed=seed + 10
            )
        )
        D_energy_geo = energy_distance(Xg_src_z, Xg_tgt_z, seed=seed + 11)
        D_sinkhorn_geo = _sinkhorn_distance(
            Xg_src_z, Xg_tgt_z, blur=blur_geo_, device=device, seed=seed + 12
        )
    else:
        D_mmd2_geo = D_energy_geo = D_sinkhorn_geo = float("nan")

    # --- true joint Sinkhorn (climate + topo only — geo kept separate) ---
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
        "D_mmd2_geo": D_mmd2_geo,
        "D_energy_geo": D_energy_geo,
        "D_sinkhorn_geo": D_sinkhorn_geo,
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


def split_pool_holdout_sinkhorn(
    df_region: pd.DataFrame,
    monthly_cols: list[str],
    static_cols: list[str],
    scaler_m,
    scaler_s,
    blur_joint: float,
    glacier_col: str = "GLACIER",
    id_col: str = "ID",
    elev_diff_col: str = "ELEVATION_DIFFERENCE",
    pool_frac: float = 0.2,
    seed: int = 0,
    n_restarts: int = 50,
    frac_tol: float = 0.05,
    min_pool_glaciers: int = 3,
) -> dict:
    """
    Split a target region into a fine-tuning pool and a holdout set.

    Distances are computed at measurement-row level using the true joint
    (climate + topo stacked) Sinkhorn divergence, consistent with
    D_sinkhorn_joint in compute_domain_shift.

    The split is optimized so that:
      (1) Sinkhorn(pool → holdout) is small  — pool covers the holdout distribution
      (2) |Sinkhorn(pool → all) - Sinkhorn(holdout → all)| is small — balanced shift

    Parameters
    ----------
    df_region : pd.DataFrame
        All rows for the target region.
    monthly_cols : list[str]
        Monthly feature column names.
    static_cols : list[str]
        Static feature column names.
    scaler_m :
        Global monthly scaler fitted via build_global_scalers_multi_source.
    scaler_s :
        Global static scaler fitted via build_global_scalers_multi_source.
    blur_joint : float
        Sinkhorn blur for the joint space — pass blur_joint from 1.7.
    glacier_col : str
        Column identifying glaciers.
    id_col : str
        Column identifying individual measurements.
    elev_diff_col : str
        Name of the elevation difference column (averaged per ID, not .first()).
    pool_frac : float
        Target fraction of measurements to use as fine-tuning pool.
    seed : int
        Base random seed.
    n_restarts : int
        Number of random orderings to try; best split by score is kept.
    frac_tol : float
        Splits where |achieved_frac - pool_frac| > frac_tol are skipped.
    min_pool_glaciers : int
        Minimum number of glaciers required in the pool.

    Returns
    -------
    dict with keys:
        pool_glaciers, holdout_glaciers,
        n_meas_pool, n_meas_holdout,
        actual_pool_frac,
        sinkhorn_pool_vs_holdout,
        sinkhorn_pool_vs_region,
        sinkhorn_holdout_vs_region,
        blur_joint, best_seed, best_score
    """
    # ------------------------------------------------------------------
    # 1. Glacier inventory — names and measurement counts
    # ------------------------------------------------------------------
    glaciers = df_region[glacier_col].unique().tolist()
    n_glaciers = len(glaciers)
    glacier_to_idx = {g: i for i, g in enumerate(glaciers)}

    n_meas = np.array(
        [(df_region[glacier_col] == g).sum() for g in glaciers], dtype=int
    )

    n_total_meas = int(n_meas.sum())
    target_pool_meas = int(round(pool_frac * n_total_meas))

    print(f"  Total measurements : {n_total_meas}")
    print(f"  Target pool meas   : {target_pool_meas} ({pool_frac:.0%})")
    print(f"  Total glaciers     : {n_glaciers}")
    print(f"  blur_joint         : {blur_joint:.4f}")

    # ------------------------------------------------------------------
    # 2. Precompute row-level joint features for the whole region
    #    (consistent with compute_domain_shift's D_sinkhorn_joint)
    # ------------------------------------------------------------------
    pure_static = [c for c in static_cols if c != elev_diff_col]

    def _stake_topo(df):
        parts = [df.groupby(id_col)[pure_static].first()]
        if elev_diff_col in static_cols:
            parts.append(df.groupby(id_col)[[elev_diff_col]].mean())
        return pd.concat(parts, axis=1)[static_cols].to_numpy(dtype=np.float64)

    Xm_all = scaler_m.transform(df_region[monthly_cols].to_numpy(dtype=np.float64))
    Xs_id = scaler_s.transform(_stake_topo(df_region))
    id_codes = pd.Categorical(df_region[id_col]).codes
    topo_per_row = Xs_id[id_codes]
    X_joint_all = np.hstack([Xm_all, topo_per_row]).astype(np.float32)

    # row-to-glacier mapping for fast subsetting
    glacier_codes = np.array(
        [glacier_to_idx[g] for g in df_region[glacier_col]], dtype=int
    )

    loss_fn = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=blur_joint,
        scaling=0.9,
        debias=True,
        backend="tensorized",
    )

    def _sinkhorn(mask_a, mask_b, max_samples=2000):
        Xa = X_joint_all[mask_a]
        Xb = X_joint_all[mask_b]
        if len(Xa) < 2 or len(Xb) < 2:
            return 0.0
        if len(Xa) > max_samples:
            Xa = Xa[np.random.choice(len(Xa), max_samples, replace=False)]
        if len(Xb) > max_samples:
            Xb = Xb[np.random.choice(len(Xb), max_samples, replace=False)]
        ta = torch.as_tensor(Xa, dtype=torch.float32)
        tb = torch.as_tensor(Xb, dtype=torch.float32)
        wa = torch.ones(len(ta)) / len(ta)
        wb = torch.ones(len(tb)) / len(tb)
        with torch.no_grad():
            return float(loss_fn(wa, ta, wb, tb).item())

    def _mask_for(glacier_idxs):
        return np.isin(glacier_codes, list(glacier_idxs))

    all_mask = np.ones(len(X_joint_all), dtype=bool)

    # ------------------------------------------------------------------
    # 3. Greedy split with n_restarts random orderings, keep best score
    # ------------------------------------------------------------------
    best_score = np.inf
    best_result = None

    for restart in range(n_restarts):
        rng = np.random.default_rng(seed + restart)
        order = np.arange(n_glaciers)
        rng.shuffle(order)

        pool_idxs = set()
        holdout_idxs = set()
        pool_meas_count = 0

        for glacier_idx in order:
            gl_meas = int(n_meas[glacier_idx])
            pool_full = pool_meas_count + gl_meas > target_pool_meas * (1 + frac_tol)

            if pool_full:
                holdout_idxs.add(glacier_idx)
                continue

            trial_pool = pool_idxs | {glacier_idx}
            trial_holdout = holdout_idxs | {glacier_idx}

            holdout_mask = _mask_for(holdout_idxs) if holdout_idxs else all_mask
            pool_mask = _mask_for(pool_idxs) if pool_idxs else all_mask

            score_if_pool = _sinkhorn(_mask_for(trial_pool), holdout_mask) + abs(
                _sinkhorn(_mask_for(trial_pool), all_mask)
                - _sinkhorn(holdout_mask, all_mask)
            )
            score_if_holdout = _sinkhorn(pool_mask, _mask_for(trial_holdout)) + abs(
                _sinkhorn(pool_mask, all_mask)
                - _sinkhorn(_mask_for(trial_holdout), all_mask)
            )

            if score_if_pool <= score_if_holdout:
                pool_idxs.add(glacier_idx)
                pool_meas_count += gl_meas
            else:
                holdout_idxs.add(glacier_idx)

        achieved_frac = pool_meas_count / n_total_meas
        if abs(achieved_frac - pool_frac) > frac_tol:
            continue
        if len(pool_idxs) < min_pool_glaciers:
            continue

        pool_mask = _mask_for(pool_idxs)
        holdout_mask = _mask_for(holdout_idxs)

        sk_pool_holdout = _sinkhorn(pool_mask, holdout_mask)
        sk_pool_region = _sinkhorn(pool_mask, all_mask)
        sk_holdout_region = _sinkhorn(holdout_mask, all_mask)

        score = sk_pool_holdout + abs(sk_pool_region - sk_holdout_region)

        print(
            f"  restart {restart:3d} | frac={achieved_frac:.2f} | "
            f"sk(pool↔holdout)={sk_pool_holdout:.4f} | "
            f"balance={abs(sk_pool_region - sk_holdout_region):.4f} | "
            f"score={score:.4f}"
        )

        if score < best_score:
            best_score = score
            best_result = {
                "pool_idxs": pool_idxs,
                "holdout_idxs": holdout_idxs,
                "pool_meas_count": pool_meas_count,
                "achieved_frac": achieved_frac,
                "sk_pool_holdout": sk_pool_holdout,
                "sk_pool_region": sk_pool_region,
                "sk_holdout_region": sk_holdout_region,
                "best_seed": seed + restart,
            }

    if best_result is None:
        raise RuntimeError(
            f"No valid split found within frac_tol={frac_tol} and "
            f"min_pool_glaciers={min_pool_glaciers} after {n_restarts} restarts. "
            f"Try increasing n_restarts or frac_tol."
        )

    r = best_result
    pool_glaciers = [glaciers[i] for i in r["pool_idxs"]]
    holdout_glaciers = [glaciers[i] for i in r["holdout_idxs"]]

    print(f"\n  Best split (seed={r['best_seed']}):")
    print(
        f"  Pool    : {len(pool_glaciers)} glaciers, {r['pool_meas_count']} meas ({r['achieved_frac']:.1%})"
    )
    print(
        f"  Holdout : {len(holdout_glaciers)} glaciers, {n_total_meas - r['pool_meas_count']} meas ({1 - r['achieved_frac']:.1%})"
    )
    print(f"  Sinkhorn(pool → holdout)  : {r['sk_pool_holdout']:.4f}")
    print(f"  Sinkhorn(pool → region)   : {r['sk_pool_region']:.4f}")
    print(f"  Sinkhorn(holdout → region): {r['sk_holdout_region']:.4f}")
    print(f"  Score                     : {best_score:.4f}")

    return {
        "pool_glaciers": pool_glaciers,
        "holdout_glaciers": holdout_glaciers,
        "n_meas_pool": int(r["pool_meas_count"]),
        "n_meas_holdout": int(n_total_meas - r["pool_meas_count"]),
        "actual_pool_frac": float(r["achieved_frac"]),
        "sinkhorn_pool_vs_holdout": float(r["sk_pool_holdout"]),
        "sinkhorn_pool_vs_region": float(r["sk_pool_region"]),
        "sinkhorn_holdout_vs_region": float(r["sk_holdout_region"]),
        "blur_joint": float(blur_joint),
        "best_seed": int(r["best_seed"]),
        "best_score": float(best_score),
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


# def build_global_scalers_multi_source_simple(
#     res_xreg_by_source: dict,
#     monthly_cols: list[str],
#     static_cols: list[str],
# ) -> tuple[StandardScaler, StandardScaler, StandardScaler]:
#     """
#     Simple version of build_global_scalers_multi_source.
#     Pools df_train across all sources and fits scalers directly
#     without ID deduplication or topo aggregation.

#     Fits:
#       - scaler_m: on monthly climate columns
#       - scaler_s: on static/topographic columns
#       - scaler_joint: on all columns combined
#     """
#     df_all = pd.concat(
#         [res_xreg["df_train"] for res_xreg in res_xreg_by_source.values()],
#         ignore_index=True,
#     )

#     scaler_m = StandardScaler().fit(df_all[monthly_cols].to_numpy(dtype=np.float64))
#     scaler_s = StandardScaler().fit(df_all[static_cols].to_numpy(dtype=np.float64))
#     scaler_joint = StandardScaler().fit(
#         df_all[monthly_cols + static_cols].to_numpy(dtype=np.float64)
#     )

#     return scaler_m, scaler_s, scaler_joint


def build_global_scalers_multi_source_simple(
    res_xreg_by_source: dict,
    monthly_cols: list[str],
    static_cols: list[str],
    geo_cols: list[str] | None = None,
) -> tuple[StandardScaler, StandardScaler, StandardScaler, StandardScaler | None]:

    all_dfs = []
    for res_xreg in res_xreg_by_source.values():
        all_dfs.append(res_xreg["df_train"])
        # Also include target data so scaler covers the full feature range
        if "df_test" in res_xreg:
            all_dfs.append(res_xreg["df_test"])

    df_all = pd.concat(all_dfs, ignore_index=True).drop_duplicates()

    scaler_m = StandardScaler().fit(df_all[monthly_cols].to_numpy(dtype=np.float64))
    scaler_s = StandardScaler().fit(df_all[static_cols].to_numpy(dtype=np.float64))
    scaler_joint = StandardScaler().fit(
        df_all[monthly_cols + static_cols].to_numpy(dtype=np.float64)
    )
    scaler_geo = (
        StandardScaler().fit(df_all[geo_cols].to_numpy(dtype=np.float64))
        if geo_cols is not None
        else None
    )

    return scaler_m, scaler_s, scaler_joint, scaler_geo


def estimate_global_bandwidths_simple(
    res_xreg_by_source: dict,
    monthly_cols: list[str],
    static_cols: list[str],
    scaler_m: StandardScaler,
    scaler_s: StandardScaler,
    geo_cols: list[str] | None = None,
    scaler_geo: StandardScaler | None = None,
    blur_quantile_multiplier: float = 0.1,
    seed: int = 0,
) -> tuple[float, float, float, float | None]:
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

    if geo_cols is not None and scaler_geo is not None:
        Xg = scaler_geo.transform(df_all[geo_cols].to_numpy(dtype=np.float64))
        blur_geo = _estimate_blur(
            Xg, Xg, blur_quantile_multiplier=blur_quantile_multiplier, seed=seed + 3
        )
    else:
        blur_geo = None

    return blur_m, blur_s, blur_joint, blur_geo


def build_global_scalers_from_dfs(
    dfs: list[pd.DataFrame] | dict[str, pd.DataFrame],
    monthly_cols: list[str],
    static_cols: list[str],
    topo_id_col: str = "ID",
    glacier_col: str = "GLACIER",
) -> tuple[StandardScaler, StandardScaler, StandardScaler]:

    if isinstance(dfs, dict):
        dfs = list(dfs.values())

    df_all = pd.concat(dfs, ignore_index=True)

    # Climate scaler: all rows (monthly variability matters)
    scaler_m = StandardScaler().fit(df_all[monthly_cols].to_numpy(dtype=np.float64))

    # Topo scaler: one row per unique (GLACIER, ID) spatial point
    if topo_id_col in df_all.columns and glacier_col in df_all.columns:
        topo_data = df_all.drop_duplicates(subset=[glacier_col, topo_id_col])[
            static_cols
        ]
    else:
        topo_data = df_all.drop_duplicates(subset=static_cols)[static_cols]
    scaler_s = StandardScaler().fit(topo_data.to_numpy(dtype=np.float64))

    # Joint scaler: all rows (climate variability matters for joint)
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


def rank_glaciers_by_distance_to_target(
    res_xreg_by_source: dict,
    training_regions: list,
    test_region: str,
    monthly_cols: list,
    static_cols: list,
    scaler_m,
    scaler_s,
    blur_joint: float,
    glacier_col: str = "GLACIER",
    id_col: str = "ID",
    elev_diff_col: str = "ELEVATION_DIFFERENCE",
    max_samples: int = 2000,
    seed: int = 42,
    mode: str = "joint",  # <-- new: "joint" or "topo"
    topo_extra_cols: list = None,  # <-- new: e.g. ["ELEVATION_DIFFERENCE"]
) -> pd.DataFrame:
    """
    Rank training glaciers by Sinkhorn distance to the test region.
    mode="joint" : climate + topo features (same as split_pool_holdout_sinkhorn_v2)
    mode="topo"  : topo features only (static_cols + topo_extra_cols)
    """
    topo_extra_cols = topo_extra_cols or []

    def _build_X(df):
        pure_static = [c for c in static_cols if c != elev_diff_col]

        def _stake_topo(df_):
            parts = [df_.groupby(id_col)[pure_static].first()]
            if elev_diff_col in static_cols:
                parts.append(df_.groupby(id_col)[[elev_diff_col]].mean())
            return pd.concat(parts, axis=1)[static_cols].to_numpy(dtype=np.float64)

        id_codes_cat = pd.Categorical(df[id_col]).codes
        Xs_id = scaler_s.transform(_stake_topo(df))
        topo_per_row = Xs_id[id_codes_cat]

        if mode == "joint":
            Xm_all = scaler_m.transform(df[monthly_cols].to_numpy(dtype=np.float64))
            return np.hstack([Xm_all, topo_per_row]).astype(np.float32)

        else:  # topo
            extra_parts = []
            for col in topo_extra_cols:
                col_idx = monthly_cols.index(col)
                col_vals = df[col].to_numpy(dtype=np.float64).reshape(-1, 1)
                extra_parts.append(
                    (col_vals - scaler_m.mean_[col_idx]) / scaler_m.scale_[col_idx]
                )
            if extra_parts:
                return np.hstack([topo_per_row] + extra_parts).astype(np.float32)
            return topo_per_row.astype(np.float32)

    # blur: scale for topo-only mode same as split_pool_holdout_sinkhorn_v2
    if mode == "topo":
        n_joint = len(monthly_cols) + len(static_cols)
        n_topo = len(static_cols) + len(topo_extra_cols)
        blur = blur_joint * np.sqrt(n_topo / n_joint)
        print(
            f"  Mode: TOPO  blur={blur:.4f}  features={static_cols + topo_extra_cols}"
        )
    else:
        blur = blur_joint
        print(f"  Mode: JOINT  blur={blur:.4f}")

    loss_fn = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=blur,
        scaling=0.9,
        debias=True,
        backend="tensorized",
    )

    def _sinkhorn(Xa, Xb):
        if len(Xa) < 2 or len(Xb) < 2:
            return float("nan")
        rng = np.random.default_rng(seed)
        if len(Xa) > max_samples:
            Xa = Xa[rng.choice(len(Xa), max_samples, replace=False)]
        if len(Xb) > max_samples:
            Xb = Xb[rng.choice(len(Xb), max_samples, replace=False)]
        ta = torch.as_tensor(Xa, dtype=torch.float32)
        tb = torch.as_tensor(Xb, dtype=torch.float32)
        wa = torch.ones(len(ta)) / len(ta)
        wb = torch.ones(len(tb)) / len(tb)
        with torch.no_grad():
            return float(loss_fn(wa, ta, wb, tb).item())

    # --- test region feature matrix ---
    df_test = res_xreg_by_source[test_region]["data_monthly"]
    X_test = _build_X(df_test)
    print(
        f"Test region ({test_region}): {len(X_test)} rows, "
        f"{df_test[glacier_col].nunique()} glaciers"
    )

    # --- per-glacier distances ---
    rows = []
    for region in training_regions:
        df_region = res_xreg_by_source[region]["data_monthly"]
        glaciers = df_region[glacier_col].unique()
        print(f"\n{region}: {len(glaciers)} glaciers")

        for glacier in glaciers:
            df_gl = df_region[df_region[glacier_col] == glacier]
            if len(df_gl) < 2:
                continue
            X_gl = _build_X(df_gl)
            d = _sinkhorn(X_gl, X_test)
            rows.append(
                {
                    "glacier": glacier,
                    "region": region,
                    "distance": d,
                    "n_meas": len(df_gl),
                }
            )
            print(f"  {glacier:40s}  d={d:.4f}  n={len(df_gl)}")

    df_ranked = pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)
    df_ranked["rank"] = np.arange(len(df_ranked))
    df_ranked["cum_meas"] = df_ranked["n_meas"].cumsum()
    df_ranked["cum_frac"] = df_ranked["cum_meas"] / df_ranked["n_meas"].sum()

    print(f"\n=== Ranking complete ({mode}) ===")
    print(f"  Total glaciers : {len(df_ranked)}")
    print(
        f"  Distance range : [{df_ranked['distance'].min():.4f} "
        f"— {df_ranked['distance'].max():.4f}]"
    )
    print(f"\n  Closest 10 to {test_region}:")
    print(
        df_ranked.head(10)[
            ["rank", "glacier", "region", "distance", "n_meas", "cum_frac"]
        ].to_string(index=False)
    )

    return df_ranked


def split_pool_holdout_by_id(
    df_region: pd.DataFrame,
    id_col: str = "ID",
    train_frac: float = 0.8,
    seed: int = 0,
) -> dict:
    """
    Random split at the individual stake (ID) level.
    Fallback for regions with too few glaciers (e.g. SJM with 3 glaciers).
    """
    all_ids = sorted(df_region[id_col].unique())
    n_total_meas = len(df_region)
    target_train = int(np.round(train_frac * n_total_meas))

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(all_ids)

    train_ids, test_ids = [], []
    train_meas_count = 0

    for id_ in shuffled:
        id_meas = (df_region[id_col] == id_).sum()
        if train_meas_count < target_train:
            train_ids.append(id_)
            train_meas_count += id_meas
        else:
            test_ids.append(id_)

    actual_frac = train_meas_count / n_total_meas

    print(f"  Total measurements   : {n_total_meas}")
    print(f"  Target train meas    : {target_train} ({train_frac:.0%})")
    print(
        f"  Train : {len(train_ids)} IDs, {train_meas_count} measurements ({actual_frac:.1%})"
    )
    print(
        f"  Test  : {len(test_ids)} IDs, {n_total_meas - train_meas_count} measurements ({1-actual_frac:.1%})"
    )

    return {
        "train_glaciers": train_ids,
        "test_glaciers": test_ids,
        "n_meas_train": int(train_meas_count),
        "n_meas_test": int(n_total_meas - train_meas_count),
        "actual_train_frac": float(actual_frac),
        "sinkhorn_test_vs_region": float("nan"),
        "sinkhorn_train_vs_region": float("nan"),
    }


def split_train_test_sinkhorn(
    df_region: pd.DataFrame,
    monthly_cols: list[str],
    static_cols: list[str],
    scaler_m,
    scaler_s,
    blur_joint: float,
    glacier_col: str = "GLACIER",
    id_col: str = "ID",
    elev_diff_col: str = "ELEVATION_DIFFERENCE",
    train_frac: float = 0.8,
    seed: int = 0,
    n_restarts: int = 100,
    frac_tol: float = 0.05,
    min_train_glaciers: int = 3,
) -> dict:
    """
    Split a region into a training set and a test set using Sinkhorn divergence.

    Optimised so that:
      (1) Sinkhorn(train → test) is small  — train covers the test distribution
      (2) |Sinkhorn(train → all) - Sinkhorn(test → all)| is small — balanced shift

    Returns
    -------
    dict with keys:
        train_glaciers, test_glaciers,
        n_meas_train, n_meas_test,
        actual_train_frac,
        sinkhorn_train_vs_test,
        sinkhorn_train_vs_region,
        sinkhorn_test_vs_region,
        blur_joint, best_seed, best_score
    """
    glaciers = df_region[glacier_col].unique().tolist()
    n_glaciers = len(glaciers)
    glacier_to_idx = {g: i for i, g in enumerate(glaciers)}

    n_meas = np.array(
        [(df_region[glacier_col] == g).sum() for g in glaciers], dtype=int
    )
    n_total_meas = int(n_meas.sum())
    target_train_meas = int(round(train_frac * n_total_meas))

    print(f"  Total measurements : {n_total_meas}")
    print(f"  Target train meas  : {target_train_meas} ({train_frac:.0%})")
    print(f"  Total glaciers     : {n_glaciers}")
    print(f"  blur_joint         : {blur_joint:.4f}")

    pure_static = [c for c in static_cols if c != elev_diff_col]

    def _stake_topo(df):
        parts = [df.groupby(id_col)[pure_static].first()]
        if elev_diff_col in static_cols:
            parts.append(df.groupby(id_col)[[elev_diff_col]].mean())
        return pd.concat(parts, axis=1)[static_cols].to_numpy(dtype=np.float64)

    Xm_all = scaler_m.transform(df_region[monthly_cols].to_numpy(dtype=np.float64))
    Xs_id = scaler_s.transform(_stake_topo(df_region))
    id_codes = pd.Categorical(df_region[id_col]).codes
    topo_per_row = Xs_id[id_codes]
    X_joint_all = np.hstack([Xm_all, topo_per_row]).astype(np.float32)

    glacier_codes = np.array(
        [glacier_to_idx[g] for g in df_region[glacier_col]], dtype=int
    )

    loss_fn = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=blur_joint,
        scaling=0.9,
        debias=True,
        backend="tensorized",
    )

    def _sinkhorn(mask_a, mask_b, max_samples=2000):
        Xa = X_joint_all[mask_a]
        Xb = X_joint_all[mask_b]
        if len(Xa) < 2 or len(Xb) < 2:
            return 0.0
        if len(Xa) > max_samples:
            Xa = Xa[np.random.choice(len(Xa), max_samples, replace=False)]
        if len(Xb) > max_samples:
            Xb = Xb[np.random.choice(len(Xb), max_samples, replace=False)]
        ta = torch.as_tensor(Xa, dtype=torch.float32)
        tb = torch.as_tensor(Xb, dtype=torch.float32)
        wa = torch.ones(len(ta)) / len(ta)
        wb = torch.ones(len(tb)) / len(tb)
        with torch.no_grad():
            return float(loss_fn(wa, ta, wb, tb).item())

    def _mask_for(glacier_idxs):
        return np.isin(glacier_codes, list(glacier_idxs))

    all_mask = np.ones(len(X_joint_all), dtype=bool)

    best_score = np.inf
    best_result = None

    for restart in range(n_restarts):
        rng = np.random.default_rng(seed + restart)
        order = np.arange(n_glaciers)
        rng.shuffle(order)

        train_idxs = set()
        test_idxs = set()
        train_meas_count = 0

        for glacier_idx in order:
            gl_meas = int(n_meas[glacier_idx])
            train_full = train_meas_count + gl_meas > target_train_meas * (1 + frac_tol)

            if train_full:
                test_idxs.add(glacier_idx)
                continue

            trial_train = train_idxs | {glacier_idx}
            trial_test = test_idxs | {glacier_idx}

            test_mask = _mask_for(test_idxs) if test_idxs else all_mask
            train_mask = _mask_for(train_idxs) if train_idxs else all_mask

            score_if_train = _sinkhorn(_mask_for(trial_train), test_mask) + abs(
                _sinkhorn(_mask_for(trial_train), all_mask)
                - _sinkhorn(test_mask, all_mask)
            )
            score_if_test = _sinkhorn(train_mask, _mask_for(trial_test)) + abs(
                _sinkhorn(train_mask, all_mask)
                - _sinkhorn(_mask_for(trial_test), all_mask)
            )

            if score_if_train <= score_if_test:
                train_idxs.add(glacier_idx)
                train_meas_count += gl_meas
            else:
                test_idxs.add(glacier_idx)

        achieved_frac = train_meas_count / n_total_meas
        if abs(achieved_frac - train_frac) > frac_tol:
            continue
        if len(train_idxs) < min_train_glaciers:
            continue

        train_mask = _mask_for(train_idxs)
        test_mask = _mask_for(test_idxs)

        sk_train_test = _sinkhorn(train_mask, test_mask)
        sk_train_region = _sinkhorn(train_mask, all_mask)
        sk_test_region = _sinkhorn(test_mask, all_mask)

        score = sk_train_test + abs(sk_train_region - sk_test_region)

        print(
            f"  restart {restart:3d} | frac={achieved_frac:.2f} | "
            f"sk(train↔test)={sk_train_test:.4f} | "
            f"balance={abs(sk_train_region - sk_test_region):.4f} | "
            f"score={score:.4f}"
        )

        if score < best_score:
            best_score = score
            best_result = {
                "train_idxs": train_idxs,
                "test_idxs": test_idxs,
                "train_meas_count": train_meas_count,
                "achieved_frac": achieved_frac,
                "sk_train_test": sk_train_test,
                "sk_train_region": sk_train_region,
                "sk_test_region": sk_test_region,
                "best_seed": seed + restart,
            }

    if best_result is None:
        raise RuntimeError(
            f"No valid split found within frac_tol={frac_tol} and "
            f"min_train_glaciers={min_train_glaciers} after {n_restarts} restarts. "
            f"Try increasing n_restarts or frac_tol."
        )

    r = best_result
    train_glaciers = [glaciers[i] for i in r["train_idxs"]]
    test_glaciers = [glaciers[i] for i in r["test_idxs"]]

    print(f"\n  Best split (seed={r['best_seed']}):")
    print(
        f"  Train : {len(train_glaciers)} glaciers, {r['train_meas_count']} meas ({r['achieved_frac']:.1%})"
    )
    print(
        f"  Test  : {len(test_glaciers)} glaciers, {n_total_meas - r['train_meas_count']} meas ({1 - r['achieved_frac']:.1%})"
    )
    print(f"  Sinkhorn(train → test)   : {r['sk_train_test']:.4f}")
    print(f"  Sinkhorn(train → region) : {r['sk_train_region']:.4f}")
    print(f"  Sinkhorn(test → region)  : {r['sk_test_region']:.4f}")
    print(f"  Score                    : {best_score:.4f}")

    return {
        "train_glaciers": train_glaciers,
        "test_glaciers": test_glaciers,
        "n_meas_train": int(r["train_meas_count"]),
        "n_meas_test": int(n_total_meas - r["train_meas_count"]),
        "actual_train_frac": float(r["achieved_frac"]),
        "sinkhorn_train_vs_test": float(r["sk_train_test"]),
        "sinkhorn_train_vs_region": float(r["sk_train_region"]),
        "sinkhorn_test_vs_region": float(r["sk_test_region"]),
        "blur_joint": float(blur_joint),
        "best_seed": int(r["best_seed"]),
        "best_score": float(best_score),
    }
