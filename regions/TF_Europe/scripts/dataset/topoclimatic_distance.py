import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance


def to_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    dropna: bool = True,
) -> np.ndarray:
    """Return X (n_samples, n_features) as float64."""
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns in df: {missing}")

    X = df[feature_cols].to_numpy(dtype=np.float64, copy=True)

    if dropna:
        mask = np.isfinite(X).all(axis=1)
        X = X[mask]

    return X


def wasserstein_distance_multivariate(
    X: np.ndarray,
    Y: np.ndarray,
) -> float:
    """
    Mean 1D Wasserstein distance across features.
    Assumes features are already standardized.
    """
    if X.size == 0 or Y.size == 0:
        return np.nan

    dists = [wasserstein_distance(X[:, j], Y[:, j]) for j in range(X.shape[1])]
    return float(np.mean(dists))


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


def compute_topoclimatic_distances(
    df_src: pd.DataFrame,
    df_pool: pd.DataFrame,
    df_holdout: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler,
    seed: int,
    energy_max_samples: int = 4000,
) -> dict:
    """
    Compute topoclimatic distances in a globally standardized feature space.

    Distances returned:
      - src vs pool
      - src vs holdout
      - pool vs holdout
      - src vs tgt_all (pool ∪ holdout)

    Parameters
    ----------
    df_src, df_pool, df_holdout : pd.DataFrame
        Source / pool / holdout dataframes.
    feature_cols : list[str]
        Features used for distance computation.
    scaler : StandardScaler
        Pre-fitted global scaler, fitted once across all regions/settings.
    seed : int
        Random seed for energy-distance subsampling.
    energy_max_samples : int
        Max number of samples per set for energy distance.

    Returns
    -------
    dict
    """
    df_tgt_all = pd.concat([df_pool, df_holdout], axis=0, ignore_index=True)

    Xs = to_feature_matrix(df_src, feature_cols, dropna=True)
    Xp = to_feature_matrix(df_pool, feature_cols, dropna=True)
    Xh = to_feature_matrix(df_holdout, feature_cols, dropna=True)
    Xt = to_feature_matrix(df_tgt_all, feature_cols, dropna=True)

    if Xs.size == 0 or Xp.size == 0 or Xh.size == 0 or Xt.size == 0:
        return {
            "n_src": int(Xs.shape[0]),
            "n_pool": int(Xp.shape[0]),
            "n_holdout": int(Xh.shape[0]),
            "n_tgt_all": int(Xt.shape[0]),
            "D_energy_src_pool": np.nan,
            "D_energy_src_holdout": np.nan,
            "D_energy_pool_holdout": np.nan,
            "D_energy_src_tgt_all": np.nan,
            "D_wass_src_pool": np.nan,
            "D_wass_src_holdout": np.nan,
            "D_wass_pool_holdout": np.nan,
            "D_wass_src_tgt_all": np.nan,
        }

    # Transform with one shared global scaler
    Xs_z = scaler.transform(Xs)
    Xp_z = scaler.transform(Xp)
    Xh_z = scaler.transform(Xh)
    Xt_z = scaler.transform(Xt)

    # Energy distance
    d_en_src_pool = energy_distance(
        Xs_z, Xp_z, max_samples=energy_max_samples, seed=seed + 1
    )
    d_en_src_hold = energy_distance(
        Xs_z, Xh_z, max_samples=energy_max_samples, seed=seed + 2
    )
    d_en_pool_hold = energy_distance(
        Xp_z, Xh_z, max_samples=energy_max_samples, seed=seed + 3
    )
    d_en_src_tgt = energy_distance(
        Xs_z, Xt_z, max_samples=energy_max_samples, seed=seed + 4
    )

    # Wasserstein distance
    d_wass_src_pool = wasserstein_distance_multivariate(Xs_z, Xp_z)
    d_wass_src_hold = wasserstein_distance_multivariate(Xs_z, Xh_z)
    d_wass_pool_hold = wasserstein_distance_multivariate(Xp_z, Xh_z)
    d_wass_src_tgt = wasserstein_distance_multivariate(Xs_z, Xt_z)

    return {
        "n_src": int(Xs.shape[0]),
        "n_pool": int(Xp.shape[0]),
        "n_holdout": int(Xh.shape[0]),
        "n_tgt_all": int(Xt.shape[0]),
        "D_energy_src_pool": float(d_en_src_pool),
        "D_energy_src_holdout": float(d_en_src_hold),
        "D_energy_pool_holdout": float(d_en_pool_hold),
        "D_energy_src_tgt_all": float(d_en_src_tgt),
        "D_wass_src_pool": float(d_wass_src_pool),
        "D_wass_src_holdout": float(d_wass_src_hold),
        "D_wass_pool_holdout": float(d_wass_pool_hold),
        "D_wass_src_tgt_all": float(d_wass_src_tgt),
    }


def compute_topoclimatic_distances_sets(
    df_src: pd.DataFrame,
    df_ft: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler,
    seed: int,
    df_holdout: pd.DataFrame | None = None,
    energy_max_samples: int = 4000,
) -> dict:
    """
    Compute multivariate topoclimatic distances between source, fine-tuning,
    and optionally hold-out distributions.

    Parameters
    ----------
    df_src : pd.DataFrame
        Source-region samples.
    df_ft : pd.DataFrame
        Fine-tuning / monitoring samples.
    feature_cols : list[str]
        Feature columns used to build the multivariate distribution.
    scaler : StandardScaler
        Fitted scaler used to standardize all feature matrices in the same space.
        This should normally be fitted once on a reference dataset and reused.
    seed : int
        Random seed used for subsampling in the energy distance computation.
    df_holdout : pd.DataFrame | None, optional
        Hold-out / evaluation samples. If provided, distances involving the
        hold-out set are also computed.
    energy_max_samples : int, default=4000
        Maximum number of samples used inside the energy distance computation.

    Returns
    -------
    dict
        Dictionary containing sample counts and pairwise distances.
    """
    Xs = to_feature_matrix(df_src, feature_cols, dropna=True)
    Xf = to_feature_matrix(df_ft, feature_cols, dropna=True)

    Xh = None
    if df_holdout is not None:
        Xh = to_feature_matrix(df_holdout, feature_cols, dropna=True)

    out = {
        "n_src": int(Xs.shape[0]),
        "n_ft": int(Xf.shape[0]),
        "n_holdout": int(Xh.shape[0]) if Xh is not None else np.nan,
        "D_energy_src_ft": np.nan,
        "D_wass_src_ft": np.nan,
        "D_energy_ft_holdout": np.nan,
        "D_wass_ft_holdout": np.nan,
        "D_energy_src_holdout": np.nan,
        "D_wass_src_holdout": np.nan,
    }

    # Need at least src and ft to do anything
    if Xs.size == 0 or Xf.size == 0:
        return out

    Xs_z = scaler.transform(Xs)
    Xf_z = scaler.transform(Xf)

    out["D_energy_src_ft"] = float(
        energy_distance(Xs_z, Xf_z, max_samples=energy_max_samples, seed=seed + 101)
    )
    out["D_wass_src_ft"] = float(wasserstein_distance_multivariate(Xs_z, Xf_z))

    if Xh is not None and Xh.size > 0:
        Xh_z = scaler.transform(Xh)

        out["D_energy_ft_holdout"] = float(
            energy_distance(Xf_z, Xh_z, max_samples=energy_max_samples, seed=seed + 202)
        )
        out["D_wass_ft_holdout"] = float(wasserstein_distance_multivariate(Xf_z, Xh_z))

        out["D_energy_src_holdout"] = float(
            energy_distance(Xs_z, Xh_z, max_samples=energy_max_samples, seed=seed + 303)
        )
        out["D_wass_src_holdout"] = float(wasserstein_distance_multivariate(Xs_z, Xh_z))

    return out
