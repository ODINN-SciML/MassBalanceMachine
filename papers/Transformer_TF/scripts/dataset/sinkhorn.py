from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import math
import numpy as np
import torch
from torch import nn
from geomloss import SamplesLoss
from sklearn.preprocessing import StandardScaler


@dataclass
class RelaxedSplitResult:
    pi_groups: np.ndarray
    logits_groups: np.ndarray
    objective: float
    sinkhorn_divergence: float
    glacier_fraction_soft: float
    measurement_fraction_soft: float
    binarization_penalty: float
    blur: float
    epsilon: float
    n_iter: int
    converged: bool


@dataclass
class HardSplitResult:
    train_indices: np.ndarray  # monthly-row indices
    test_indices: np.ndarray  # monthly-row indices
    train_measurement_indices: np.ndarray  # unique-measurement indices
    test_measurement_indices: np.ndarray  # unique-measurement indices
    train_groups: np.ndarray
    test_groups: np.ndarray
    test_group_mask: np.ndarray
    scores_used: np.ndarray
    threshold: Optional[float]
    actual_n_test_groups: int
    actual_n_test_measurements: int
    actual_n_test_monthly_rows: int
    glacier_fraction_test: float
    measurement_fraction_test: float
    sinkhorn_divergence: float
    objective: float


class SinkhornGroupSplit(nn.Module):
    """
    Continuous-relaxation group split based on Sinkhorn divergence.

    Each group g has a learnable logit theta_g and a soft test membership
        pi_g = sigmoid(theta_g) in (0, 1)

    Observations belonging to group g inherit pi_g:
        - test weight contribution proportional to pi_g
        - train weight contribution proportional to (1 - pi_g)

    The optimization objective is:

        J = SinkhornDivergence(train_measure(pi), test_measure(pi))
            + lambda_glaciers * (mean(pi) - target_test_glacier_fraction)^2
            + lambda_observations * (soft_test_obs_fraction(pi) - target_test_obs_fraction)^2
            + lambda_binary * mean(pi * (1 - pi))

    Notes
    -----
    - This is a *relaxation*: groups may temporarily contribute to both train and test.
    - No hard projection is done here.
    - Annealing is supported by continuing optimization with a changing lambda_binary schedule.

    Parameters
    ----------
    test_size_glaciers : float, default=0.2
        Target soft fraction of glaciers in the test split.

    test_size_observations : float, default=0.2
        Target soft fraction of observations in the test split.

    lambda_glaciers : float, default=10.0
        Penalty weight for mismatch in glacier fraction.

    lambda_observations : float, default=10.0
        Penalty weight for mismatch in observation fraction.

    lambda_binary : float, default=0.0
        Initial binarization penalty. Set to 0.0 for the first optimization stage.

    p : int, default=2
        Ground metric exponent for Sinkhorn.

    blur_quantile_multiplier : float, default=0.1
        Automatic blur scaling based on median pairwise squared distance.

    scaling : float, default=0.9
        GeomLoss scaling parameter.

    reach : float or None, default=None
        Optional GeomLoss reach parameter.

    sinkhorn_backend : str, default="auto"
        GeomLoss backend: "auto", "tensorized", "online", or "multiscale".

    optimizer_name : str, default="adam"
        Either "adam" or "lbfgs".

    lr : float, default=0.1
        Learning rate for Adam.

    max_iter : int, default=500
        Maximum number of iterations for a single optimization call.

    tol_rel_obj : float, default=1e-5
        Relative objective-improvement tolerance used for stopping.

    patience : int, default=30
        Number of consecutive small-improvement iterations before stopping.

    subsample_size : int or None, default=None
        If given, compute Sinkhorn divergence on a fixed random subset of points
        of that size. This speeds up optimization on large datasets.

    device : str or None, default=None
        Torch device. If None, uses "cuda" if available else "cpu".

    dtype : torch.dtype, default=torch.float32
        Torch dtype.

    random_state : int or None, default=None
        Random seed.

    verbose : bool, default=False
        Print optimization progress.

    track_history : bool, default=True
        Store iteration history in history_.

    freq_print : int, default=25
        Print frequency of the cost function.
    """

    def __init__(
        self,
        test_size_glaciers: float = 0.2,
        test_size_observations: float = 0.2,
        lambda_glaciers: float = 10.0,
        lambda_observations: float = 10.0,
        lambda_binary: float = 0.0,
        p: int = 2,
        blur_quantile_multiplier: float = 0.1,
        scaling: float = 0.9,
        reach: Optional[float] = None,
        sinkhorn_backend: str = "auto",
        optimizer_name: str = "adam",
        lr: float = 0.1,
        max_iter: int = 500,
        tol_rel_obj: float = 1e-5,
        patience: int = 30,
        subsample_size: Optional[int] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        random_state: Optional[int] = None,
        verbose: bool = False,
        track_history: bool = True,
        freq_print: int = 25,
    ) -> None:
        super().__init__()

        if not (0.0 < test_size_glaciers < 1.0):
            raise ValueError("test_size_glaciers must be in (0, 1).")
        if not (0.0 < test_size_observations < 1.0):
            raise ValueError("test_size_observations must be in (0, 1).")
        if optimizer_name.lower() not in {"adam", "lbfgs"}:
            raise ValueError("optimizer_name must be 'adam' or 'lbfgs'.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if patience <= 0:
            raise ValueError("patience must be positive.")

        self.test_size_glaciers = float(test_size_glaciers)
        self.test_size_observations = float(test_size_observations)
        self.lambda_glaciers = float(lambda_glaciers)
        self.lambda_observations = float(lambda_observations)
        self.lambda_binary = float(lambda_binary)

        self.p = int(p)
        self.blur_quantile_multiplier = float(blur_quantile_multiplier)
        self.scaling = float(scaling)
        self.reach = reach
        self.sinkhorn_backend = sinkhorn_backend

        self.optimizer_name = optimizer_name.lower()
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.tol_rel_obj = float(tol_rel_obj)
        self.patience = int(patience)

        self.subsample_size = subsample_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.random_state = random_state
        self.verbose = verbose
        self.track_history = track_history
        self.freq_print = freq_print

        self._rng = np.random.default_rng(self.random_state)

        # Learned / prepared during fit
        self.scaler_: Optional[StandardScaler] = None
        self.blur_: Optional[float] = None
        self.loss_fn_: Optional[SamplesLoss] = None

        self.unique_groups_: Optional[np.ndarray] = None
        self.group_inverse_: Optional[np.ndarray] = None
        self.group_sizes_: Optional[np.ndarray] = None
        self.n_groups_: Optional[int] = None
        self.n_samples_: Optional[int] = None

        self.X_scaled_np_: Optional[np.ndarray] = None
        self.group_index_torch_: Optional[torch.Tensor] = None
        self.group_sizes_torch_: Optional[torch.Tensor] = None

        self.X_opt_: Optional[torch.Tensor] = None
        self.logits_: Optional[nn.Parameter] = None

        self.history_: Dict[str, List[float]] = {}
        self.phase_history_: List[Dict[str, object]] = []

        self.unique_measurements_: Optional[np.ndarray] = None
        self.measurement_inverse_rows_: Optional[np.ndarray] = None
        self.measurement_group_inverse_: Optional[np.ndarray] = None
        self.group_measurement_counts_: Optional[np.ndarray] = None
        self.n_measurements_: Optional[int] = None

        self.group_measurement_counts_torch_: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        groups: np.ndarray,
        measurement_ids: np.ndarray,
        init_logits: Optional[np.ndarray] = None,
    ) -> RelaxedSplitResult:
        """
        First optimization phase, typically without annealing.
        This initializes the model state from data and optimizes the relaxed split.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        groups : ndarray of shape (n_samples,)
        init_logits : ndarray of shape (n_groups,), optional
            Initial logits for groups. If None, initialize all logits so that
            sigmoid(logit) ~= test_size_glaciers.

        Returns
        -------
        RelaxedSplitResult
        """
        self._prepare_data(X, groups, measurement_ids)
        self._initialize_parameters(init_logits=init_logits)
        result = self._optimize(
            phase_name="fit",
            max_iter=self.max_iter,
            lambda_binary_schedule=None,
            reset_history=True,
        )
        return result

    def continue_with_schedule(
        self,
        schedule: Sequence[Dict[str, object]],
        reset_phase_history: bool = False,
    ) -> RelaxedSplitResult:
        """
        Continue optimization from the current solution using a custom optimizer schedule.

        Each stage in `schedule` is a dict with keys:
            - optimizer: "adam" or "lbfgs"   (required)
            - max_iter: int                  (required)
            - lr: float                      (optional; defaults to self.lr)
            - lambda_binary: float           (optional; defaults to self.lambda_binary)

        Example
        -------
        schedule = [
            {"optimizer": "adam", "max_iter": 200, "lr": 0.05},
            {"optimizer": "lbfgs", "max_iter": 80, "lr": 0.5},
        ]
        """
        if self.logits_ is None:
            raise RuntimeError("Call fit(...) before continue_with_schedule(...).")
        if len(schedule) == 0:
            raise ValueError("schedule must not be empty.")

        if reset_phase_history:
            self.phase_history_ = []

        return self._optimize_with_schedule(
            schedule=schedule,
            phase_name="scheduled",
            reset_history=False,
        )

    def continue_with_annealing(
        self,
        lambda_binary_schedule: Sequence[float],
        max_iter_per_stage: int = 200,
        adam_first: bool = False,
        adam_iter: int = 50,
        adam_lr: Optional[float] = None,
        lbfgs_lr: Optional[float] = None,
        reset_phase_history: bool = False,
    ) -> RelaxedSplitResult:
        """
        Continue optimization from the current solution while annealing lambda_binary.

        If adam_first=True, each annealing level runs:
            1) Adam
            2) LBFGS

        Otherwise, each level uses the class default optimizer.
        """
        if self.logits_ is None:
            raise RuntimeError("Call fit(...) before continue_with_annealing(...).")
        if len(lambda_binary_schedule) == 0:
            raise ValueError("lambda_binary_schedule must not be empty.")

        if reset_phase_history:
            self.phase_history_ = []

        schedule = []
        for lam in lambda_binary_schedule:
            if adam_first:
                if adam_iter > 0:
                    schedule.append(
                        {
                            "optimizer": "adam",
                            "max_iter": adam_iter,
                            "lr": self.lr if adam_lr is None else float(adam_lr),
                            "lambda_binary": float(lam),
                        }
                    )
                schedule.append(
                    {
                        "optimizer": "lbfgs",
                        "max_iter": max_iter_per_stage,
                        "lr": self.lr if lbfgs_lr is None else float(lbfgs_lr),
                        "lambda_binary": float(lam),
                    }
                )
            else:
                schedule.append(
                    {
                        "optimizer": self.optimizer_name,
                        "max_iter": max_iter_per_stage,
                        "lr": self.lr,
                        "lambda_binary": float(lam),
                    }
                )

        return self._optimize_with_schedule(
            schedule=schedule,
            phase_name="anneal",
            reset_history=False,
        )

    def _optimize_with_schedule(
        self,
        schedule: Sequence[Dict[str, object]],
        phase_name: str,
        reset_history: bool,
    ) -> RelaxedSplitResult:
        self._check_ready_for_optimization()

        if reset_history:
            self.history_ = {
                "objective": [],
                "sinkhorn": [],
                "penalty_glaciers": [],
                "penalty_observations": [],
                "penalty_binary": [],
                "lambda_binary": [],
                "glacier_fraction_soft": [],
                "measurement_fraction_soft": [],
                "binarization_value": [],
                "grad_norm": [],
                "ambiguous_groups": [],
                "n_train_groups_hard": [],
                "n_test_groups_hard": [],
                "phase_name": [],
            }

        total_iter = 0
        converged_all = True
        last_stats = None

        for stage_idx, stage in enumerate(schedule):
            optimizer_name = str(stage["optimizer"]).lower()
            max_iter = int(stage["max_iter"])
            lr = float(stage.get("lr", self.lr))
            lambda_binary = float(stage.get("lambda_binary", self.lambda_binary))

            if optimizer_name not in {"adam", "lbfgs"}:
                raise ValueError(f"Unknown optimizer '{optimizer_name}'.")

            stage_result = self._run_single_stage(
                optimizer_name=optimizer_name,
                lambda_binary=lambda_binary,
                max_iter=max_iter,
                lr=lr,
                phase_name=f"{phase_name}_{stage_idx}_{optimizer_name}",
            )
            total_iter += stage_result["n_iter"]
            last_stats = stage_result

            self.phase_history_.append(
                {
                    "phase_name": phase_name,
                    "stage_index": stage_idx,
                    "optimizer": optimizer_name,
                    "lambda_binary": lambda_binary,
                    "lr": lr,
                    "n_iter": int(stage_result["n_iter"]),
                    "converged": bool(stage_result["converged"]),
                    "final_objective": float(stage_result["objective"]),
                }
            )

            if not stage_result["converged"]:
                converged_all = False

        assert last_stats is not None
        return self._build_result(
            objective=float(last_stats["objective"]),
            sinkhorn_divergence=float(last_stats["sinkhorn"]),
            glacier_fraction_soft=float(last_stats["glacier_fraction_soft"]),
            measurement_fraction_soft=float(last_stats["measurement_fraction_soft"]),
            binarization_penalty=float(last_stats["binarization_value"]),
            n_iter=total_iter,
            converged=converged_all,
        )

    def get_group_probabilities(self) -> np.ndarray:
        """Return current soft test memberships pi_g."""
        self._check_fitted()
        with torch.no_grad():
            return torch.sigmoid(self.logits_).detach().cpu().numpy().copy()

    def get_group_logits(self) -> np.ndarray:
        """Return current group logits theta_g."""
        self._check_fitted()
        with torch.no_grad():
            return self.logits_.detach().cpu().numpy().copy()

    def get_observation_soft_memberships(self) -> np.ndarray:
        """Return soft test memberships for each observation."""
        self._check_fitted()
        with torch.no_grad():
            pi_g = torch.sigmoid(self.logits_)
            pi_obs = pi_g[self.group_index_torch_]
            return pi_obs.detach().cpu().numpy().copy()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def plot_history(self) -> None:
        """Quick diagnostic plots for the full optimization history."""
        self._check_fitted()
        if not self.history_:
            raise RuntimeError(
                "No history available. Set track_history=True and run fit()."
            )

        import matplotlib.pyplot as plt

        it = np.arange(1, len(self.history_["objective"]) + 1)

        plt.figure(figsize=(7, 4))
        plt.plot(it, self.history_["objective"], label="objective")
        plt.plot(it, self.history_["sinkhorn"], label="sinkhorn")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.title("Objective and Sinkhorn divergence")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7, 4))
        plt.plot(it, self.history_["penalty_glaciers"], label="glacier penalty")
        plt.plot(it, self.history_["penalty_observations"], label="observation penalty")
        plt.plot(it, self.history_["penalty_binary"], label="binary penalty")
        plt.xlabel("Iteration")
        plt.ylabel("Penalty")
        plt.title("Penalty terms")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7, 4))
        plt.plot(
            it, self.history_["glacier_fraction_soft"], label="soft glacier fraction"
        )
        plt.plot(
            it,
            self.history_["measurement_fraction_soft"],
            label="soft measurement fraction",
        )
        plt.axhline(
            self.test_size_glaciers, linestyle="--", label="target glacier fraction"
        )
        plt.axhline(
            self.test_size_observations,
            linestyle=":",
            label="target measurement fraction",
        )
        plt.xlabel("Iteration")
        plt.ylabel("Fraction")
        plt.title("Soft split fractions")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7, 4))
        plt.plot(it, self.history_["grad_norm"], label="grad norm")
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Gradient norm")
        plt.title("Gradient norm")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7, 4))
        plt.plot(it, self.history_["ambiguous_groups"], label="ambiguous groups")
        plt.xlabel("Iteration")
        plt.ylabel("Count")
        plt.title("Number of ambiguous groups (0.1 < pi < 0.9)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_group_probability_histogram(self, bins: int = 20) -> None:
        """Histogram of current soft group memberships."""
        self._check_fitted()
        import matplotlib.pyplot as plt

        pi = self.get_group_probabilities()
        plt.figure(figsize=(6, 4))
        plt.hist(pi, bins=bins)
        plt.xlabel("pi_g")
        plt.ylabel("Count")
        plt.title("Current soft group memberships")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------
    def _prepare_data(
        self, X: np.ndarray, groups: np.ndarray, measurement_ids: np.ndarray
    ) -> None:
        X = np.asarray(X, dtype=np.float64)
        groups = np.asarray(groups)
        measurement_ids = np.asarray(measurement_ids)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if groups.ndim != 1:
            raise ValueError("groups must be a 1D array.")
        if measurement_ids.ndim != 1:
            raise ValueError("measurement_ids must be a 1D array.")
        if len(X) != len(groups) or len(X) != len(measurement_ids):
            raise ValueError(
                "X, groups, and measurement_ids must have the same number of rows."
            )
        if len(X) < 2:
            raise ValueError("Need at least 2 monthly rows.")

        # Glacier groups
        unique_groups, group_inverse = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)
        if n_groups < 2:
            raise ValueError("Need at least 2 distinct groups.")

        # Measurements
        unique_measurements, measurement_first_idx, measurement_inverse_rows = (
            np.unique(
                measurement_ids,
                return_index=True,
                return_inverse=True,
            )
        )
        n_measurements = len(unique_measurements)
        if n_measurements < 2:
            raise ValueError("Need at least 2 distinct measurements.")

        # Each measurement must belong to exactly one glacier
        measurement_group_inverse = group_inverse[measurement_first_idx]
        if not np.all(
            group_inverse == measurement_group_inverse[measurement_inverse_rows]
        ):
            raise ValueError(
                "Each measurement_id must belong to exactly one glacier group."
            )

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X).astype(np.float32)

        # Optional subsampling for Sinkhorn only (monthly rows)
        if self.subsample_size is not None and self.subsample_size < len(X_scaled):
            idx = self._rng.choice(
                len(X_scaled), size=self.subsample_size, replace=False
            )
            X_opt = X_scaled[idx]
            group_inverse_opt = group_inverse[idx]
        else:
            X_opt = X_scaled
            group_inverse_opt = group_inverse

        self.blur_ = self._estimate_blur(X_opt)
        self.loss_fn_ = SamplesLoss(
            loss="sinkhorn",
            p=self.p,
            blur=self.blur_,
            scaling=self.scaling,
            reach=self.reach,
            backend=self.sinkhorn_backend,
            debias=True,
        )

        group_sizes = np.bincount(group_inverse, minlength=n_groups).astype(
            np.int64
        )  # monthly rows per glacier
        group_measurement_counts = np.bincount(
            measurement_group_inverse, minlength=n_groups
        ).astype(np.int64)

        self.unique_groups_ = unique_groups
        self.group_inverse_ = group_inverse
        self.group_sizes_ = group_sizes
        self.n_groups_ = n_groups
        self.n_samples_ = len(X)  # monthly rows

        self.unique_measurements_ = unique_measurements
        self.measurement_inverse_rows_ = measurement_inverse_rows
        self.measurement_group_inverse_ = measurement_group_inverse
        self.group_measurement_counts_ = group_measurement_counts
        self.n_measurements_ = n_measurements

        self.X_scaled_np_ = X_scaled

        self.X_opt_ = torch.as_tensor(
            X_opt, dtype=self.dtype, device=self.device
        ).contiguous()
        self.group_index_torch_ = torch.as_tensor(
            group_inverse_opt, dtype=torch.long, device=self.device
        ).contiguous()
        self.group_sizes_torch_ = torch.as_tensor(
            group_sizes, dtype=self.dtype, device=self.device
        ).contiguous()
        self.group_measurement_counts_torch_ = torch.as_tensor(
            group_measurement_counts,
            dtype=self.dtype,
            device=self.device,
        ).contiguous()

    def _initialize_parameters(self, init_logits: Optional[np.ndarray] = None) -> None:
        if self.n_groups_ is None:
            raise RuntimeError("Data has not been prepared.")

        if init_logits is None:
            p = np.clip(self.test_size_glaciers, 1e-4, 1 - 1e-4)
            init_value = math.log(p / (1 - p))
            init = np.full(self.n_groups_, init_value, dtype=np.float32)
        else:
            init = np.asarray(init_logits, dtype=np.float32)
            if init.shape != (self.n_groups_,):
                raise ValueError(f"init_logits must have shape ({self.n_groups_},).")

        self.logits_ = nn.Parameter(
            torch.as_tensor(init, dtype=self.dtype, device=self.device)
        )

    # ------------------------------------------------------------------
    # Optimization core
    # ------------------------------------------------------------------
    def _optimize(
        self,
        phase_name: str,
        max_iter: int,
        lambda_binary_schedule: Optional[List[float]],
        reset_history: bool,
    ) -> RelaxedSplitResult:
        if lambda_binary_schedule is None:
            schedule = [
                {
                    "optimizer": self.optimizer_name,
                    "max_iter": max_iter,
                    "lr": self.lr,
                    "lambda_binary": self.lambda_binary,
                }
            ]
        else:
            schedule = [
                {
                    "optimizer": self.optimizer_name,
                    "max_iter": max_iter,
                    "lr": self.lr,
                    "lambda_binary": float(lam),
                }
                for lam in lambda_binary_schedule
            ]

        return self._optimize_with_schedule(
            schedule=schedule,
            phase_name=phase_name,
            reset_history=reset_history,
        )

    def _run_single_stage(
        self,
        optimizer_name: str,
        lambda_binary: float,
        max_iter: int,
        lr: float,
        phase_name: str,
    ) -> Dict[str, float]:
        if optimizer_name == "adam":
            return self._run_single_stage_adam(
                lambda_binary=lambda_binary,
                max_iter=max_iter,
                lr=lr,
                phase_name=phase_name,
            )
        if optimizer_name == "lbfgs":
            return self._run_single_stage_lbfgs(
                lambda_binary=lambda_binary,
                max_iter=max_iter,
                lr=lr,
                phase_name=phase_name,
            )
        raise ValueError(f"Unknown optimizer '{optimizer_name}'.")

    def _run_single_stage_adam(
        self,
        lambda_binary: float,
        max_iter: int,
        lr: float,
        phase_name: str,
    ) -> Dict[str, float]:
        optimizer = torch.optim.Adam([self.logits_], lr=lr)

        best_obj = math.inf
        n_bad = 0
        converged = False
        last_stats = None

        for it in range(max_iter):
            optimizer.zero_grad(set_to_none=True)
            stats = self._compute_terms(lambda_binary=lambda_binary)
            stats["objective"].backward()

            grad_norm = self._compute_grad_norm()
            optimizer.step()

            obj_val = float(stats["objective"].detach().cpu().item())
            last_stats = self._detach_stats(
                stats,
                grad_norm=grad_norm,
                phase_name=phase_name,
                lambda_binary=lambda_binary,
            )

            if self.track_history:
                self._append_history(last_stats)

            rel_impr = (
                (best_obj - obj_val) / (abs(best_obj) + 1e-12)
                if math.isfinite(best_obj)
                else math.inf
            )
            if obj_val < best_obj and rel_impr > self.tol_rel_obj:
                best_obj = obj_val
                n_bad = 0
            elif obj_val < best_obj:
                best_obj = obj_val
                n_bad += 1
            else:
                n_bad += 1

            if n_bad >= self.patience:
                converged = True

            if (
                self.verbose
                and ((it + 1) % self.freq_print == 0 or it == 0 or it == max_iter - 1)
                or converged
            ):
                print(
                    f"[{phase_name}] iter={it+1:04d} "
                    f"obj={obj_val:.6f} "
                    f"sink={last_stats['sinkhorn']:.6f} "
                    f"lambda_bin={lambda_binary:.4g} "
                    f"amb={int(last_stats['ambiguous_groups'])} "
                    f"train={int(last_stats['n_train_groups_hard'])} "
                    f"test={int(last_stats['n_test_groups_hard'])}"
                )

            if converged:
                break

        if last_stats is None:
            raise RuntimeError(
                "Optimization stage failed before producing any statistics."
            )

        last_stats["n_iter"] = it + 1
        last_stats["converged"] = converged
        return last_stats

    def _run_single_stage_lbfgs(
        self,
        lambda_binary: float,
        max_iter: int,
        lr: float,
        phase_name: str,
    ) -> Dict[str, float]:
        optimizer = torch.optim.LBFGS(
            [self.logits_],
            lr=lr,
            max_iter=20,
            history_size=20,
            line_search_fn="strong_wolfe",
        )

        best_obj = math.inf
        n_bad = 0
        converged = False
        last_stats = None

        for it in range(max_iter):
            cached = {}

            def closure():
                optimizer.zero_grad(set_to_none=True)
                stats = self._compute_terms(lambda_binary=lambda_binary)
                stats["objective"].backward()
                cached["stats"] = stats
                return stats["objective"]

            optimizer.step(closure)

            if "stats" not in cached:
                raise RuntimeError("LBFGS closure did not produce stats.")

            stats = cached["stats"]
            grad_norm = self._compute_grad_norm()
            obj_val = float(stats["objective"].detach().cpu().item())
            last_stats = self._detach_stats(
                stats,
                grad_norm=grad_norm,
                phase_name=phase_name,
                lambda_binary=lambda_binary,
            )

            if self.track_history:
                self._append_history(last_stats)

            rel_impr = (
                (best_obj - obj_val) / (abs(best_obj) + 1e-12)
                if math.isfinite(best_obj)
                else math.inf
            )
            if obj_val < best_obj and rel_impr > self.tol_rel_obj:
                best_obj = obj_val
                n_bad = 0
            elif obj_val < best_obj:
                best_obj = obj_val
                n_bad += 1
            else:
                n_bad += 1

            if n_bad >= self.patience:
                converged = True

            if (
                self.verbose
                and ((it + 1) % self.freq_print == 0 or it == 0 or it == max_iter - 1)
                or converged
            ):
                print(
                    f"[{phase_name}] iter={it+1:04d} "
                    f"obj={obj_val:.6f} "
                    f"sink={last_stats['sinkhorn']:.6f} "
                    f"lambda_bin={lambda_binary:.4g} "
                    f"amb={int(last_stats['ambiguous_groups'])} "
                    f"train={int(last_stats['n_train_groups_hard'])} "
                    f"test={int(last_stats['n_test_groups_hard'])}"
                )

            if converged:
                break

        if last_stats is None:
            raise RuntimeError(
                "Optimization stage failed before producing any statistics."
            )

        last_stats["n_iter"] = it + 1
        last_stats["converged"] = converged
        return last_stats

    # ------------------------------------------------------------------
    # Objective terms
    # ------------------------------------------------------------------
    def _compute_terms(self, lambda_binary: float) -> Dict[str, torch.Tensor]:
        """
        Compute all objective terms for the current logits.

        Sinkhorn is computed on monthly rows.
        The size penalty is computed on unique measurements.
        """
        pi_g = torch.sigmoid(self.logits_)  # (G,)

        # Monthly rows inherit their glacier soft membership
        pi_month = pi_g[self.group_index_torch_]  # (N_month_opt,)

        test_mass = torch.sum(pi_month)
        train_mass = torch.sum(1.0 - pi_month)

        eps_mass = torch.tensor(1e-8, dtype=self.dtype, device=self.device)
        test_mass = torch.clamp(test_mass, min=eps_mass)
        train_mass = torch.clamp(train_mass, min=eps_mass)

        a = (1.0 - pi_month) / train_mass
        b = pi_month / test_mass

        # Sinkhorn on monthly features
        sinkhorn = self.loss_fn_(a, self.X_opt_, b, self.X_opt_)

        glacier_fraction_soft = torch.mean(pi_g)

        # Measurement fraction is computed on unique measurements, not monthly rows
        measurement_fraction_soft = torch.sum(
            self.group_measurement_counts_torch_ * pi_g
        ) / torch.sum(self.group_measurement_counts_torch_)

        penalty_glaciers = (glacier_fraction_soft - self.test_size_glaciers) ** 2
        penalty_observations = (
            measurement_fraction_soft - self.test_size_observations
        ) ** 2

        # Small when pi_g is close to 0 or 1
        binarization_value = torch.mean(pi_g * (1.0 - pi_g))
        penalty_binary = lambda_binary * binarization_value

        objective = (
            sinkhorn
            + self.lambda_glaciers * penalty_glaciers
            + self.lambda_observations * penalty_observations
            + penalty_binary
        )

        return {
            "objective": objective,
            "sinkhorn": sinkhorn,
            "penalty_glaciers": self.lambda_glaciers * penalty_glaciers,
            "penalty_observations": self.lambda_observations * penalty_observations,
            "penalty_binary": penalty_binary,
            "glacier_fraction_soft": glacier_fraction_soft,
            "measurement_fraction_soft": measurement_fraction_soft,
            "binarization_value": binarization_value,
            "pi_g": pi_g,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_result(
        self,
        objective: float,
        sinkhorn_divergence: float,
        glacier_fraction_soft: float,
        measurement_fraction_soft: float,
        binarization_penalty: float,
        n_iter: int,
        converged: bool,
    ) -> RelaxedSplitResult:
        pi_groups = self.get_group_probabilities()
        logits_groups = self.get_group_logits()

        return RelaxedSplitResult(
            pi_groups=pi_groups,
            logits_groups=logits_groups,
            objective=float(objective),
            sinkhorn_divergence=float(sinkhorn_divergence),
            glacier_fraction_soft=float(glacier_fraction_soft),
            measurement_fraction_soft=float(measurement_fraction_soft),
            binarization_penalty=float(binarization_penalty),
            blur=float(self.blur_),
            epsilon=float(self.blur_**self.p),
            n_iter=int(n_iter),
            converged=bool(converged),
        )

    def _estimate_blur(self, X_scaled: np.ndarray, max_points: int = 4000) -> float:
        """
        Estimate blur automatically from the median pairwise squared distance.

        For p=2:
            epsilon ~= alpha * median(||x_i - x_j||^2)
            blur = sqrt(epsilon)
        """
        if len(X_scaled) > max_points:
            idx = self._rng.choice(len(X_scaled), size=max_points, replace=False)
            X_sub = X_scaled[idx]
        else:
            X_sub = X_scaled

        n = len(X_sub)
        if n < 2:
            return 0.5

        n_pairs = min(20000, n * (n - 1) // 2)
        i = self._rng.integers(0, n, size=n_pairs)
        j = self._rng.integers(0, n, size=n_pairs)
        mask = i != j
        i, j = i[mask], j[mask]

        if len(i) == 0:
            return 0.5

        sq_dists = np.sum((X_sub[i] - X_sub[j]) ** 2, axis=1)
        median_sq_dist = max(float(np.median(sq_dists)), 1e-8)

        epsilon = self.blur_quantile_multiplier * median_sq_dist
        if self.p == 2:
            blur = math.sqrt(epsilon)
        else:
            blur = epsilon ** (1.0 / self.p)

        return max(float(blur), 1e-4)

    def _compute_grad_norm(self) -> float:
        if self.logits_.grad is None:
            return 0.0
        return float(torch.linalg.norm(self.logits_.grad.detach()).cpu().item())

    def _detach_stats(
        self,
        stats: Dict[str, torch.Tensor],
        grad_norm: float,
        phase_name: str,
        lambda_binary: float,
    ) -> Dict[str, float]:
        with torch.no_grad():
            pi_g = stats["pi_g"].detach()
            ambiguous = torch.sum((pi_g > 0.1) & (pi_g < 0.9)).item()

            test_mask_hard = pi_g > 0.5
            n_test_groups_hard = torch.sum(test_mask_hard).item()
            n_train_groups_hard = len(pi_g) - n_test_groups_hard

        return {
            "objective": float(stats["objective"].detach().cpu().item()),
            "sinkhorn": float(stats["sinkhorn"].detach().cpu().item()),
            "penalty_glaciers": float(stats["penalty_glaciers"].detach().cpu().item()),
            "penalty_observations": float(
                stats["penalty_observations"].detach().cpu().item()
            ),
            "penalty_binary": float(stats["penalty_binary"].detach().cpu().item()),
            "lambda_binary": float(lambda_binary),
            "glacier_fraction_soft": float(
                stats["glacier_fraction_soft"].detach().cpu().item()
            ),
            "measurement_fraction_soft": float(
                stats["measurement_fraction_soft"].detach().cpu().item()
            ),
            "binarization_value": float(
                stats["binarization_value"].detach().cpu().item()
            ),
            "grad_norm": float(grad_norm),
            "ambiguous_groups": float(ambiguous),
            "n_train_groups_hard": float(n_train_groups_hard),
            "n_test_groups_hard": float(n_test_groups_hard),
            "phase_name": phase_name,
        }

    def _append_history(self, last_stats: Dict[str, float]) -> None:
        for key in self.history_.keys():
            self.history_[key].append(last_stats[key])

    def _check_ready_for_optimization(self) -> None:
        if self.loss_fn_ is None or self.X_opt_ is None or self.logits_ is None:
            raise RuntimeError(
                "Model is not initialized. Call fit(...) first or prepare data properly."
            )

    def _check_fitted(self) -> None:
        if self.logits_ is None:
            raise RuntimeError("The model is not fitted yet.")

    # ------------------------------------------------------------------
    # Convenience summaries
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, float]:
        """Return a compact summary of the current solution."""
        self._check_fitted()
        with torch.no_grad():
            stats = self._compute_terms(lambda_binary=self.lambda_binary)
            pi_g = torch.sigmoid(self.logits_)
            ambiguous = torch.sum((pi_g > 0.1) & (pi_g < 0.9)).item()
            test_mask_hard = pi_g > 0.5
            n_test_groups_hard = torch.sum(test_mask_hard).item()
            n_train_groups_hard = len(pi_g) - n_test_groups_hard

        return {
            "objective": float(stats["objective"].detach().cpu().item()),
            "sinkhorn_divergence": float(stats["sinkhorn"].detach().cpu().item()),
            "soft_glacier_fraction_test": float(
                stats["glacier_fraction_soft"].detach().cpu().item()
            ),
            "soft_measurement_fraction_test": float(
                stats["measurement_fraction_soft"].detach().cpu().item()
            ),
            "binarization_value": float(
                stats["binarization_value"].detach().cpu().item()
            ),
            "ambiguous_groups": float(ambiguous),
            "n_train_groups_hard": float(n_train_groups_hard),
            "n_test_groups_hard": float(n_test_groups_hard),
            "blur": float(self.blur_),
            "epsilon": float(self.blur_**self.p),
        }

    def project_threshold(
        self,
        threshold: float = 0.5,
        use_full_data: bool = True,
    ) -> HardSplitResult:
        """
        Hard projection by thresholding glacier probabilities.

        Monthly rows inherit their glacier split.
        Measurement fractions are evaluated on unique measurements.
        """
        self._check_fitted()
        if not (0.0 < threshold < 1.0):
            raise ValueError("threshold must be in (0, 1).")

        scores = self.get_group_probabilities()
        test_mask = scores > threshold

        if test_mask.sum() == 0 or test_mask.sum() == self.n_groups_:
            raise ValueError(
                "Threshold projection produced an empty train or test set. "
                "Use a different threshold."
            )

        metrics = self._evaluate_hard_split(
            test_group_mask=test_mask,
            use_full_data=use_full_data,
        )

        test_groups = self.unique_groups_[test_mask]
        train_groups = self.unique_groups_[~test_mask]

        # Monthly-row indices
        monthly_test_mask = test_mask[self.group_inverse_]
        test_indices = np.flatnonzero(monthly_test_mask)
        train_indices = np.flatnonzero(~monthly_test_mask)

        # Unique-measurement indices
        measurement_test_mask = test_mask[self.measurement_group_inverse_]
        test_measurement_indices = np.flatnonzero(measurement_test_mask)
        train_measurement_indices = np.flatnonzero(~measurement_test_mask)

        return HardSplitResult(
            train_indices=train_indices,
            test_indices=test_indices,
            train_measurement_indices=train_measurement_indices,
            test_measurement_indices=test_measurement_indices,
            train_groups=train_groups,
            test_groups=test_groups,
            test_group_mask=test_mask,
            scores_used=scores,
            threshold=float(threshold),
            actual_n_test_groups=int(test_mask.sum()),
            actual_n_test_measurements=int(metrics["n_test_measurements"]),
            actual_n_test_monthly_rows=int(metrics["n_test_rows"]),
            glacier_fraction_test=float(metrics["glacier_fraction_test"]),
            measurement_fraction_test=float(metrics["measurement_fraction_test"]),
            sinkhorn_divergence=float(metrics["sinkhorn_divergence"]),
            objective=float(metrics["objective"]),
        )

    def project_by_measurement_target(
        self,
        target_fraction_measurements: Optional[float] = None,
        use_full_data: bool = True,
    ) -> HardSplitResult:
        """
        Hard projection by ranking glaciers with their soft score and adding them
        to the test set until the target number of measurements is reached as closely
        as possible.

        Parameters
        ----------
        target_fraction_measurements : float, optional
            Target fraction of unique measurements in test.
            If None, uses self.test_size_observations.

        use_full_data : bool, default=True
            If True, evaluate the hard split on the full dataset.
            Otherwise use the optimization subset for the Sinkhorn evaluation.

        Returns
        -------
        HardSplitResult
        """
        self._check_fitted()

        if target_fraction_measurements is None:
            target_fraction_measurements = self.test_size_observations

        if not (0.0 < target_fraction_measurements < 1.0):
            raise ValueError("target_fraction_measurements must be in (0, 1).")

        scores = self.get_group_probabilities()
        n_groups = int(self.n_groups_)
        n_measurements = int(self.n_measurements_)

        target_n_test_measurements = int(
            round(target_fraction_measurements * n_measurements)
        )
        target_n_test_measurements = int(
            np.clip(target_n_test_measurements, 1, n_measurements - 1)
        )

        # Number of unique measurements attached to each glacier
        group_measurement_counts = self.group_measurement_counts_.astype(int)

        # Sort glaciers by descending score, with deterministic tie-breaking by group index
        order = np.lexsort((np.arange(n_groups), -scores))

        cumulative_measurements = 0
        best_k = None
        best_diff = None

        # Add glaciers one by one in ranked order, keeping the prefix
        for k in range(1, n_groups):
            cumulative_measurements += group_measurement_counts[order[k - 1]]
            diff = abs(cumulative_measurements - target_n_test_measurements)

            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_k = k

        if best_k is None:
            raise RuntimeError("Failed to build a projected test set.")

        selected_groups = order[:best_k]

        test_mask = np.zeros(n_groups, dtype=bool)
        test_mask[selected_groups] = True

        metrics = self._evaluate_hard_split(
            test_group_mask=test_mask,
            use_full_data=use_full_data,
        )

        test_groups = self.unique_groups_[test_mask]
        train_groups = self.unique_groups_[~test_mask]

        # Monthly-row indices
        monthly_test_mask = test_mask[self.group_inverse_]
        test_indices = np.flatnonzero(monthly_test_mask)
        train_indices = np.flatnonzero(~monthly_test_mask)

        # Measurement indices
        measurement_test_mask = test_mask[self.measurement_group_inverse_]
        test_measurement_indices = np.flatnonzero(measurement_test_mask)
        train_measurement_indices = np.flatnonzero(~measurement_test_mask)

        return HardSplitResult(
            train_indices=train_indices,
            test_indices=test_indices,
            train_measurement_indices=train_measurement_indices,
            test_measurement_indices=test_measurement_indices,
            train_groups=train_groups,
            test_groups=test_groups,
            test_group_mask=test_mask,
            scores_used=scores,
            threshold=None,
            actual_n_test_groups=int(test_mask.sum()),
            actual_n_test_measurements=int(metrics["n_test_measurements"]),
            actual_n_test_monthly_rows=int(metrics["n_test_rows"]),
            glacier_fraction_test=float(metrics["glacier_fraction_test"]),
            measurement_fraction_test=float(metrics["measurement_fraction_test"]),
            sinkhorn_divergence=float(metrics["sinkhorn_divergence"]),
            objective=float(metrics["objective"]),
        )

    def _evaluate_hard_split(
        self,
        test_group_mask: np.ndarray,
        use_full_data: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate a hard split induced by a boolean test_group_mask over glacier groups.

        Sinkhorn is computed on monthly rows.
        Split size is evaluated on unique measurements.
        """
        if test_group_mask.shape != (self.n_groups_,):
            raise ValueError(f"test_group_mask must have shape ({self.n_groups_},).")
        if test_group_mask.sum() == 0 or test_group_mask.sum() == self.n_groups_:
            raise ValueError("Hard split must place at least one group in each side.")

        # Monthly rows used for Sinkhorn
        if use_full_data:
            X_np = self.X_scaled_np_
            group_inverse_rows = self.group_inverse_
        else:
            X_np = self.X_opt_.detach().cpu().numpy()
            group_inverse_rows = self.group_index_torch_.detach().cpu().numpy()

        monthly_test_mask = test_group_mask[group_inverse_rows]
        n_test_rows = int(np.sum(monthly_test_mask))
        n_train_rows = int(len(monthly_test_mask) - n_test_rows)

        if n_test_rows == 0 or n_train_rows == 0:
            raise ValueError("Hard split produced an empty train or test monthly set.")

        a_np = (~monthly_test_mask).astype(np.float64)
        b_np = monthly_test_mask.astype(np.float64)
        a_np /= a_np.sum()
        b_np /= b_np.sum()

        xt = torch.as_tensor(X_np, dtype=self.dtype, device=self.device).contiguous()
        a = torch.as_tensor(a_np, dtype=self.dtype, device=self.device).contiguous()
        b = torch.as_tensor(b_np, dtype=self.dtype, device=self.device).contiguous()

        with torch.no_grad():
            sinkhorn = self.loss_fn_(a, xt, b, xt)

        sinkhorn_val = float(sinkhorn.detach().cpu().item())

        glacier_fraction_test = float(np.mean(test_group_mask))

        # Measurement fraction is computed on unique measurements
        measurement_test_mask = test_group_mask[self.measurement_group_inverse_]
        n_test_measurements = int(np.sum(measurement_test_mask))
        measurement_fraction_test = float(
            n_test_measurements / len(measurement_test_mask)
        )

        penalty_glaciers = (
            self.lambda_glaciers
            * (glacier_fraction_test - self.test_size_glaciers) ** 2
        )
        penalty_observations = (
            self.lambda_observations
            * (measurement_fraction_test - self.test_size_observations) ** 2
        )

        objective = sinkhorn_val + penalty_glaciers + penalty_observations

        return {
            "sinkhorn_divergence": sinkhorn_val,
            "glacier_fraction_test": glacier_fraction_test,
            "measurement_fraction_test": measurement_fraction_test,
            "n_test_measurements": n_test_measurements,
            "n_test_rows": n_test_rows,
            "objective": float(objective),
        }

    def get_measurement_affectation(
        self,
        test_group_mask: np.ndarray,
        as_numpy: bool = True,
    ):
        """
        Return an affectation array of length n_unique_measurements
        with values 'train' or 'test'.
        """
        self._check_fitted()
        if test_group_mask.shape != (self.n_groups_,):
            raise ValueError(f"test_group_mask must have shape ({self.n_groups_},).")

        measurement_test_mask = test_group_mask[self.measurement_group_inverse_]
        affectation = np.where(measurement_test_mask, "test", "train")
        return affectation if as_numpy else affectation.tolist()

    # def get_monthly_affectation(
    #     self,
    #     test_group_mask: np.ndarray,
    #     as_numpy: bool = True,
    # ):
    #     """
    #     Return an affectation array of length n_monthly_rows
    #     with values 'train' or 'test'.
    #     """
    #     self._check_fitted()
    #     if test_group_mask.shape != (self.n_groups_,):
    #         raise ValueError(f"test_group_mask must have shape ({self.n_groups_},).")

    #     monthly_test_mask = test_group_mask[self.group_inverse_]
    #     affectation = np.where(monthly_test_mask, "test", "train")
    #     return affectation if as_numpy else affectation.tolist()


def sinkhorn_train_test_split(
    X_monthly,
    groups_glacier,
    measurement_ids,
    sinkhorn_backend="auto",
    test_size_observations=0.2,
):

    splitter = SinkhornGroupSplit(
        test_size_glaciers=0.2,
        test_size_observations=test_size_observations,  # interpreted as target fraction of measurements
        lambda_glaciers=0.0,
        lambda_observations=10.0,
        lambda_binary=0.1,
        optimizer_name="adam",
        lr=0.1,
        max_iter=500,
        tol_rel_obj=1e-5,
        patience=40,
        subsample_size=None,
        sinkhorn_backend=sinkhorn_backend,
        random_state=0,
        verbose=True,
        track_history=True,
    )

    # X_monthly.shape = (N_monthly_rows, N_features)
    # groups_glacier.shape = (N_monthly_rows,)
    # measurement_ids.shape = (N_monthly_rows,)
    result0 = splitter.fit(X_monthly, groups_glacier, measurement_ids)
    print("\n=== After first optimization ===")
    print(result0)

    # Continue with annealing
    result1 = splitter.continue_with_annealing(
        # lambda_binary_schedule=[0.5, 1.0, 5.0, 10.0],
        lambda_binary_schedule=[0.5, 1.0],
        adam_first=True,
        adam_iter=50,
        adam_lr=0.02,
        lbfgs_lr=0.5,
        max_iter_per_stage=60,
    )
    print("\n=== After annealing ===")
    print(result1)

    print("\nSummary:")
    print(splitter.summary())

    # Optional diagnostics
    splitter.plot_history()
    splitter.plot_group_probability_histogram()

    # hard_result = splitter.project_threshold(threshold=0.5, use_full_data=True)

    # print("\n=== Hard split after threshold projection ===")
    # print(hard_result)
    # print(f"Train monthly rows: {len(hard_result.train_indices)}")
    # print(f"Test monthly rows: {len(hard_result.test_indices)}")
    # print(f"Train measurements: {len(hard_result.train_measurement_indices)}")
    # print(f"Test measurements: {len(hard_result.test_measurement_indices)}")
    # print(f"Train glaciers: {len(hard_result.train_groups)}")
    # print(f"Test glaciers: {len(hard_result.test_groups)}")

    hard_result = splitter.project_by_measurement_target(
        target_fraction_measurements=0.2,
        use_full_data=True,
    )

    print("\n=== Hard split after measurement-target projection ===")
    print(hard_result)
    print(f"Train monthly rows: {len(hard_result.train_indices)}")
    print(f"Test monthly rows: {len(hard_result.test_indices)}")
    print(f"Train measurements: {len(hard_result.train_measurement_indices)}")
    print(f"Test measurements: {len(hard_result.test_measurement_indices)}")
    print(f"Train glaciers: {len(hard_result.train_groups)}")
    print(f"Test glaciers: {len(hard_result.test_groups)}")
    print(f"Test measurement fraction: {hard_result.measurement_fraction_test:.3f}")

    test_groups = hard_result.test_groups
    train_groups = hard_result.train_groups

    # Affectation at the measurement level
    affectation = splitter.get_measurement_affectation(hard_result.test_group_mask)

    return affectation, train_groups, test_groups


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Synthetic data with variable glacier sizes
    n_glaciers = 30
    glacier_ids = np.array([f"RGI60-11.{i:05d}" for i in range(n_glaciers)])

    X_parts = []
    groups_parts = []
    measurement_parts = []

    measurement_counter = 0

    for g, glacier_id in enumerate(glacier_ids):
        # Number of measurements for this glacier
        n_measurements_g = rng.integers(2, 12)

        # Glacier-level center in feature space
        glacier_center = rng.normal(size=4)

        for _ in range(n_measurements_g):
            measurement_id = f"M{measurement_counter:06d}"
            measurement_counter += 1

            # Number of monthly rows composing this measurement
            n_months = rng.integers(3, 25)

            # Measurement-specific center, close to the glacier center
            measurement_center = glacier_center + rng.normal(scale=0.5, size=4)

            # Monthly feature rows
            Xm = rng.normal(loc=measurement_center, scale=1.0, size=(n_months, 4))

            X_parts.append(Xm)
            groups_parts.append(np.full(n_months, glacier_id, dtype=object))
            measurement_parts.append(np.full(n_months, measurement_id, dtype=object))

    X = np.vstack(X_parts)  # monthly rows
    groups = np.concatenate(groups_parts)  # glacier ID per monthly row
    measurement_ids = np.concatenate(
        measurement_parts
    )  # measurement ID per monthly row

    print("X shape:", X.shape)
    print("Number of monthly rows:", len(X))
    print("Number of unique glaciers:", len(np.unique(groups)))
    print("Number of unique measurements:", len(np.unique(measurement_ids)))

    splitter = SinkhornGroupSplit(
        test_size_glaciers=0.2,  # kept for compatibility, but lambda_glaciers=0 so not used
        test_size_observations=0.2,  # interpreted as target fraction of MEASUREMENTS
        lambda_glaciers=0.0,
        lambda_observations=10.0,
        lambda_binary=0.0,  # first stage without annealing
        optimizer_name="adam",
        lr=0.1,
        max_iter=300,
        tol_rel_obj=1e-5,
        patience=40,
        subsample_size=None,
        sinkhorn_backend="auto",
        random_state=0,
        verbose=True,
        track_history=True,
    )

    # X.shape = (N_monthly_rows, N_features)
    # groups.shape = (N_monthly_rows,)            -> glacier IDs
    # measurement_ids.shape = (N_monthly_rows,)   -> measurement IDs
    result0 = splitter.fit(X, groups, measurement_ids)
    print("\n=== After first optimization ===")
    print(result0)

    result1 = splitter.continue_with_annealing(
        lambda_binary_schedule=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
        adam_first=True,
        adam_iter=50,
        adam_lr=0.02,
        lbfgs_lr=0.5,
        max_iter_per_stage=60,
    )
    print("\n=== After annealing ===")
    print(result1)

    print("\nSummary:")
    print(splitter.summary())

    # Optional diagnostics
    splitter.plot_history()
    splitter.plot_group_probability_histogram()

    hard_result = splitter.project_threshold(threshold=0.5, use_full_data=True)

    print("\n=== Hard split after threshold projection ===")
    print(hard_result)
    print(f"Train monthly rows: {len(hard_result.train_indices)}")
    print(f"Test monthly rows: {len(hard_result.test_indices)}")
    print(f"Train measurements: {len(hard_result.train_measurement_indices)}")
    print(f"Test measurements: {len(hard_result.test_measurement_indices)}")
    print(f"Train glaciers: {len(hard_result.train_groups)}")
    print(f"Test glaciers: {len(hard_result.test_groups)}")

    # Measurement-level affectation
    affectation_measurements = splitter.get_measurement_affectation(
        hard_result.test_group_mask
    )
    measurement_ids_unique = splitter.unique_measurements_

    print("\n=== First 10 measurement assignments ===")
    for mid, split in zip(measurement_ids_unique[:10], affectation_measurements[:10]):
        print(mid, split)

    # Optional: monthly-row affectation
    affectation_monthly = splitter.get_monthly_affectation(hard_result.test_group_mask)

    print("\n=== Counts ===")
    print(
        "Monthly train/test:",
        np.sum(affectation_monthly == "train"),
        np.sum(affectation_monthly == "test"),
    )
    print(
        "Measurement train/test:",
        np.sum(affectation_measurements == "train"),
        np.sum(affectation_measurements == "test"),
    )

    # rng = np.random.default_rng(42)

    # # Synthetic data with variable glacier sizes
    # n_glaciers = 30
    # glacier_sizes = rng.integers(5, 120, size=n_glaciers)

    # X_parts = []
    # g_parts = []

    # for g, n in enumerate(glacier_sizes):
    #     center = rng.normal(size=4)
    #     Xg = rng.normal(loc=center, scale=1.0, size=(n, 4))
    #     X_parts.append(Xg)
    #     g_parts.append(np.full(n, g))

    # X = np.vstack(X_parts)
    # groups = np.concatenate(g_parts)
    # import pdb; pdb.set_trace()

    # splitter = SinkhornGroupSplit(
    #     test_size_glaciers=0.2,
    #     test_size_observations=0.2,
    #     lambda_glaciers=0.0,
    #     lambda_observations=10.0,
    #     lambda_binary=0.0,   # first stage without annealing
    #     optimizer_name="adam",
    #     lr=0.1,
    #     max_iter=300,
    #     tol_rel_obj=1e-5,
    #     patience=40,
    #     subsample_size=None,
    #     sinkhorn_backend="auto",
    #     random_state=0,
    #     verbose=True,
    #     track_history=True,
    # )

    # # X.shape = (Nsamples, Nfeatures)
    # # groups.shape = (Nsamples,)
    # result0 = splitter.fit(X, groups)
    # print("\n=== After first optimization ===")
    # print(result0)

    # # Continue with annealing
    # # result1 = splitter.continue_with_annealing(
    # #     lambda_binary_schedule=[0.01, 0.05, 0.1, 0.5, 1.0],
    # #     max_iter_per_stage=150,
    # # )
    # # result1 = splitter.continue_with_annealing(
    # #     lambda_binary_schedule=[0.01],
    # #     adam_first=True,
    # #     adam_iter=50,
    # #     adam_lr=0.02,
    # #     lbfgs_lr=0.5,
    # #     max_iter_per_stage=100,
    # # )
    # result1 = splitter.continue_with_annealing(
    #     lambda_binary_schedule=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    #     adam_first=True,
    #     adam_iter=50,
    #     adam_lr=0.02,
    #     lbfgs_lr=0.5,
    #     max_iter_per_stage=60,
    # )
    # print("\n=== After annealing ===")
    # print(result1)

    # print("\nSummary:")
    # print(splitter.summary())

    # # Optional diagnostics
    # splitter.plot_history()
    # splitter.plot_group_probability_histogram()

    # hard_result = splitter.project_threshold(threshold=0.5, use_full_data=True)

    # print("\n=== Hard split after threshold projection ===")
    # print(hard_result)
    # print(f"Train observations: {len(hard_result.train_indices)}")
    # print(f"Test observations: {len(hard_result.test_indices)}")
    # print(f"Train glaciers: {len(hard_result.train_groups)}")
    # print(f"Test glaciers: {len(hard_result.test_groups)}")
