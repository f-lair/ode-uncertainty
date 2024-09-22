from functools import partial
from typing import Callable, Tuple

import jax
from jax import Array, lax
from jax import numpy as jnp
from jax import random
from jax import scipy as jsp
from tensorflow_probability.substrates import jax as tfp

from src.filters.deprecated.ekf import EKF
from src.filters.sigma_fns import SigmaFn
from src.ode.ode import ODE
from src.solvers.rksolver import RKSolver
from src.utils import bmmT, jeffrey_divergence_sqrt, sqrt_L_sum_qr


@jax.jit
def compute_merge_mapping_step(
    state: Tuple[Array, Array, Array], x: None
) -> Tuple[Tuple[Array, Array, Array], Array]:
    """
    Performs one step in greedy merge mapping computation.
    M: Mixture components.

    Args:
        state (Tuple[Array, Array, Array]): Distances [M, M], previous merge indices [2], previous
            number of valid merge pairs [].
        x (None): Not used.

    Returns:
        Tuple[Tuple[Array, Array, Array], Array]: Updated distances [M, M], new merge indices [2],
            new number of valid merge pairs [], new merge indices [2].
    """

    distances, prev_indices, num_valid_pairs = state
    merge_indices = jnp.stack(jnp.unravel_index(jnp.argmin(distances), distances.shape))  # [2]
    valid_indices = jnp.isfinite(distances[merge_indices[0], merge_indices[1]])  # []
    merge_indices = jnp.where(valid_indices[None], merge_indices, prev_indices)  # [2]

    cross_indices_1 = jnp.tile(merge_indices, distances.shape[0])  # [2*M]
    cross_indices_2 = jnp.repeat(
        jnp.arange(0, distances.shape[0], dtype=merge_indices.dtype), 2
    )  # [2*M]
    cross_indices_row = jnp.concatenate([cross_indices_1, cross_indices_2])  # [4*M]
    cross_indices_col = jnp.concatenate([cross_indices_2, cross_indices_1])  # [4*M]

    distances_reduced = distances.at[cross_indices_row, cross_indices_col].set(jnp.inf)
    return (distances_reduced, merge_indices, num_valid_pairs + valid_indices), merge_indices


@jax.jit
def compute_merge_mapping(
    merge_threshold: float, m: Array, S: Array, merge_flags: Array
) -> Tuple[Array, Array, Array]:
    """
    Computes merge mapping using greedy algorithm, successively mapping unmapped Gaussians with
    minimal Jeffrey divergence.
    D: Latent dimension.
    M: Mixture components.
    N: ODE order.

    Args:
        merge_threshold (float): Jeffrey divergence threshold to be valid for mapping.
        m (Array): Gaussian means [M, N*D].
        S (Array): Gaussian square-root covariances [M, N*D, N*D].
        merge_flags (Array): Flags indicating Gaussians valid for mapping [M].

    Returns:
        Tuple[Array, Array, Array]: Merge mapping [M, M'], flags indicating Gaussians not mapped
            [M], number of valid mappings [].
    """

    merge_condition = jnp.logical_and(merge_flags[None, :], merge_flags[:, None])
    j_distances = jeffrey_divergence_sqrt(
        m[:, None, :], m[None, :, :], S[:, None, :, :], S[None, :, :, :]
    )  # [M, M]
    j_distances = jnp.fill_diagonal(j_distances, jnp.inf, inplace=False)  # [M, M]
    j_distances = jnp.where(
        jnp.logical_and(j_distances < merge_threshold, merge_condition), j_distances, jnp.inf
    )  # [M, M]
    (_, _, num_valid_pairs), merge_indices = lax.scan(
        compute_merge_mapping_step,
        (
            j_distances,
            jnp.full((2,), m.shape[0], dtype=jnp.int64),
            jnp.zeros((), dtype=jnp.int64),
        ),
        length=m.shape[0] // 2,
    )  # [M', 2]
    merge_mapping = jnp.zeros((m.shape[0], merge_indices.shape[0]))  # [M, M']
    merge_mapping = merge_mapping.at[
        merge_indices.ravel(),
        jnp.repeat(jnp.arange(0, merge_indices.shape[0], dtype=merge_indices.dtype), 2),
    ].set(
        1.0
    )  # [M, M']
    keep_flags = jnp.all(merge_mapping == 0, axis=1)  # [M]

    return (
        merge_mapping,
        keep_flags,
        num_valid_pairs,
    )


@jax.jit
def merge_refit(m: Array, S: Array, w: Array, merge_mapping: Array) -> Tuple[Array, Array, Array]:
    """
    Merges pairs of Gaussians in a mixture model using a pre-computed mapping.
    D: Latent dimension.
    M: Mixture components.
    N: ODE order.

    Args:
        m (Array): Gaussian means [M, N*D].
        S (Array): Gaussian square-root covariances [M, N*D, N*D].
        w (Array): Mixture weights [M].
        merge_mapping (Array): Merge mapping [M, M'].

    Returns:
        Tuple[Array, Array, Array]: Merged means [M', N*D], merged square-root covariances
            [M', N*D, N*D], merged mixture weights [M'].
    """

    merge_mapping_inv = merge_mapping.T  # [M', M]
    w_merged = merge_mapping_inv @ w  # [M']

    m_merged = 1 / w_merged[:, None] * ((w[None, :] * merge_mapping_inv) @ m)  # [M', N*D]

    m_delta = m[None, :, :] - m_merged[:, None, :]  # [M', M, N*D]
    S_updates = jnp.broadcast_to(
        S[None, :, :, :], merge_mapping_inv.shape + S.shape[-2:]
    )  # [M', M, N*D, N*D]
    S_updates = jnp.sqrt(w)[None, :, None, None] * tfp.math.cholesky_update(
        S_updates, m_delta, jnp.ones_like(merge_mapping_inv)
    )  # [M', M, N*D, N*D]
    merge_indices = jnp.nonzero(
        merge_mapping_inv,
        size=2 * merge_mapping_inv.shape[0],
        fill_value=merge_mapping_inv.shape[1],
    )  # [2*M'], [2*M']
    S_merged = (
        S_updates.at[merge_indices[0], merge_indices[1]]
        .get(indices_are_sorted=True, mode="fill", fill_value=0.0)
        .reshape(S_updates.shape[0], 2, S_updates.shape[2], S_updates.shape[3])
    )  # [M', 2, N*D, N*D]
    S_merged = w_merged[:, None, None] ** (-1 / 2) * sqrt_L_sum_qr(
        S_merged[:, 0], S_merged[:, 1]
    )  # [M', N*D, N*D]

    return m_merged, S_merged, w_merged


@partial(jax.jit, static_argnums=[0, 1])
def step_jit(
    step_fn: Callable[[Array, Array], Tuple[Array, Array, Array, Array]],
    sigma_sqrt_fn_vmap: Callable[[Array], Array],
    t: Array,
    m: Array,
    S: Array,
) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Jitted step function of GMM-EKF.
    D: Latent dimension.
    M: Mixture components.
    N: ODE order.

    Args:
        step_fn (Callable[[Array, Array], Tuple[Array, Array, Array, Array]]): RK-solver step
            function.
        sigma_sqrt_fn_vmap (SigmaFn): Vectorized Sigma-sqrt function.
        t (Array): Time [M].
        m (Array): Mean state [M, N, D].
        S (Array): Covariance square-root [M, N*D, N*D].

    Returns:
        Tuple[Array, Array, Array, Array, Array]: Time [1], mean state [M, N, D],
            covariance square-root [M, N*D, N*D], Jacobian [M, N*D, N*D],
            sigma square-root [M, N*D, N*D].
    """

    t_next, m_next, eps, _ = step_fn(t, m)  # [M], [M, N, D], [M, N, D]
    jac = jax.jacfwd(partial(EKF._rk_solver_step_AD, step_fn, t))(m)  # [M, N, D, M, N, D]
    jac = jnp.diagonal(jac, axis1=0, axis2=3)  # [N, D, N, D, M]
    jac = jnp.transpose(jac, (4, 0, 1, 2, 3))  # [M, N, D, N, D]

    M, N, D = jac.shape[:3]
    jac = jnp.reshape(jac, (M, N * D, N * D))  # [M, N*D, N*D]

    sigma_sqrt = sigma_sqrt_fn_vmap(eps.reshape(M, N * D))  # [M, N*D, N*D]
    S_next = sqrt_L_sum_qr(jnp.einsum("bij,bjk->bik", jac, S), sigma_sqrt)  # [M, N*D, N*D]

    return t_next[0:1], m_next, S_next, jac, sigma_sqrt


@jax.jit
def invalidate_by_distance(m: Array, valid_flags: Array, m_distance_threshold: float) -> Array:
    """
    Invalidates Gaussians whose means are too far off in terms of absolute distance from all other
    means, in any dimension.
    D: Latent dimension.
    M: Mixture components.
    N: ODE order.

    Args:
        m (Array): Gaussian means [M, N, D].
        valid_flags (Array): Prior flags indicating validity of Gaussians [M].
        m_distance_threshold (float): Distance threshold.

    Returns:
        Array: Updated validity flags [M].
    """

    m_r = m.reshape(m.shape[0], m.shape[1] * m.shape[2])  # [M, N*D]
    m_r_delta = jnp.abs(m_r[None, :, :] - m_r[:, None, :])  # [M, M, N*D]

    diag_indices_1 = jnp.tile(
        jnp.stack(jnp.diag_indices(m_r_delta.shape[0]), axis=0), (1, m_r_delta.shape[-1])
    )  # [2, M*N*D]
    diag_indices_2 = jnp.repeat(
        jnp.arange(m_r_delta.shape[-1], dtype=diag_indices_1.dtype), m_r_delta.shape[0]
    )  # [M*N*D]
    m_r_delta = m_r_delta.at[diag_indices_1[0], diag_indices_1[1], diag_indices_2].set(
        jnp.inf
    )  # [M, M, N*D]

    invalid_flags = jnp.all(jnp.any(m_r_delta > m_distance_threshold, axis=2), axis=1)  # [M]

    return jnp.logical_and(jnp.logical_not(invalid_flags), valid_flags)


@partial(jax.jit, static_argnums=[0, 1])
def estimate_nl(
    step_fn: Callable[[Array, Array], Tuple[Array, Array, Array, Array]],
    ode: ODE,
    t: Array,
    m: Array,
    valid_flags: Array,
    nl_threshold: float,
) -> Tuple[Array, Array]:
    """
    Estimates upcoming degree of nonlinearity using look-ahead step.
    D: Latent dimension.
    M: Mixture components.
    N: ODE order.

    Args:
        step_fn (Callable[[Array, Array], Tuple[Array, Array, Array, Array]]): RK-solver step
            function.
        ode (ODE): ODE.
        t (Array): Time [1].
        m (Array): Mean state [M, N, D].
        valid_flags (Array): Flags indicating validity of Gaussians [M].
        nl_threshold (float): Nonlinearity threshold.

    Returns:
        Tuple[Array, Array]: Indices of Gaussians sorted descendingly w.r.t degree of nonlinearity,
            number of Gaussians above nonlinearity threshold.
    """

    # Time derivative
    dx_dt = ode(t, m)  # [M, N, D]

    # Look-ahead step
    t_next, m_next, _, h = step_fn(t, m)  # [M], [M, N, D], [M, N, D], [M]
    dx_dt_next = ode(t_next, m_next)  # [M, N, D]

    # Difference quotient -> Second time derivative
    d2x_dt2 = (dx_dt_next[:, 0, :] - dx_dt[:, 0, :]) / h[:, None]  # [M, D]

    # L2 norm over D
    nl_estimate = jnp.linalg.norm(d2x_dt2, ord=2, axis=1)  # [M]

    # Mask out invalid mixtures
    nl_estimate = jnp.where(valid_flags, nl_estimate, -jnp.inf)  # [M]

    return (
        jnp.argsort(nl_estimate)[::-1],  # type: ignore
        jnp.count_nonzero((nl_estimate > nl_threshold)),  # type: ignore
    )


@jax.jit
def merge_2(
    m: Array,
    S: Array,
    w: Array,
    merge_flags: Array,
    merge_threshold: float,
) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Merges close Gaussians in pairs of two.
    D: Latent dimension.
    M: Mixture components.
    N: ODE order.

    Args:
        m (Array): Mean state [M, N, D].
        S (Array): Covariance square-root [M, N*D, N*D].
        w (Array): Mixture weights [M].
        merge_flags (Array): Flags indicating validity of Gaussians for being merged [M].
        merge_threshold (float): Jeffrey divergence threshold for merging.

    Returns:
        Tuple[Array, Array, Array, Array, Array]: Merged mean state [M', N, D], merged covariance
            square-root [M', N*D, N*D], merged mixture weights [M'], flags indicating non-merged
            Gaussians [M], number of merges [].
    """

    m_r = m.reshape(m.shape[0], S.shape[-1])

    merge_mapping, keep_flags, num_valid_pairs = compute_merge_mapping(
        merge_threshold, m_r, S, merge_flags
    )  # [M, M'], [M], []

    m_merged, S_merged, w_merged = merge_refit(
        m_r, S, w, merge_mapping
    )  # [M', N*D], [M', N*D, N*D], [M']
    m_merged = m_merged.reshape(w_merged.shape[0], m.shape[1], m.shape[2])

    return m_merged, S_merged, w_merged, keep_flags, num_valid_pairs


@jax.jit
def split_2(
    m: Array,
    S: Array,
    w: Array,
    split_displacement: float,
) -> Tuple[Array, Array, Array]:
    """
    Splits Gaussians along their covariances' eigenvector corresponding to the largest eigenvalue.

    Args:
        m (Array): Mean state [M, N, D].
        S (Array): Covariance square-root [M, N*D, N*D].
        w (Array): Mixture weights [M].
        split_displacement (float): Split displacement.

    Returns:
        Tuple[Array, Array, Array]: Split mean state [2*M, N, D], split covariance square-root
            [2*M, N*D, N*D], split mixture weights [2*M].
    """

    # Eigendecomposition: only largest eigenvalue lambda_ -> direction d most susceptible to
    # nonlinearity
    eigvals, eigvecs = jnp.linalg.eigh(bmmT(S, S))
    lambda_ = eigvals[:, -1]  # [M]
    lambda_sqrt = jnp.sqrt(lambda_)[:, None, None]  # [M, 1, 1]  # type: ignore
    d = eigvecs[:, :, -1]  # [M, N*D]
    d_r = d.reshape(*m.shape)  # [M, N, D]

    # Split means
    m_split_1 = m + split_displacement * lambda_sqrt * d_r  # [M, N, D]
    m_split_2 = m - split_displacement * lambda_sqrt * d_r  # [M, N, D]
    m_split = jnp.concatenate([m_split_1, m_split_2], axis=0)  # [2*M, N, D]

    # Split covariance square-roots
    # Can be done efficiently using rank-1 Cholesky downdate (take care of zero cov!)
    S_zero = jnp.all(jnp.abs(S) < 1e-6, axis=(-1, -2), keepdims=True)
    S_split = jnp.where(
        S_zero, 0.0, tfp.math.cholesky_update(S, d, -(split_displacement**2) * lambda_)
    )  # [M, N*D, N*D]

    S_split = jnp.concatenate([S_split, S_split], axis=0)  # [2*M, N*D, N*D]  # type: ignore

    # Split weights
    w_split = 0.5 * jnp.concatenate([w, w], axis=0)  # [2*M]

    return m_split, S_split, w_split


class GMM_EKF(EKF):
    """Gaussian Mixture Model Extended Kalman Filter."""

    def __init__(
        self,
        max_components: int = 8,
        m_distance_threshold: float = 100.0,
        nl_threshold: float = 0.1,
        merge_threshold: float = 10.0,
        split_displacement: float = 0.5,
        min_w: float = 0.01,
    ) -> None:
        """
        Initializes filter.

        Args:
            max_components (int, optional): Maximum number of mixture components. Defaults to 8.
            m_distance_threshold (float, optional): Distance threshold used for invalidation.
                Defaults to 100.0.
            nl_threshold (float, optional): Degree of nonlinearity threshold used to decide between
                splitting and merging. Defaults to 0.1.
            merge_threshold (float, optional): Jeffrey divergence threshold for merging. Defaults
                to 10.0.
            split_displacement (float, optional): Split displacement. Defaults to 0.5.
            min_w (float, optional): Minimum mixture weight (otherwise pruned). Defaults to 0.01.
        """

        self.max_components = max_components
        self.m_distance_threshold = m_distance_threshold
        self.nl_threshold = nl_threshold
        self.merge_threshold = merge_threshold
        self.split_displacement = split_displacement
        self.min_w = min_w

    def setup(self, rk_solver: RKSolver, P0: Array, sigma_fn: SigmaFn) -> None:
        """
        Setups filter.
        D: Latent dimension.
        N: ODE order.

        Args:
            rk_solver (RKSolver): RK solver.
            P0 (Array): Initial covariance [N*D, N*D].
            sigma_fn (SigmaFn): Sigma function.
        """

        self.rk_solver = rk_solver
        self.t = rk_solver.t0
        self.m = jnp.broadcast_to(rk_solver.x0, (self.max_components,) + rk_solver.x0.shape[1:])
        self._P = P0
        self.w = jnp.concatenate([jnp.ones((1,)), jnp.zeros((self.max_components - 1))])
        self.sigma_sqrt_fn_vmap = jax.vmap(sigma_fn.sqrt)  # type: ignore
        self.S = jnp.nan_to_num(jsp.linalg.cholesky(self._P, lower=True))  # type: ignore
        self.S = jnp.broadcast_to(self.S, (self.max_components,) + self.S.shape[1:])
        self.M_valid = jnp.ones((), dtype=jnp.int64)

    def batch_dim(self) -> int:
        """
        Batch dimension.

        Returns:
            int: Batch dimension.
        """

        return self.max_components

    @property
    def P(self) -> Array:
        """
        Covariance getter.
        D: Latent dimension.
        M: Mixture components.
        N: ODE order.

        Returns:
            Array: Covariance [M, N*D, N*D].
        """

        return bmmT(self.S, self.S)

    @staticmethod
    @partial(jax.jit, static_argnums=[0, 1, 2])
    def _predict_jit(
        rk_solver_step: Callable,
        sigma_sqrt_fn: Callable,
        ode: ODE,
        t: Array,
        m: Array,
        S: Array,
        w: Array,
        M_valid: Array,
        m_distance_threshold: float,
        nl_threshold: float,
        merge_threshold: float,
        split_displacement: float,
        min_w: float,
    ) -> Tuple[Array, Array, Array, Array, Array]:
        """
        Predicts state after performing one step of the ODE solver.
        D: Latent dimension.
        M: Mixture components.
        N: ODE order.

        Args:
            rk_solver_step (Callable): RK-solver step function.
            sigma_sqrt_fn (Callable): Sigma-sqrt function.
            ode (ODE): ODE.
            t (Array): Time [1].
            m (Array): Mean state [M, N, D].
            S (Array): Covariance square-root [M, N*D, N*D]-
            w (Array): Mixture weights [M].
            M_valid (Array): Number of mixtures [].
            m_distance_threshold (float): Distance threshold used for invalidation.
            nl_threshold (float): Degree of nonlinearity threshold used to decide between splitting
                and merging.
            merge_threshold (float): Jeffrey divergence threshold for merging.
            split_displacement (float): Split displacement.
            min_w (float): Minimum mixture weight (otherwise pruned).

        Returns:
            Tuple[Array, Array, Array, Array, Array]: Time [], mean state [M, N, D],
                covariance square-root [M, N*D, N*D], mixture weights [M], number of mixtures [].
        """

        # Determine valid mixtures by counting (input is padded to the right)
        M = m.shape[0]
        M_indices = jnp.arange(M, dtype=M_valid.dtype)  # [M]
        valid_flags = jnp.logical_and(M_indices < M_valid, w >= min_w)  # [M]

        # Perform solver step
        t_next, m_next, S_next, _, _ = step_jit(
            rk_solver_step,
            sigma_sqrt_fn,
            t,
            m,
            S,
        )  # [M], [M, N, D], [M, N*D, N*D]

        # Invalidate NaN/inf/too far off mixtures
        valid_flags = jnp.logical_and(
            jnp.all(jnp.isfinite(m_next), axis=(-1, -2)), valid_flags
        )  # [M]
        valid_flags = invalidate_by_distance(m_next, valid_flags, m_distance_threshold)  # [M]

        # Renormalize mixture weights
        w_next = w / (valid_flags * w).sum()  # [M]
        M_valid = jnp.count_nonzero(valid_flags)  # []

        # Estimate upcoming degree of nonlinearity
        split_indices, num_splits = estimate_nl(
            rk_solver_step,
            ode,
            t_next,
            m_next,
            valid_flags,
            nl_threshold,
        )  # [M], [], [M, N, D]

        # Determine mixtures valid for merging, then merge
        non_merge_indices = jnp.where(M_indices < num_splits, split_indices, M)  # [M]
        merge_flags = valid_flags.at[non_merge_indices].set(False)  # [M]
        m_merged, S_merged, w_merged, keep_flags, num_valid_pairs = merge_2(
            m_next, S_next, w_next, merge_flags, merge_threshold
        )  # [M', N, D], [M', N*D, N*D], [M']
        # Keep track of non-merged, kept mixtures
        keep_flags = jnp.logical_and(merge_flags, keep_flags)  # [M]
        merge_flags = jnp.logical_and(merge_flags, jnp.logical_not(keep_flags))  # [M]
        merged_flags = (
            jnp.arange(m_merged.shape[0], dtype=split_indices.dtype) < num_valid_pairs
        )  # [M']
        num_kept = jnp.count_nonzero(keep_flags)  # []

        # Determine mixtures than can be split without exceeding the max number of mixtures, then
        # split
        num_splits_valid = jnp.minimum(
            M - num_valid_pairs - num_kept - num_splits, num_splits
        )  # []
        non_split_indices = jnp.where(M_indices >= num_splits_valid, split_indices, M)  # [M]
        split_flags_valid = valid_flags.at[non_split_indices].set(False)  # [M]
        m_split, S_split, w_split = split_2(
            m_next,
            S_next,
            w_next,
            split_displacement,
        )  # [2*M, N, D], [2*M, N*D, N*D], [2*M]

        # Invalidate too far off mixtures
        split_flags_valid = jnp.all(
            invalidate_by_distance(
                m_split,
                jnp.concatenate([split_flags_valid, split_flags_valid]),
                m_distance_threshold,
            ).reshape(2, -1),
            axis=0,
        )  # [M]

        # Determine remaining mixtures to be kept
        non_keep_indices = jnp.where(
            jnp.logical_or(M_indices >= num_splits, M_indices < num_splits_valid), split_indices, M
        )  # [M]
        keep_flags = jnp.logical_or(
            valid_flags.at[non_keep_indices].set(False),
            keep_flags,
        )  # [M]

        # Build combined arrays, containing all merged, split and kept mixtures (including invalid)
        m_next = jnp.concatenate([m_merged, m_split, m_next], axis=0)  # [3*M+M', N, D]
        S_next = jnp.concatenate([S_merged, S_split, S_next], axis=0)  # [3*M+M', N*D, N*D]
        w_next = jnp.concatenate([w_merged, w_split, w_next], axis=0)  # [3*M+M']
        # Combine arrays of flags indicating their validity accordingly, then take the allowed
        # maximum of indices
        valid_flags_next = jnp.concatenate(
            [merged_flags, split_flags_valid, split_flags_valid, keep_flags]
        )  # [3*M+M']
        valid_indices_next = jnp.flatnonzero(valid_flags_next, size=M, fill_value=m_next.shape[0])

        # Select valid mixtures using these indices, pad by 0.
        m_next = m_next.at[valid_indices_next].get(
            indices_are_sorted=True, mode="fill", fill_value=0.0
        )  # [M, N, D]
        S_next = S_next.at[valid_indices_next].get(
            indices_are_sorted=True, mode="fill", fill_value=0.0
        )  # [M, N*D, N*D]
        w_next = w_next.at[valid_indices_next].get(
            indices_are_sorted=True, mode="fill", fill_value=0.0
        )  # [M]
        M_valid = jnp.count_nonzero(w_next)  # []

        return t_next, m_next, S_next, w_next, M_valid

    def _predict(self) -> Tuple[Array, Array, Array, Array, Array, Array]:
        """
        Predicts state after performing one step of the ODE solver.
        D: Latent dimension.
        M: Mixture components.
        N: ODE order.

        Returns:
            Tuple[Array, Array, Array, Array, Array, Array]: Time [1], mean state [M, N, D],
                covariance [M, N*D, N*D], mean state derivative [1, N, D], mixture weights [M],
                number of mixtures [].
        """

        dx_dts = self.rk_solver.fn(self.t, self.m)
        self.t, self.m, self.S, self.w, self.M_valid = self._predict_jit(
            self.rk_solver.step,
            self.sigma_sqrt_fn_vmap,
            self.rk_solver.fn,
            self.t,
            self.m,
            self.S,
            self.w,
            self.M_valid,
            self.m_distance_threshold,
            self.nl_threshold,
            self.merge_threshold,
            self.split_displacement,
            self.min_w,
        )

        return self.t, self.m, self.P, dx_dts, self.w, self.M_valid

    @staticmethod
    def results_spec_predict() -> Tuple[str, ...]:
        """
        Results specification.

        Returns:
            Tuple[str, ...]: Results keys.
        """

        return "ts", "xs", "Ps", "dx_dts", "ws", "Ms"
