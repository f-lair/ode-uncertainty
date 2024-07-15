from functools import partial
from typing import Callable, Tuple

import jax
from jax import Array
from jax import numpy as jnp

from src.filters.filter import Filter
from src.filters.sigma_fns import SigmaFn
from src.solvers.rksolver import RKSolver


class EKF(Filter):
    """Extended Kalman Filter."""

    def __init__(
        self,
        rk_solver: RKSolver,
        P0: Array,
        sigma_fn: SigmaFn,
        clip_gradients: float = jnp.inf,
        norm: float = 2.0,
    ) -> None:
        """
        Initializes filter.
        D: Latent dimension.
        N: ODE order.

        Args:
            rk_solver (RKSolver): RK solver.
            P0 (Array): Initial covariance [N*D, N*D].
            sigma_fn (SigmaFn): Sigma function.
            clip_gradients (float, optional): Clips gradients, if <inf. Defaults to jnp.inf.
            norm (float, optional): Norm used for gradient clipping. Defaults to 2.0.
        """

        super().__init__(rk_solver, P0, sigma_fn)
        self.sigma_fn = sigma_fn
        self.jac_buffer = []
        self.sigma_buffer = []
        self.clip_gradients = clip_gradients
        self.norm = norm

    @staticmethod
    def _rk_solver_step_AD(
        step_fn: Callable[[Array, Array], Tuple[Array, Array, Array, Array]], t: Array, x: Array
    ) -> Array:
        """
        Auxiliary function used for differentiating an RK-solver step.
        D: Latent dimension.
        N: ODE order.

        Args:
            step_fn (Callable[[Array, Array], Tuple[Array, Array, Array, Array]]):
                RK-solver step function.
            t (Array): Time [1].
            x (Array): State [1, N, D].

        Returns:
            Array: Next state [1, N, D].
        """

        return step_fn(t, x)[1]

    @staticmethod
    @partial(jax.jit, static_argnums=[0, 1])
    def _predict(
        step_fn: Callable[[Array, Array], Tuple[Array, Array, Array, Array]],
        sigma_fn: SigmaFn,
        t: Array,
        m: Array,
        P: Array,
        clip_gradients: float,
        norm: float,
    ) -> Tuple[Array, Array, Array, Array, Array]:
        """
        Jitted predict function of EKF.
        D: Latent dimension.
        N: ODE order.

        Args:
            step_fn (Callable[[Array, Array], Tuple[Array, Array, Array, Array]]):
                RK-solver step function.
            sigma_fn (SigmaFn): Sigma function.
            t (Array): Time [1].
            m (Array): Mean state [1, N, D].
            P (Array): Covariance [N*C, N*C].

        Returns:
            Tuple[Array, Array, Array, Array, Array]: Time [1], mean state [1, N, D],
                covariance [N*D, N*D], Jacobian [N*D, N*D], sigma [N*D, N*D].
        """

        t_next, m_next, eps, _ = step_fn(t, m)  # [1], [1, N, D], [1, N, D]
        jac = jax.jacfwd(partial(EKF._rk_solver_step_AD, step_fn, t))(m).reshape(
            m.size, m.size
        )  # [N*D, N*D]
        jac_norm = jnp.linalg.norm(jac, ord=norm, axis=1, keepdims=True)  # [N*D, 1]
        jac = jnp.where(
            jac_norm <= clip_gradients, jac, clip_gradients * (jac / jac_norm)
        )  # [N*D, N*D]

        sigma = sigma_fn(eps.ravel())  # [N*D, N*D]
        P_next = jac @ P @ jac.T + sigma  # [N*D, N*D]

        return t_next, m_next, P_next, jac, sigma

    def predict(self) -> Tuple[Array, Array, Array]:
        """
        Predicts state after performing one step of the ODE solver.
        D: Latent dimension.
        N: ODE order.

        Returns:
            Tuple[Array, Array, Array]: Time [1], mean state [1, N, D], covariance [N*D, N*D].
        """

        self.t, self.m, self.P, jac, sigma = self._predict(
            self.rk_solver.step, self.sigma_fn, self.t, self.m, self.P
        )
        self.jac_buffer.append(jac)
        self.sigma_buffer.append(sigma)

        return self.t, self.m, self.P
