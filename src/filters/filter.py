from typing import Tuple

from jax import Array

from src.filters.sigma_fns import SigmaFn
from src.solvers.rksolver import RKSolver


class Filter:
    """Abstract base class for Gaussian filters, used for ODE solving."""

    def __init__(self, rk_solver: RKSolver, P0: Array, sigma_fn: SigmaFn) -> None:
        """
        Initializes filter.
        D: Latent dimension.
        N: ODE order.

        Args:
            rk_solver (RKSolver): RK solver.
            P0 (Array): Initial covariance [N*D, N*D].
            sigma_fn (SigmaFn): Sigma function.
        """

        self.rk_solver = rk_solver
        self.t = rk_solver.t0
        self.m = rk_solver.x0
        self._P = P0
        self.sigma_fn = sigma_fn

    def predict(self) -> Tuple[Array, Array, Array]:
        """
        Predicts state after performing one step of the ODE solver.
        D: Latent dimension.
        N: ODE order.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete filter.

        Returns:
            Tuple[Array, Array, Array]: Time [1], mean state [1, N, D], covariance [N*D, N*D].
        """

        raise NotImplementedError

    @property
    def P(self) -> Array:
        """
        Covariance getter.
        D: Latent dimension.
        N: ODE order.

        Returns:
            Array: Covariance [N*D, N*D].
        """

        return self._P

    @P.setter
    def P(self, value: Array) -> None:
        """
        Covariance setter.
        D: Latent dimension.
        N: ODE order.

        Args:
            value (Array): Covariance [N*D, N*D].
        """

        self._P = value
