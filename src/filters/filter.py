from typing import Dict, Tuple

from jax import Array

from src.filters.sigma_fns import SigmaFn
from src.solvers.rksolver import RKSolver


class Filter:
    """Abstract base class for Gaussian filters, used for ODE solving."""

    def setup(self, rk_solver: RKSolver, P0: Array, sigma_fn: SigmaFn) -> None:
        """
        Setups filter.
        D: Latent dimension.
        N: ODE order.

        Args:
            rk_solver (RKSolver): RK solver.
            P0 (Array): Initial covariance [1, N*D, N*D].
            sigma_fn (SigmaFn): Sigma function.
        """

        self.rk_solver = rk_solver
        self.t = rk_solver.t0
        self.m = rk_solver.x0
        self._P = P0
        self.sigma_fn = sigma_fn

    def _predict(self) -> Tuple[Array, ...]:
        """
        Predicts state after performing one step of the ODE solver.
        D: Latent dimension.
        N: ODE order.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete filter.

        Returns:
            Tuple[Array, ...]: Results data according to results_spec.
        """

        raise NotImplementedError

    @staticmethod
    def results_spec() -> Tuple[str, ...]:
        """
        Results specification.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete filter.

        Returns:
            Tuple[str, ...]: Results keys.
        """

        raise NotImplementedError

    def batch_dim(self) -> int:
        """
        Batch dimension.

        Returns:
            int: Batch dimension.
        """

        return 1

    def predict(self) -> Dict[str, Array]:
        """
        Predicts state after performing one step of the ODE solver.

        Returns:
            Dict[str, Array]: Results according to results_spec.
        """

        return {key: datum for key, datum in zip(self.results_spec(), self._predict())}

    @property
    def P(self) -> Array:
        """
        Covariance getter.
        M: Batch dimension.
        D: Latent dimension.
        N: ODE order.

        Returns:
            Array: Covariance [M, N*D, N*D].
        """

        return self._P

    @P.setter
    def P(self, value: Array) -> None:
        """
        Covariance setter.
        M: Batch dimension.
        D: Latent dimension.
        N: ODE order.

        Args:
            value (Array): Covariance [M, N*D, N*D].
        """

        self._P = value
