from jax import Array
from jax import numpy as jnp

from src.covariance_update_functions.covariance_update_function import (
    CovarianceUpdateFunction,
    CovarianceUpdateFunctionBuilder,
)
from src.utils import sqrt_L_sum_qr


class OuterCovarianceUpdate(CovarianceUpdateFunctionBuilder):
    """Outer covariance update function."""

    def __init__(self, scale: float = 1.0) -> None:
        """
        Initializes covariance update function.

        Args:
            scale (float, optional): Scale factor for local error estimate. Defaults to 1.0.
        """

        self.scale = scale

    def build(self) -> CovarianceUpdateFunction:
        def cov_update(cov: Array, eps: Array) -> Array:
            """
            Computes covariance update.
            D: Latent dimension.
            N: ODE order.

            Args:
                cov (Array): Covariance [N*D, N*D].
                eps (Array): Local error estimate [N*D].

            Returns:
                Array: Updated covariance [N*D, N*D].
            """

            scaled_eps = self.scale * eps
            return cov + jnp.outer(scaled_eps, scaled_eps)

        return cov_update

    def build_sqrt(self) -> CovarianceUpdateFunction:
        def cov_update_sqrt(cov_sqrt: Array, eps: Array) -> Array:
            """
            Computes covariance square-root update.

            Args:
                cov_sqrt (Array): Covariance square-root [N*D, N*D].
                eps (Array): Local error estimate [N*D].

            Returns:
                Array: Updated covariance square-root [N*D, N*D].
            """

            scaled_eps = self.scale * eps
            Q_sqrt = jnp.outer(scaled_eps, scaled_eps) / (scaled_eps @ scaled_eps) ** 0.5

            return sqrt_L_sum_qr(cov_sqrt, Q_sqrt)

        return cov_update_sqrt
