from jax import Array
from jax import numpy as jnp

from src.covariance_update_functions.static_covariance_update_function import (
    StaticCovarianceUpdateFunction,
    StaticCovarianceUpdateFunctionBuilder,
)
from src.utils import sqrt_L_sum_qr


class StaticDiagonalCovarianceUpdate(StaticCovarianceUpdateFunctionBuilder):
    """Static diagonal covariance update function."""

    def build(self) -> StaticCovarianceUpdateFunction:
        def cov_update(static_cov: Array, cov: Array, eps: Array) -> Array:
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

            return cov + static_cov**2 * jnp.eye(cov.shape[0])

        return cov_update

    def build_sqrt(self) -> StaticCovarianceUpdateFunction:
        def cov_update_sqrt(static_cov: Array, cov_sqrt: Array, eps: Array) -> Array:
            """
            Computes covariance square-root update.

            Args:
                cov_sqrt (Array): Covariance square-root [N*D, N*D].
                eps (Array): Local error estimate [N*D].

            Returns:
                Array: Updated covariance square-root [N*D, N*D].
            """

            return sqrt_L_sum_qr(cov_sqrt, static_cov * jnp.eye(cov_sqrt.shape[0]))

        return cov_update_sqrt
