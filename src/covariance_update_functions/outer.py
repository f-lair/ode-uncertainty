from jax import Array
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from src.covariance_update_functions.covariance_update_function import (
    CovarianceUpdateFunction,
    CovarianceUpdateFunctionBuilder,
)


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

            return tfp.math.cholesky_update(cov_sqrt, self.scale * eps)

        return cov_update_sqrt
