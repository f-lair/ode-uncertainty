from jax import Array
from jax import numpy as jnp

from src.covariance_functions.covariance_function import (
    CovarianceFunction,
    CovarianceFunctionBuilder,
)


class OuterCovariance(CovarianceFunctionBuilder):
    """Outer covariance function."""

    def __init__(self, scale: float = 1.0) -> None:
        """
        Initializes covariance function.

        Args:
            scale (float, optional): Scale factor for local error estimate. Defaults to 1.0.
        """

        self.scale = scale

    def build(self) -> CovarianceFunction:
        def cov(eps: Array) -> Array:
            """
            Computes covariance.
            D: Latent dimension.
            N: ODE order.

            Args:
                eps (Array): Local error estimate [N*D].

            Returns:
                Array: Covariance [N*D, N*D].
            """

            scaled_eps = self.scale * eps
            return jnp.outer(scaled_eps, scaled_eps)

        return cov

    def build_sqrt(self) -> CovarianceFunction:
        def cov_sqrt(eps: Array) -> Array:
            """
            Computes covariance square-root.

            Args:
                eps (Array): Local error estimate [N*D].

            Returns:
                Array: Covariance square-root [N*D, N*D].
            """

            scaled_eps = self.scale * eps
            return jnp.outer(scaled_eps, scaled_eps) / jnp.sqrt(jnp.dot(scaled_eps, scaled_eps))

        return cov_sqrt
