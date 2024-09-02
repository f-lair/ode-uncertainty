from typing import Tuple

from jax import Array
from jax import numpy as jnp
from jax import random

from src.filters.sigma_fns import SigmaFn


class PerturbationFn:
    """Abstract base class for perturbation functions."""

    def __call__(
        self, sigma_fn_vmap: SigmaFn, dx_dts: Array, eps: Array, prng_key: Array
    ) -> Tuple[Array, Array]:
        """
        Evaluates function.
        D: Latent dimension.
        N: ODE order.

        Args:
            eps (Array): Local error estimate [N, D].

        Raises:
            NotImplementedError: Needs to be implemented for a concrete function.

        Returns:
            Array: Perturbation [N, D].
        """

        raise NotImplementedError


class Gaussian(PerturbationFn):
    def __call__(
        self, sigma_fn_vmap: SigmaFn, dx_dts: Array, eps: Array, prng_key: Array
    ) -> Tuple[Array, Array]:
        M, N, D = eps.shape
        sigma = sigma_fn_vmap(eps.reshape(M, N * D))  # [M, N*D, N*D]
        p = random.multivariate_normal(
            prng_key, jnp.zeros((M, N * D)), sigma, method="svd"
        ).reshape(
            M, N, D
        )  # [M, N, D]

        return p, sigma
