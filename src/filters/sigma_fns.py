from jax import Array
from jax import numpy as jnp


class SigmaFn:
    def __call__(self, eps: Array) -> Array:
        raise NotImplementedError


class DiagonalSigma:
    def __call__(self, eps: Array) -> Array:
        return jnp.diag(eps)


class OuterSigma:
    def __call__(self, eps: Array) -> Array:
        return jnp.outer(eps, eps)
