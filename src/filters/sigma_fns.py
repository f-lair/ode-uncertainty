from jax import Array
from jax import numpy as jnp


class SigmaFn:
    def __call__(self, eps: Array) -> Array:
        raise NotImplementedError

    @staticmethod
    def sqrt(eps: Array) -> Array:
        raise NotImplementedError


class DiagonalSigma(SigmaFn):
    def __call__(self, eps: Array) -> Array:
        return jnp.diag(eps)

    @staticmethod
    def sqrt(eps: Array) -> Array:
        return jnp.diag(jnp.sqrt(eps))


class OuterSigma(SigmaFn):
    def __call__(self, eps: Array) -> Array:
        return jnp.outer(eps, eps)

    @staticmethod
    def sqrt(eps: Array) -> Array:
        return jnp.outer(eps, eps) / jnp.sqrt(jnp.dot(eps, eps))
