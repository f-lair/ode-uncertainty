from jax import Array
from jax import numpy as jnp


class SigmaFn:
    def __call__(self, eps: Array) -> Array:
        raise NotImplementedError


class DiagonalSigma:
    def __call__(self, eps: Array) -> Array:
        return jnp.diag(eps)


class OuterSqrtSigma:
    def __call__(self, eps: Array) -> Array:
        eps_sqrt = jnp.sqrt(eps)
        return jnp.outer(eps_sqrt, eps_sqrt)


class OuterSigma:
    def __call__(self, eps: Array) -> Array:
        return jnp.outer(eps, eps)
