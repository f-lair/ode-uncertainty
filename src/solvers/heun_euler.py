from jax import Array
from jax import numpy as jnp

from src.solvers.rksolver import RKSolverBuilder


class HeunEuler(RKSolverBuilder):
    """Heun-Euler solver (stage S=2, order p=1(2))."""

    @classmethod
    def build_A(cls) -> Array:
        return jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ]
        )

    @classmethod
    def build_b(cls) -> Array:
        return jnp.array(
            [
                [0.5, 0.5],
                [0.5, 0.0],
            ]
        )

    @classmethod
    def build_c(cls) -> Array:
        return jnp.array([0.0, 1.0])
