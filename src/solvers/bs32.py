from jax import Array
from jax import numpy as jnp

from src.solvers.rksolver import RKSolverBuilder


class BS32(RKSolverBuilder):
    """Bogacki-Shampine solver (stage S=4, order p=3(2))."""

    @classmethod
    def build_A(cls) -> Array:
        return jnp.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1 / 2, 0.0, 0.0, 0.0],
                [0.0, 3 / 4, 0.0, 0.0],
                [2 / 9, 1 / 3, 4 / 9, 0.0],
            ]
        )

    @classmethod
    def build_b(cls) -> Array:
        return jnp.array(
            [
                [7 / 24, 1 / 4, 1 / 3, 1 / 8],
                [2 / 9, 1 / 3, 4 / 9, 0.0],
            ]
        )

    @classmethod
    def build_c(cls) -> Array:
        return jnp.array([0.0, 1 / 2, 3 / 4, 1.0])
