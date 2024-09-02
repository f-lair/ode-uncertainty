from jax import Array
from jax import numpy as jnp

from src.solvers.rksolver import RKSolver


class HeunEuler(RKSolver):
    """Heun-Euler solver (stage S=2, order p=1(2))."""

    @staticmethod
    def _A() -> Array:
        return jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ]
        )

    @staticmethod
    def _b() -> Array:
        return jnp.array(
            [
                [0.5, 0.5],
                [0.5, 0.0],
            ]
        )

    @staticmethod
    def _c() -> Array:
        return jnp.array([0.0, 1.0])

    @staticmethod
    def _q() -> int:
        return 1
