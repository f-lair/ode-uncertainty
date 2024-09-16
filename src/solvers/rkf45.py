from jax import Array
from jax import numpy as jnp

from src.solvers.rksolver import RKSolverBuilder


class RKF45(RKSolverBuilder):
    """Runge-Kutta-Fehlberg solver (stage S=6, order p=4(5))."""

    @classmethod
    def build_A(cls) -> Array:
        return jnp.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1 / 4, 0.0, 0.0, 0.0, 0.0, 0.0],
                [3 / 32, 9 / 32, 0.0, 0.0, 0.0, 0.0],
                [1932 / 2197, -7200 / 2197, 7296 / 2197, 0.0, 0.0, 0.0],
                [439 / 216, -8.0, 3680 / 513, -845 / 4104, 0.0, 0.0],
                [-8 / 27, 2.0, -3544 / 2565, 1859 / 4104, -11 / 40, 0.0],
            ]
        )

    @classmethod
    def build_b(cls) -> Array:
        return jnp.array(
            [
                [16 / 135, 0.0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55],
                [25 / 216, 0.0, 1408 / 2565, 2197 / 4104, -1 / 5, 0.0],
            ]
        )

    @classmethod
    def build_c(cls) -> Array:
        return jnp.array([0.0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2])
