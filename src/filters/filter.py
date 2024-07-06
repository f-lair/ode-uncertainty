from typing import Tuple

from jax import Array

from src.filters.sigma_fns import SigmaFn
from src.solvers.rksolver import RKSolver


class Filter:

    def __init__(self, rk_solver: RKSolver, P0: Array, sigma_fn: SigmaFn) -> None:
        self.rk_solver = rk_solver
        self.t = rk_solver.t0
        self.m = rk_solver.x0
        self._P = P0
        self.sigma_fn = sigma_fn

    def predict(self) -> Tuple[Array, Array, Array]:
        raise NotImplementedError

    @property
    def P(self) -> Array:
        return self._P

    @P.setter
    def P(self, value: Array) -> None:
        self._P = value
