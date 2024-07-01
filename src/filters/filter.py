from typing import Tuple

from jax import Array

from src.solvers.rksolver import RKSolver


class Filter:

    def __init__(self, rk_solver: RKSolver, P0: Array) -> None:
        self.rk_solver = rk_solver
        self.t = rk_solver.t0
        self.m = rk_solver.x0
        self.P = P0

    def predict(self) -> Tuple[Array, Array, Array]:
        raise NotImplementedError
