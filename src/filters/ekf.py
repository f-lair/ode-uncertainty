from functools import partial
from typing import Callable, Tuple

import jax
from jax import Array
from jax import numpy as jnp

from src.filters.filter import Filter
from src.filters.sigma_fns import SigmaFn
from src.solvers.rksolver import RKSolver


class EKF(Filter):

    def __init__(self, rk_solver: RKSolver, P0: Array, sigma_fn: SigmaFn) -> None:
        super().__init__(rk_solver, P0, sigma_fn)

    @staticmethod
    def _rk_solver_step_AD(
        step_fn: Callable[[Array, Array], Tuple[Array, Array, Array, Array]], t: Array, x: Array
    ) -> Array:
        return step_fn(t, x)[1]

    @staticmethod
    @partial(jax.jit, static_argnums=[0, 1])
    def _predict(
        step_fn: Callable[[Array, Array], Tuple[Array, Array, Array, Array]],
        sigma_fn: SigmaFn,
        t: Array,
        m: Array,
        P: Array,
    ) -> Tuple[Array, Array, Array]:
        t_next, m_next, eps, _ = step_fn(t, m)
        jac = jax.jacfwd(partial(EKF._rk_solver_step_AD, step_fn, t))(m).reshape(m.size, m.size)
        sigma = sigma_fn(eps.ravel())
        P_next = jac @ P @ jac.T + sigma

        return t_next, m_next, P_next

    def predict(self) -> Tuple[Array, Array, Array]:
        self.t, self.m, self.P = self._predict(
            self.rk_solver.step, self.sigma_fn, self.t, self.m, self.P
        )

        return self.t, self.m, self.P
