from functools import partial
from typing import Callable, Tuple

import jax
from jax import Array
from jax import numpy as jnp
from jax import scipy as jsp
from jax.lax import dynamic_slice, scan

from src.filters.filter import Filter
from src.filters.sigma_fns import SigmaFn
from src.solvers.rksolver import RKSolver


class EKF_IND(Filter):

    def __init__(self, rk_solver: RKSolver, P0: Array, sigma_fn: SigmaFn) -> None:
        super().__init__(rk_solver, P0, sigma_fn)
        self.sigma_fn = sigma_fn
        self.extract_block_diag_jit = jax.jit(
            partial(EKF_IND.extract_block_diag, n=rk_solver.x0.shape[-1])
        )

    @staticmethod
    def _rk_solver_step_AD(
        step_fn: Callable[[Array, Array], Tuple[Array, Array, Array, Array]], t: Array, x: Array
    ) -> Array:
        return step_fn(t, x)[1]

    @staticmethod
    def extract_block_diag(A: Array, n: int) -> Array:
        slices = jnp.arange(0, A.shape[-1], n)
        slices = jnp.stack([slices, slices], axis=1)

        extract_block = lambda c, x: (c, dynamic_slice(c, x, (n, n)))

        return scan(extract_block, A, slices)[1]

    @staticmethod
    @partial(jax.jit, static_argnums=[0, 1, 2])
    def _predict(
        step_fn: Callable[[Array, Array], Tuple[Array, Array, Array, Array]],
        sigma_fn: SigmaFn,
        extract_block_diag: Callable[[Array], Array],
        t: Array,
        m: Array,
        P: Array,
    ) -> Tuple[Array, Array, Array]:
        t_next, m_next, eps, _ = step_fn(t, m)

        jac = jax.jacfwd(partial(EKF_IND._rk_solver_step_AD, step_fn, t))(m)  # [1, N, D, 1, N, D]
        jac = jnp.squeeze(jac, (0, 3)).transpose(0, 2, 1, 3)  # [N, N, D, D]
        jac = jac[jnp.arange(m.shape[-2]), jnp.arange(m.shape[-2])]  # [N, D, D]

        sigma = sigma_fn(eps.ravel())  # [N*D, N*D]
        sigma = extract_block_diag(sigma)  # [N, D, D]
        P_blocks = extract_block_diag(P)  # [N, D, D]

        P_next = jnp.einsum("nij,njk,nlk->nil", jac, P_blocks, jac) + sigma  # [N, D, D]
        P_next = jsp.linalg.block_diag(*P_next)  # [N*D, N*D]

        return t_next, m_next, P_next

    def predict(self) -> Tuple[Array, Array, Array]:
        self.t, self.m, self.P = self._predict(
            self.rk_solver.step, self.sigma_fn, self.extract_block_diag_jit, self.t, self.m, self.P
        )

        return self.t, self.m, self.P
