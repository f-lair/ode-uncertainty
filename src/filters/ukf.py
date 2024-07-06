from functools import partial
from typing import Callable, Tuple

import jax
from jax import Array
from jax import numpy as jnp
from jax import scipy as jsp
from tensorflow_probability.substrates import jax as tfp

from src.filters.filter import Filter
from src.filters.sigma_fns import SigmaFn
from src.solvers.rksolver import RKSolver


class UKF(Filter):

    def __init__(
        self,
        rk_solver: RKSolver,
        P0: Array,
        sigma_fn: SigmaFn,
        alpha: float = 0.001,
        beta: float = 2.0,
        kappa: float | None = None,
    ) -> None:
        super().__init__(rk_solver, P0, sigma_fn)
        self.alpha = alpha
        self.beta = beta
        self.kappa = 3.0 - P0.shape[0] if kappa is None else kappa
        self.n = 2 * P0.shape[0]
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n

        self.w_m = jnp.concatenate(
            [
                jnp.array([self.lambda_ / (self.n + self.lambda_)]),
                jnp.full((2 * self.n,), 1 / (2 * (self.n + self.lambda_))),
            ]
        )  # [4*N*D+1]
        self.w_c = jnp.concatenate(
            [
                jnp.array(
                    [self.lambda_ / (self.n + self.lambda_) + (1 - self.alpha**2 + self.beta)]
                ),
                jnp.full((2 * self.n,), 1 / (2 * (self.n + self.lambda_))),
            ]
        )  # [4*N*D+1]

        self.S = jsp.linalg.cholesky(self._P, lower=True)  # type: ignore
        self.sigma_sqrt_fn = jax.vmap(self.sigma_fn.sqrt)

    @property
    def P(self) -> Array:
        return self.S @ self.S.T

    @staticmethod
    @partial(jax.jit, static_argnums=[0, 1])
    def _predict(
        step_fn: Callable[[Array, Array], Tuple[Array, Array, Array, Array]],
        sigma_sqrt_fn: Callable[[Array], Array],
        t: Array,
        m: Array,
        S: Array,
        w_m: Array,
        w_c: Array,
        n: int,
        lambda_: float,
    ) -> Tuple[Array, Array, Array]:
        m_0 = m.ravel()[None, :]  # [1, N*D]
        m_1 = jnp.broadcast_to(m_0, S.shape)  # [N*D, N*D]
        m_2 = jnp.sqrt(n + lambda_) * S  # [N*D, N*D]
        q_0 = jnp.zeros_like(m_0)  # [1, N*D]
        q_1 = jnp.zeros_like(S)  # [N*D, N*D]
        q_2 = jnp.full_like(S, jnp.sqrt(n + lambda_))  # [N*D, N*D]

        x_m = jnp.concatenate(
            [
                m_0,
                m_0 + m_2,
                m_1,
                m_0 - m_2,
                m_1,
            ],
            axis=0,
        ).reshape(
            -1, m.shape[-2], m.shape[-1]
        )  # [4*N*D+1, N, D]
        x_q = jnp.concatenate(
            [
                q_0,
                q_1,
                q_2,
                q_1,
                -q_2,
            ],
            axis=0,
        )  # [4*N*D+1, N*D]

        t_b = jnp.broadcast_to(t, (x_m.shape[0],))  # [4*N*D+1]
        t_next, x_m_next, eps, _ = step_fn(t_b, x_m)  # [4*N*D+1], [4*N*D+1, N, D], [4*N*D+1, N, D]
        sigma_sqrt = sigma_sqrt_fn(
            eps.reshape(-1, m.shape[-2] * m.shape[-1])
        )  # [4*N*D+1, N*D, N*D]
        x_m_next = x_m_next.reshape(-1, m.shape[-2] * m.shape[-1]) + jnp.einsum(
            "bij,bj->bi", sigma_sqrt, x_q
        )  # [4*N*D+1, N*D]

        m_next = x_m_next.T @ w_m  # [N*D]
        _, S_next = jsp.linalg.qr(
            jnp.sqrt(w_c[1:, None]) * (x_m_next[1:, :] - m_next[None, :]), mode="economic"
        )  # [N*D, N*D]
        S_next = tfp.math.cholesky_update(
            S_next.T, x_m_next[0] - m_next, jnp.sign(w_c[0]) * jnp.sqrt(jnp.abs(w_c[0]))
        )  # [N*D, N*D]

        return t_next[0:1], m_next[None, :].reshape(*m.shape), S_next

    def predict(self) -> Tuple[Array, Array, Array]:
        self.t, self.m, self.S = self._predict(
            self.rk_solver.step,
            self.sigma_sqrt_fn,
            self.t,
            self.m,
            self.S,
            self.w_m,
            self.w_c,
            self.n,
            self.lambda_,
        )

        return self.t, self.m, self.S
