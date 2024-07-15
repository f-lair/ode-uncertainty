from functools import partial
from typing import Callable, Tuple

import jax
from jax import Array
from jax import numpy as jnp
from jax import scipy as jsp

from src.filters.filter import Filter
from src.filters.sigma_fns import SigmaFn
from src.solvers.rksolver import RKSolver


class UKF(Filter):

    def __init__(
        self,
        rk_solver: RKSolver,
        P0: Array,
        sigma_fn: SigmaFn,
        alpha: float = 0.1,
        beta: float = 2.0,
        kappa: float | None = None,
        anomaly_detection: bool = False,
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

        self.sigma_sqrt_fn = jax.vmap(self.sigma_fn.sqrt)

        self.anomaly_detection = anomaly_detection

    @staticmethod
    @partial(jax.jit, static_argnums=[0, 1])
    def _predict(
        step_fn: Callable[[Array, Array], Tuple[Array, Array, Array, Array]],
        sigma_sqrt_fn: Callable[[Array], Array],
        t: Array,
        m: Array,
        P: Array,
        w_m: Array,
        w_c: Array,
        n: int,
        lambda_: float,
    ) -> Tuple[Array, Array, Array, Array]:
        anomaly_flags = []

        x_P = jnp.block(
            [[P, jnp.zeros_like(P)], [jnp.zeros_like(P), jnp.eye(P.shape[-1])]]
        )  # [2*N*D, 2*N*D]
        S = jnp.nan_to_num(jsp.linalg.cholesky(x_P, lower=True))  # [2*N*D, 2*N*D]

        m_0 = jnp.concatenate(
            [
                m.ravel(),
                jnp.zeros(
                    m.size,
                ),
            ]
        )[
            None, :
        ]  # [1, 2*N*D]
        m_1 = m_0 + jnp.sqrt(n + lambda_) * S  # [2*N*D, 2*N*D]
        m_2 = m_0 - jnp.sqrt(n + lambda_) * S  # [2*N*D, 2*N*D]

        x_m = jnp.concatenate(
            [m_0, m_1, m_2],
            axis=0,
        ).reshape(
            -1, 2, m.shape[-2], m.shape[-1]
        )  # [4*N*D+1, 2, N, D]

        t_b = jnp.broadcast_to(t, (x_m.shape[0],))  # [4*N*D+1]

        anomaly_flags.append(jnp.isinf(x_m).any())
        anomaly_flags.append(jnp.isnan(x_m).any())

        t_next, x_m_next, eps, _ = step_fn(
            t_b, x_m[:, 0]
        )  # [4*N*D+1], [4*N*D+1, N, D], [4*N*D+1, N, D]

        anomaly_flags.append(jnp.isinf(x_m_next).any())
        anomaly_flags.append(jnp.isnan(x_m_next).any())
        anomaly_flags.append(jnp.isinf(eps).any())
        anomaly_flags.append(jnp.isnan(eps).any())

        sigma_sqrt = sigma_sqrt_fn(
            eps.reshape(-1, m.shape[-2] * m.shape[-1])
        )  # [4*N*D+1, N*D, N*D]

        anomaly_flags.append(jnp.isinf(sigma_sqrt).any())
        anomaly_flags.append(jnp.isnan(sigma_sqrt).any())

        x_m_next = x_m_next.reshape(-1, m.shape[-2] * m.shape[-1]) + jnp.einsum(
            "bij,bj->bi", sigma_sqrt, x_m[:, 1].reshape(-1, m.shape[-2] * m.shape[-1])
        )  # [4*N*D+1, N*D]
        m_next = x_m_next.T @ w_m  # [N*D]

        # print("m_next", m_next)

        anomaly_flags.append(jnp.isinf(m_next).any())
        anomaly_flags.append(jnp.isnan(m_next).any())

        P_next = jnp.einsum(
            "b,bi,bj->ij", w_c, x_m_next - m_next[None, :], x_m_next - m_next[None, :]
        )  # [N*D, N*D]

        anomaly_flags.append(jnp.isinf(P_next).any())
        anomaly_flags.append(jnp.isnan(P_next).any())

        return (
            t_next[0:1],
            m_next[None, :].reshape(*m.shape),
            P_next,
            jnp.stack(anomaly_flags),
        )

    def predict(self) -> Tuple[Array, Array, Array]:
        self.t, self.m, self.P, anomaly_flags = self._predict(
            self.rk_solver.step,
            self.sigma_sqrt_fn,
            self.t,
            self.m,
            self.P,
            self.w_m,
            self.w_c,
            self.n,
            self.lambda_,
        )

        # if jnp.any(anomaly_flags):
        #     print("OLD", self.m)
        #     print("NEW", m_next)

        if self.anomaly_detection:
            self.detect_anomaly(anomaly_flags)

        return self.t, self.m, self.P

    def detect_anomaly(self, anomaly_flags: Array) -> None:
        if anomaly_flags[0]:
            raise ValueError("Anomaly Detection: \"x_m\" contains +/-inf!")
        elif anomaly_flags[1]:
            raise ValueError("Anomaly Detection: \"x_m\" contains NaN!")
        elif anomaly_flags[2]:
            raise ValueError("Anomaly Detection: \"x_m_next\" contains +/-inf!")
        elif anomaly_flags[3]:
            raise ValueError("Anomaly Detection: \"x_m_next\" contains NaN!")
        elif anomaly_flags[4]:
            raise ValueError("Anomaly Detection: \"eps\" contains +/-inf!")
        elif anomaly_flags[5]:
            raise ValueError("Anomaly Detection: \"eps\" contains NaN!")
        elif anomaly_flags[6]:
            raise ValueError("Anomaly Detection: \"sigma_sqrt\" contains +/-inf!")
        elif anomaly_flags[7]:
            raise ValueError("Anomaly Detection: \"sigma_sqrt\" contains NaN!")
        elif anomaly_flags[8]:
            raise ValueError("Anomaly Detection: \"m_next\" contains +/-inf!")
        elif anomaly_flags[9]:
            raise ValueError("Anomaly Detection: \"m_next\" contains NaN!")
        elif anomaly_flags[10]:
            raise ValueError("Anomaly Detection: \"P_next\" contains +/-inf!")
        elif anomaly_flags[11]:
            raise ValueError("Anomaly Detection: \"P_next\" contains NaN!")
