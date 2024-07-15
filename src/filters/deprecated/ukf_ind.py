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


class UKF_IND(Filter):

    def __init__(
        self,
        rk_solver: RKSolver,
        P0: Array,
        sigma_fn: SigmaFn,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float | None = 0.0,
        anomaly_detection: bool = True,
    ) -> None:
        super().__init__(rk_solver, P0, sigma_fn)
        self.alpha = float(rk_solver.h) * 10.0
        self.beta = beta
        self.kappa = 3.0 - self.rk_solver.x0.shape[-1] if kappa is None else kappa
        self.n = 2 * self.rk_solver.x0.shape[-1]
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
        self.extract_block_diag = jax.jit(
            partial(UKF_IND.extract_block_diag, n=self.rk_solver.x0.shape[-1])
        )
        self.extract_block_diag_vmap = jax.vmap(self.extract_block_diag)

        self.anomaly_detection = anomaly_detection

    @staticmethod
    def extract_block_diag(A: Array, n: int) -> Array:
        slices = jnp.arange(0, A.shape[-1], n)
        slices = jnp.stack([slices, slices], axis=1)

        extract_block = lambda c, x: (c, dynamic_slice(c, x, (n, n)))

        return scan(extract_block, A, slices)[1]

    @staticmethod
    @jax.jit
    @jax.vmap
    def cov_sqrt_vmap(P: Array) -> Array:
        x_P = jsp.linalg.block_diag(P, jnp.eye(P.shape[-1]))
        return jnp.nan_to_num(jsp.linalg.cholesky(x_P, lower=True))

    @staticmethod
    @partial(jax.jit, static_argnums=[0, 1, 2, 3])
    def _predict(
        step_fn: Callable[[Array, Array], Tuple[Array, Array, Array, Array]],
        sigma_sqrt_fn: Callable[[Array], Array],
        extract_block_diag: Callable[[Array], Array],
        extract_block_diag_vmap: Callable[[Array], Array],
        t: Array,
        m: Array,
        P: Array,
        w_m: Array,
        w_c: Array,
        n: int,
        lambda_: float,
    ) -> Tuple[Array, Array, Array, Array]:
        anomaly_flags = []

        P_b = extract_block_diag(P)  # [N, D, D]
        # print("P_b", P_b)
        S = UKF_IND.cov_sqrt_vmap(P_b).transpose(2, 0, 1)  # [2*D, N, 2*D]

        m_0 = jnp.concatenate([m, jnp.zeros_like(m)], axis=2)  # [1, N, 2*D]
        m_1 = m_0 + jnp.sqrt(n + lambda_) * S  # [2*D, N, 2*D]
        m_2 = m_0 - jnp.sqrt(n + lambda_) * S  # [2*D, N, 2*D]

        x_m = (
            jnp.concatenate(
                [m_0, m_1, m_2],
                axis=0,
            )
            .reshape(-1, m.shape[-2], 2, m.shape[-1])
            .transpose(2, 0, 1, 3)
        )  # [2, 4*D+1, N, D]

        t_b = jnp.broadcast_to(t, (x_m.shape[1],))  # [4*D+1]

        anomaly_flags.append(jnp.isinf(x_m).any())
        anomaly_flags.append(jnp.isnan(x_m).any())

        t_next, x_m_next, eps, _ = step_fn(
            t_b, x_m[0, :, :, :]
        )  # [4*D+1], [4*D+1, N, D], [4*D+1, N, D]

        anomaly_flags.append(jnp.isinf(x_m_next).any())
        anomaly_flags.append(jnp.isnan(x_m_next).any())
        anomaly_flags.append(jnp.isinf(eps).any())
        anomaly_flags.append(jnp.isnan(eps).any())

        sigma_sqrt = sigma_sqrt_fn(eps.reshape(-1, m.shape[-2] * m.shape[-1]))  # [4*D+1, N*D, N*D]
        sigma_sqrt = extract_block_diag_vmap(sigma_sqrt)  # [4*D+1, N, D, D]

        anomaly_flags.append(jnp.isinf(sigma_sqrt).any())
        anomaly_flags.append(jnp.isnan(sigma_sqrt).any())

        x_m_next = x_m_next + jnp.einsum(
            "bnij,bnj->bni", sigma_sqrt, x_m[1, :, :, :]
        )  # [4*D+1, N, D]
        m_next = jnp.einsum("b,bni->ni", w_m, x_m_next)  # [N, D]

        # print("m_next", m_next)

        anomaly_flags.append(jnp.isinf(m_next).any())
        anomaly_flags.append(jnp.isnan(m_next).any())

        x_m_delta = x_m_next - m_next[None, :]  # [4*D+1, N, D]
        P_next = jnp.einsum("b,bni,bnj->nij", w_c, x_m_delta, x_m_delta)  # [N, D, D]
        P_next = jsp.linalg.block_diag(*P_next)  # [N*D, N*D]

        anomaly_flags.append(jnp.isinf(P_next).any())
        anomaly_flags.append(jnp.isnan(P_next).any())

        return (
            t_next[0:1],
            m_next[None, :],
            P_next,
            jnp.stack(anomaly_flags),
        )

    def predict(self) -> Tuple[Array, Array, Array]:
        self.t, self.m, self.P, anomaly_flags = self._predict(
            self.rk_solver.step,
            self.sigma_sqrt_fn,
            self.extract_block_diag,
            self.extract_block_diag_vmap,
            self.t,
            self.m,
            self.P,
            self.w_m,
            self.w_c,
            self.n,
            self.lambda_,
        )

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
