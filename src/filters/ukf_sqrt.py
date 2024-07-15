from functools import partial
from typing import Callable, Tuple

import jax
from jax import Array
from jax import numpy as jnp
from jax import scipy as jsp
from tensorflow_probability.substrates import jax as tfp

from src.filters import UKF
from src.filters.sigma_fns import SigmaFn
from src.solvers.rksolver import RKSolver


class UKF_SQRT(UKF):
    """Square-root Unscented Kalman Filter."""

    def __init__(
        self,
        rk_solver: RKSolver,
        P0: Array,
        sigma_fn: SigmaFn,
        alpha: float = 0.1,
        beta: float = 2.0,
        kappa: float | None = None,
        anomaly_detection: bool = True,
    ) -> None:
        """
        Initializes filter.
        D: Latent dimension.
        N: ODE order.

        Args:
            rk_solver (RKSolver): RK solver.
            P0 (Array): Initial covariance [N*D, N*D].
            sigma_fn (SigmaFn): Sigma function.
            alpha (float, optional): UKF parameter alpha. Defaults to 0.1.
            beta (float, optional): UKF parameter beta. Defaults to 2.0.
            kappa (float | None, optional): UKF parameter kappa. Chosen automatically, if None.
                Defaults to None.
            anomaly_detection (bool, optional): Activates anomaly detection. Defaults to False.
        """

        super().__init__(rk_solver, P0, sigma_fn, alpha, beta, kappa, anomaly_detection)

        self.S = jnp.nan_to_num(jsp.linalg.cholesky(self._P, lower=True))  # type: ignore

    @property
    def P(self) -> Array:
        """
        Covariance getter.
        D: Latent dimension.
        N: ODE order.

        Returns:
            Array: Covariance [N*D, N*D].
        """

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
    ) -> Tuple[Array, Array, Array, Array]:
        """
        Jitted predict function of EKF.
        D: Latent dimension.
        N: ODE order.

        Args:
            step_fn (Callable[[Array, Array], Tuple[Array, Array, Array, Array]]):
                RK-solver step function.
            sigma_fn (SigmaFn): Sigma function.
            t (Array): Time [1].
            m (Array): Mean state [1, N, D].
            S (Array): Covariance square-root [N*C, N*C].
            w_m (Array): Weights for mean computation [4*N*D+1].
            w_c (Array): Weights for covariance computation [4*N*D+1].
            n (int): Number of sigma points (2*N*D).
            lambda_ (float): Scaling factor lambda.

        Returns:
            Tuple[Array, Array, Array, Array]: Time [1], mean state [1, N, D],
                covariance square-root [N*D, N*D], anomaly flags [...].
        """

        anomaly_flags = []

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

        anomaly_flags.append(jnp.isinf(x_m).any())
        anomaly_flags.append(jnp.isnan(x_m).any())

        t_next, x_m_next, eps, _ = step_fn(t_b, x_m)  # [4*N*D+1], [4*N*D+1, N, D], [4*N*D+1, N, D]

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
            "bij,bj->bi", sigma_sqrt, x_q
        )  # [4*N*D+1, N*D]
        m_next = x_m_next.T @ w_m  # [N*D]

        anomaly_flags.append(jnp.isinf(m_next).any())
        anomaly_flags.append(jnp.isnan(m_next).any())

        _, S_next = jsp.linalg.qr(
            jnp.sqrt(w_c[1:, None]) * (x_m_next[1:, :] - m_next[None, :]), mode="economic"
        )  # [N*D, N*D]

        anomaly_flags.append(jnp.isinf(S_next).any())
        anomaly_flags.append(jnp.isnan(S_next).any())

        S_next = tfp.math.cholesky_update(
            S_next.T, x_m_next[0] - m_next, jnp.sign(w_c[0]) * jnp.sqrt(jnp.abs(w_c[0]))
        )  # [N*D, N*D]

        anomaly_flags.append(jnp.isinf(S_next).any())
        anomaly_flags.append(jnp.isnan(S_next).any())

        return (
            t_next[0:1],
            m_next[None, :].reshape(*m.shape),
            S_next,
            jnp.stack(anomaly_flags),
        )

    def predict(self) -> Tuple[Array, Array, Array]:
        """
        Predicts state after performing one step of the ODE solver.
        D: Latent dimension.
        N: ODE order.

        Returns:
            Tuple[Array, Array, Array]: Time [1], mean state [1, N, D], covariance [N*D, N*D].
        """

        self.t, self.m, self.S, anomaly_flags = self._predict(
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

        if self.anomaly_detection:
            self.detect_anomaly(anomaly_flags)

        return self.t, self.m, self.P

    def detect_anomaly(self, anomaly_flags: Array) -> None:
        """
        Detects anomaly using captured flags.

        Args:
            anomaly_flags (Array): Anomaly flags [...].
        """

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
            raise ValueError("Anomaly Detection: \"S_next\" (after qr) contains +/-inf!")
        elif anomaly_flags[11]:
            raise ValueError("Anomaly Detection: \"S_next\" (after qr) contains NaN!")
        elif anomaly_flags[12]:
            raise ValueError("Anomaly Detection: \"S_next\" (after chol_update) contains +/-inf!")
        elif anomaly_flags[13]:
            raise ValueError("Anomaly Detection: \"S_next\" (after chol_update) contains NaN!")
