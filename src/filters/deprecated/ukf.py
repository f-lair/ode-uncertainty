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
    """Unscented Kalman Filter."""

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 2.0,
        kappa: float | None = None,
        anomaly_detection: bool = True,
    ) -> None:
        """
        Initializes filter.

        Args:
            alpha (float, optional): UKF parameter alpha. Defaults to 0.1.
            beta (float, optional): UKF parameter beta. Defaults to 2.0.
            kappa (float | None, optional): UKF parameter kappa. Chosen automatically, if None.
                Defaults to None.
            anomaly_detection (bool, optional): Activates anomaly detection. Defaults to False.
        """

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.anomaly_detection = anomaly_detection

    def setup(self, rk_solver: RKSolver, P0: Array, sigma_fn: SigmaFn) -> None:
        super().setup(rk_solver, P0, sigma_fn)

        self.kappa = 3.0 - P0.shape[-1] if self.kappa is None else self.kappa

        self.sigma_sqrt_fn = jax.vmap(self.sigma_fn.sqrt)  # type: ignore

    @staticmethod
    @partial(jax.jit, static_argnums=[0, 1, 2, 3, 4])
    def _predict_jit(
        step_fn: Callable[[Array, Array, Array], Tuple[Array, Array, Array, Array]],
        sigma_sqrt_fn: Callable[[Array], Array],
        alpha: float,
        beta: float,
        kappa: float,
        t: Array,
        m: Array,
        P: Array,
        gamma: Array,
        ode_params: Array,
    ) -> Tuple[Array, Array, Array, Array, Array]:
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
            P (Array): Covariance [1, N*C, N*C].
            n (int): Number of sigma points (2*N*D).
            lambda_ (float): Scaling factor lambda.

        Returns:
            Tuple[Array, Array, Array, Array, Array]: Time [1], mean state [1, N, D],
                sigma points [4*N*D+1, N, D], covariance [1, N*D, N*D], anomaly flags [...].
        """

        anomaly_flags = []

        n = 2 * P.shape[-1]
        lambda_ = alpha**2 * (n + kappa) - n
        w_m = jnp.concatenate(
            [
                jnp.array([lambda_ / (n + lambda_)]),
                jnp.full((2 * n,), 1 / (2 * (n + lambda_))),
            ]
        )  # [4*N*D+1]
        w_c = jnp.concatenate(
            [
                jnp.array([lambda_ / (n + lambda_) + (1 - alpha**2 + beta)]),
                jnp.full((2 * n,), 1 / (2 * (n + lambda_))),
            ]
        )  # [4*N*D+1]

        P_ = P[0] + jnp.diag(jnp.full(P.shape[-1], 1e-16))
        x_P = jnp.block(
            [[P_, jnp.zeros_like(P[0])], [jnp.zeros_like(P[0]), jnp.eye(P.shape[-1])]]
        )  # [2*N*D, 2*N*D]
        S = jnp.linalg.cholesky(x_P)  # [2*N*D, 2*N*D]

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
        m_1 = m_0 + jnp.sqrt(n + lambda_) * S.T  # [2*N*D, 2*N*D]  # type: ignore
        m_2 = m_0 - jnp.sqrt(n + lambda_) * S.T  # [2*N*D, 2*N*D]  # type: ignore

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
            t_b, x_m[:, 0], ode_params
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

        anomaly_flags.append(jnp.isinf(m_next).any())
        anomaly_flags.append(jnp.isnan(m_next).any())

        P_next = jnp.einsum(
            "b,bi,bj->ij", w_c, x_m_next - m_next[None, :], x_m_next - m_next[None, :]
        ) + jnp.diag(
            jnp.full(P.shape[-1], gamma)
        )  # [N*D, N*D]

        anomaly_flags.append(jnp.isinf(P_next).any())
        anomaly_flags.append(jnp.isnan(P_next).any())

        return (
            t_next[0:1],
            m_next[None, :].reshape(*m.shape),
            P_next[None, :, :],
            x_m_next.reshape(-1, *m.shape[1:]),
            jnp.stack(anomaly_flags),
        )

    @staticmethod
    @partial(jax.jit, static_argnums=[0, 1, 2])
    def _correct_jit(
        alpha: float,
        beta: float,
        kappa: float,
        m: Array,
        P: Array,
        y: Array,
        H: Array,
        R: Array,
    ) -> Tuple[Array, Array, Array, Array]:

        n = P.shape[-1]
        lambda_ = alpha**2 * (n + kappa) - n
        w_m = jnp.concatenate(
            [
                jnp.array([lambda_ / (n + lambda_)]),
                jnp.full((2 * n,), 1 / (2 * (n + lambda_))),
            ]
        )  # [2*N*D+1]
        w_c = jnp.concatenate(
            [
                jnp.array([lambda_ / (n + lambda_) + (1 - alpha**2 + beta)]),
                jnp.full((2 * n,), 1 / (2 * (n + lambda_))),
            ]
        )  # [2*N*D+1]

        S_ = jnp.linalg.cholesky(P[0] + jnp.diag(jnp.full(P.shape[-1], 1e-16)))  # [N*D, N*D]

        m_0 = m.ravel()[None, :]  # [1, N*D]
        m_1 = m_0 + jnp.sqrt(n + lambda_) * S_.T  # [N*D, N*D]  # type: ignore
        m_2 = m_0 - jnp.sqrt(n + lambda_) * S_.T  # [N*D, N*D]  # type: ignore

        x_m = jnp.concatenate(
            [m_0, m_1, m_2],
            axis=0,
        )  # [2*N*D+1, N*D]

        y_m = H @ x_m.T  # [L, 2*N*D+1]
        y_hat = y_m @ w_m  # [L]
        y_delta = y - y_hat  # [L]

        S = (
            jnp.einsum("b,ib,jb->ij", w_c, y_m - y_hat[:, None], y_m - y_hat[:, None]) + R
        )  # [L, L]
        S_cho = jsp.linalg.cho_factor(S, lower=True)  # [L, L]
        C = jnp.einsum("b,bi,jb->ij", w_c, x_m - m_0, y_m - y_hat[:, None])  # [N*D, L]
        K = jsp.linalg.cho_solve(S_cho, C.T).T  # [N*D, L]

        m_corrected = m + (K @ y_delta).reshape(*m.shape)  # [1, N, D]
        P_corrected = P - (K @ S @ K.T)[None, :, :]  # [1, N*D, N*D]

        return m_corrected, P_corrected, y_hat, S

    def _predict(self) -> Tuple[Array, Array, Array, Array, Array]:
        """
        Predicts state after performing one step of the ODE solver.
        D: Latent dimension.
        N: ODE order.

        Returns:
            Tuple[Array, Array, Array]: Time [1], mean state [1, N, D], covariance [1, N*D, N*D],
            mean state derivative [1, N, D], sigma points [4*N*D+1, N, D].
        """

        dx_dts = self.rk_solver.fn(self.t, self.m)
        self.t, self.m, self.P, sigma_points, anomaly_flags = self._predict_jit(
            self.rk_solver.step,
            self.sigma_sqrt_fn,
            self.alpha,
            self.beta,
            self.kappa,
            self.t,
            self.m,
            self.P,
            0.0,
            self.rk_solver.fn.params,
        )

        if self.anomaly_detection:
            self.detect_anomaly(anomaly_flags)

        return self.t, self.m, self.P, dx_dts, sigma_points

    @staticmethod
    def results_spec_predict() -> Tuple[str, ...]:
        """
        Results specification.

        Returns:
            Tuple[str, ...]: Results keys.
        """

        return "ts", "xs", "Ps", "dx_dts", "Sigma_points"

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
            raise ValueError("Anomaly Detection: \"P_next\" contains +/-inf!")
        elif anomaly_flags[11]:
            raise ValueError("Anomaly Detection: \"P_next\" contains NaN!")
