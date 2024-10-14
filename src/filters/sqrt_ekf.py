from typing import Callable, Dict, Tuple

from jax import Array, lax
from jax import numpy as jnp
from jax import scipy as jsp

from src.covariance_update_functions import DiagonalCovarianceUpdate
from src.covariance_update_functions.covariance_update_function import (
    CovarianceUpdateFunction,
    CovarianceUpdateFunctionBuilder,
)
from src.filters.filter import FilterBuilder, FilterCorrect, FilterPredict
from src.solvers.solver import Solver
from src.utils import sqrt_L_sum_qr, value_and_jacfwd


class SQRT_EKF(FilterBuilder):
    """Square-root Extended Kalman Filter."""

    def __init__(
        self,
        cov_update_fn_builder: CovarianceUpdateFunctionBuilder = DiagonalCovarianceUpdate(),
        disable_cov_update: bool = False,
    ) -> None:
        super().__init__(cov_update_fn_builder)
        self.disable_cov_update = disable_cov_update

    def state_def(self, N: int, D: int, L: int) -> Dict[str, Tuple[int, ...]]:
        """
        Defines the solver state.

        Args:
            N (int): ODE order.
            D (int): Latent dimension.
            L (int): Measurement dimension.

        Raises:
            NotImplementedError: Needs to be defined for a concrete filter.

        Returns:
            Dict[str, Tuple[int, ...]]: State definition.
        """

        return {
            "t": (1,),
            "x": (1, N, D),
            "P_sqrt": (1, N * D, N * D),
            "Q_sqrt": (N * D, N * D),
            "y": (L,),
            "y_hat": (1, L),
            "R_sqrt": (L, L),
            "S_sqrt": (1, L, L),
        }

    def build_cov_update_fn(self) -> CovarianceUpdateFunction:
        return self.cov_update_fn_builder.build_sqrt()

    def build_predict(self) -> FilterPredict:
        def predict(
            solver: Solver, cov_update_fn_sqrt: CovarianceUpdateFunction, state: Dict[str, Array]
        ) -> Dict[str, Array]:
            cov_update_false_Q_add_true = lambda P_sqrt_next, Q_sqrt, eps: sqrt_L_sum_qr(
                P_sqrt_next, Q_sqrt
            )
            cov_update_false_Q_add_false = lambda P_sqrt_next, Q_sqrt, eps: P_sqrt_next
            cov_update_true_Q_add_true = lambda P_sqrt_next, Q_sqrt, eps: sqrt_L_sum_qr(
                P_sqrt_next, cov_update_fn_sqrt(Q_sqrt, jnp.diag(Q_sqrt) * eps.ravel())
            )
            cov_update_true_Q_add_false = lambda P_sqrt_next, Q_sqrt, eps: cov_update_fn_sqrt(
                P_sqrt_next, eps.ravel()
            )

            cov_update_true = lambda P_sqrt_next, Q_sqrt, eps: lax.cond(
                jnp.any(Q_sqrt >= 1e-16),
                cov_update_true_Q_add_true,
                cov_update_true_Q_add_false,
                P_sqrt_next,
                Q_sqrt,
                eps,
            )

            cov_update_false = lambda P_sqrt_next, Q_sqrt, eps: lax.cond(
                jnp.any(Q_sqrt >= 1e-16),
                cov_update_false_Q_add_true,
                cov_update_false_Q_add_false,
                P_sqrt_next,
                Q_sqrt,
                eps,
            )

            t, x, P_sqrt, Q_sqrt = state["t"], state["x"], state["P_sqrt"], state["Q_sqrt"]
            solver_state = {"t": t, "x": x}

            next_solver_state, solver_jacs = value_and_jacfwd(solver, solver_state)

            t_next = next_solver_state["t"]  # [1]
            x_next = next_solver_state["x"]  # [1, N, D]
            eps = next_solver_state["eps"]  # [1, N, D]
            jac = solver_jacs["x"]["x"].reshape(x.size, x.size)  # [N*D, N*D]

            P_sqrt_next = jac @ P_sqrt[0]  # [N*D, N*D]
            P_sqrt_next = lax.cond(
                self.disable_cov_update,
                cov_update_false,
                cov_update_true,
                P_sqrt_next,
                Q_sqrt,
                eps,
            )  # [N*D, N*D]

            next_state = {
                "t": t_next,
                "x": x_next,
                "P_sqrt": P_sqrt_next[None, :, :],
                "Q_sqrt": state["Q_sqrt"],
                "y": state["y"],
                "y_hat": state["y_hat"],
                "R_sqrt": state["R_sqrt"],
                "S_sqrt": state["S_sqrt"],
            }
            return next_state

        return predict

    def build_correct(self) -> FilterCorrect:
        def correct(
            measurement_fn: Callable[[Array], Array], state: Dict[str, Array]
        ) -> Dict[str, Array]:
            K_zero_true = lambda S_sqrt, H, P_sqrt: jnp.zeros_like(H).T
            K_zero_false = lambda S_sqrt, H, P_sqrt: (
                jsp.linalg.cho_solve((S_sqrt, True), H) @ P_sqrt[0] @ P_sqrt[0].T
            ).T

            x, P_sqrt, y, R_sqrt = state["x"], state["P_sqrt"], state["y"], state["R_sqrt"]

            y_hat, H = value_and_jacfwd(measurement_fn, x.ravel())  # [L], [L, N*D]
            y_delta = y - y_hat  # [L]

            S_sqrt = sqrt_L_sum_qr(H @ P_sqrt[0], R_sqrt)  # [L, L]
            K = lax.cond(
                jnp.all(S_sqrt < 1e-16), K_zero_true, K_zero_false, S_sqrt, H, P_sqrt
            )  # [N*D, L]

            x_corrected = x + (K @ y_delta).reshape(*x.shape)  # [1, N, D]

            # Joseph update for increased numerical stability
            A = jnp.eye(P_sqrt.shape[-1]) - K @ H  # [N*D, N*D]
            P_sqrt_corrected = sqrt_L_sum_qr(A @ P_sqrt[0], K @ R_sqrt)[None, :, :]

            next_state = {
                "t": state["t"],
                "x": x_corrected,
                "P_sqrt": P_sqrt_corrected,
                "Q_sqrt": state["Q_sqrt"],
                "y": state["y"],
                "y_hat": y_hat[None, :],
                "R_sqrt": state["R_sqrt"],
                "S_sqrt": S_sqrt[None, :, :],
            }
            return next_state

        return correct
