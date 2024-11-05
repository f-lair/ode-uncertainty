import operator
from typing import Callable, Dict, Tuple

import jax
from jax import Array, jacfwd, lax
from jax import numpy as jnp
from jax import scipy as jsp
from jax import tree

from src.covariance_update_functions import DiagonalCovarianceUpdate
from src.covariance_update_functions.covariance_update_function import (
    CovarianceUpdateFunction,
    CovarianceUpdateFunctionBuilder,
)
from src.filters.filter import (
    FilterBuilder,
    FilterCorrect,
    FilterPredict,
    ParametrizedFilterPredict,
)
from src.ode.ode import ODE
from src.solvers.solver import ParametrizedSolver, Solver
from src.utils import jmp_aux, mjp_aux, sqrt_L_sum_qr, value_and_jacfwd


class SQRT_EKF(FilterBuilder):
    """Square-root Extended Kalman Filter."""

    def __init__(
        self,
        cov_update_fn_builder: CovarianceUpdateFunctionBuilder = DiagonalCovarianceUpdate(),
        disable_cov_update: bool = False,
    ) -> None:
        super().__init__(cov_update_fn_builder)
        self.disable_cov_update = disable_cov_update

    def init_state(
        self,
        solver_state: Dict[str, Array],
        P0_sqrt: Array,
        Q_sqrt: Array,
        gamma_sqrt: Array,
        R_sqrt: Array,
    ) -> Dict[str, Array]:
        """
        Initializes the filter state.
        D: Latent dimension.
        N: ODE order.

        Args:
            t0 (Array): Initial time [].
            x0 (Array): Initial state [N, D].

        Returns:
            Dict[str, Array]: Initial state.
        """

        filter_state = super().init_state(solver_state)
        L = R_sqrt.shape[-1]

        filter_state["t"] = filter_state["t"][None]
        filter_state["x"] = filter_state["x"][None, :, :]
        filter_state["eps"] = filter_state["eps"][None, :, :]
        filter_state["diffrax_state"] = tree.map(
            lambda _x: _x[None, ...],
            filter_state["diffrax_state"],
        )
        filter_state["P_sqrt"] = P0_sqrt[None, :, :]
        filter_state["Q_sqrt"] = Q_sqrt
        filter_state["gamma_sqrt"] = gamma_sqrt
        filter_state["y"] = jnp.zeros(L)
        filter_state["y_hat"] = jnp.zeros((1, L))
        filter_state["R_sqrt"] = R_sqrt
        filter_state["S_sqrt"] = jnp.zeros((1, L, L))

        return filter_state

    def build_cov_update_fn(self) -> CovarianceUpdateFunction:
        return self.cov_update_fn_builder.build_sqrt()

    def build_predict(self) -> FilterPredict:
        def predict(
            solver: Solver, cov_update_fn_sqrt: CovarianceUpdateFunction, state: Dict[str, Array]
        ) -> Dict[str, Array]:
            def cov_update_true(P_sqrt_next, Q_sqrt, gamma_sqrt, eps):
                cov_update_true_Q_add_true = (
                    lambda _P_sqrt_next, _Q_sqrt, _gamma_sqrt, _eps: sqrt_L_sum_qr(
                        cov_update_fn_sqrt(_P_sqrt_next, _eps.ravel()), _gamma_sqrt * _Q_sqrt
                    )
                )
                cov_update_true_Q_add_false = (
                    lambda _P_sqrt_next, _Q_sqrt, _gamma_sqrt, _eps: cov_update_fn_sqrt(
                        _P_sqrt_next, _eps.ravel()
                    )
                )

                return lax.cond(
                    jnp.any(Q_sqrt >= 1e-16),
                    cov_update_true_Q_add_true,
                    cov_update_true_Q_add_false,
                    P_sqrt_next,
                    Q_sqrt,
                    gamma_sqrt,
                    eps,
                )

            def cov_update_false(P_sqrt_next, Q_sqrt, gamma_sqrt, eps):
                cov_update_false_Q_add_true = (
                    lambda _P_sqrt_next, _Q_sqrt, _gamma_sqrt, _eps: sqrt_L_sum_qr(
                        _P_sqrt_next, _gamma_sqrt * _Q_sqrt
                    )
                )
                cov_update_false_Q_add_false = (
                    lambda _P_sqrt_next, _Q_sqrt, _gamma_sqrt, _eps: _P_sqrt_next
                )

                return lax.cond(
                    jnp.any(Q_sqrt >= 1e-16),
                    cov_update_false_Q_add_true,
                    cov_update_false_Q_add_false,
                    P_sqrt_next,
                    Q_sqrt,
                    gamma_sqrt,
                    eps,
                )

            t, x, diffrax_state, P_sqrt, Q_sqrt, gamma_sqrt = (
                state["t"],
                state["x"],
                state["diffrax_state"],
                state["P_sqrt"],
                state["Q_sqrt"],
                state["gamma_sqrt"],
            )

            def solver_jmp_wrapper(x_flat: Array) -> Tuple[Array, Tuple[Array, Array, Array]]:
                solver_state = {
                    "t": t,
                    "x": x_flat.reshape(*x.shape),
                    "diffrax_state": diffrax_state,
                }
                next_solver_state = solver(solver_state)
                x_next_flat = next_solver_state["x"].flatten()
                return x_next_flat, (
                    next_solver_state["t"],
                    next_solver_state["eps"],
                    next_solver_state["diffrax_state"],
                )

            x_next, P_sqrt_next, (t_next, eps, diffrax_state_next) = jmp_aux(
                solver_jmp_wrapper, (None, None, None), [x.flatten()], [P_sqrt[0]]
            )

            # x_next, P_sqrt_next, (t_next, eps, diffrax_state_next) = mjp_aux(
            #     solver_jmp_wrapper, [x.flatten()], [P_sqrt[0].T]
            # )
            # P_sqrt_next = P_sqrt_next.T

            x_next = x_next.reshape(x.shape)

            P_sqrt_next = lax.cond(
                self.disable_cov_update,
                cov_update_false,
                cov_update_true,
                P_sqrt_next,
                Q_sqrt,
                gamma_sqrt,
                eps,
            )  # [N*D, N*D]

            next_state = {
                "t": t_next,
                "x": x_next,
                "diffrax_state": diffrax_state_next,
                "eps": eps,
                "P_sqrt": P_sqrt_next[None, :, :],
                "Q_sqrt": state["Q_sqrt"],
                "gamma_sqrt": state["gamma_sqrt"],
                "y": state["y"],
                "y_hat": state["y_hat"],
                "R_sqrt": state["R_sqrt"],
                "S_sqrt": state["S_sqrt"],
            }
            return next_state

        return predict

    def build_parametrized_predict(self) -> ParametrizedFilterPredict:
        def parametrized_predict(
            solver: ParametrizedSolver,
            cov_update_fn_sqrt: CovarianceUpdateFunction,
            ode: ODE,
            params: Dict[str, Array],
            state: Dict[str, Array],
        ) -> Dict[str, Array]:
            def cov_update_true(P_sqrt_next, Q_sqrt, gamma_sqrt, eps):
                cov_update_true_Q_add_true = (
                    lambda _P_sqrt_next, _Q_sqrt, _gamma_sqrt, _eps: sqrt_L_sum_qr(
                        cov_update_fn_sqrt(_P_sqrt_next, _eps.ravel()), _gamma_sqrt * _Q_sqrt
                    )
                )
                cov_update_true_Q_add_false = (
                    lambda _P_sqrt_next, _Q_sqrt, _gamma_sqrt, _eps: cov_update_fn_sqrt(
                        _P_sqrt_next, _eps.ravel()
                    )
                )

                return lax.cond(
                    jnp.any(Q_sqrt >= 1e-16),
                    cov_update_true_Q_add_true,
                    cov_update_true_Q_add_false,
                    P_sqrt_next,
                    Q_sqrt,
                    gamma_sqrt,
                    eps,
                )

            def cov_update_false(P_sqrt_next, Q_sqrt, gamma_sqrt, eps):
                cov_update_false_Q_add_true = (
                    lambda _P_sqrt_next, _Q_sqrt, _gamma_sqrt, _eps: sqrt_L_sum_qr(
                        _P_sqrt_next, _gamma_sqrt * _Q_sqrt
                    )
                )
                cov_update_false_Q_add_false = (
                    lambda _P_sqrt_next, _Q_sqrt, _gamma_sqrt, _eps: _P_sqrt_next
                )

                return lax.cond(
                    jnp.any(Q_sqrt >= 1e-16),
                    cov_update_false_Q_add_true,
                    cov_update_false_Q_add_false,
                    P_sqrt_next,
                    Q_sqrt,
                    gamma_sqrt,
                    eps,
                )

            t, x, diffrax_state, P_sqrt, Q_sqrt, gamma_sqrt = (
                state["t"],
                state["x"],
                state["diffrax_state"],
                state["P_sqrt"],
                state["Q_sqrt"],
                state["gamma_sqrt"],
            )

            def solver_jmp_wrapper(x_flat: Array) -> Tuple[Array, Tuple[Array, Array, Array]]:
                solver_state = {
                    "t": t,
                    "x": x_flat.reshape(*x.shape),
                    "diffrax_state": diffrax_state,
                }
                next_solver_state = solver(ode, params, solver_state)
                x_next_flat = next_solver_state["x"].flatten()
                return x_next_flat, (
                    next_solver_state["t"],
                    next_solver_state["eps"],
                    next_solver_state["diffrax_state"],
                )

            x_next, P_sqrt_next, (t_next, eps, diffrax_state_next) = jmp_aux(
                solver_jmp_wrapper, (None, None, None), [x.flatten()], [P_sqrt[0]]
            )
            # x_next, P_sqrt_next, (t_next, eps, diffrax_state_next) = mjp_aux(
            #     solver_jmp_wrapper, [x.flatten()], [P_sqrt[0].T]
            # )
            # P_sqrt_next = P_sqrt_next.T

            x_next = x_next.reshape(x.shape)

            params_jac = tree.map(lambda _x: jnp.abs(_x), jacfwd(ode, 2)(t[0], x[0], params))
            Q_vec = tree.reduce(operator.add, params_jac).flatten()
            Q_sqrt = jnp.diag(Q_vec.size * Q_vec / jnp.sum(Q_vec))

            P_sqrt_next = lax.cond(
                self.disable_cov_update,
                cov_update_false,
                cov_update_true,
                P_sqrt_next,
                Q_sqrt,
                gamma_sqrt,
                eps,
            )  # [N*D, N*D]

            next_state = {
                "t": t_next,
                "x": x_next,
                "diffrax_state": diffrax_state_next,
                "eps": eps,
                "P_sqrt": P_sqrt_next[None, :, :],
                "Q_sqrt": state["Q_sqrt"],
                "gamma_sqrt": state["gamma_sqrt"],
                "y": state["y"],
                "y_hat": state["y_hat"],
                "R_sqrt": state["R_sqrt"],
                "S_sqrt": state["S_sqrt"],
            }
            return next_state

        return parametrized_predict

    def build_correct(self) -> FilterCorrect:
        def correct(H: Array, state: Dict[str, Array]) -> Dict[str, Array]:
            def K_zero_true(S_sqrt, H, P_sqrt):
                return jnp.zeros_like(H).T

            def K_zero_false(S_sqrt, H, P_sqrt):
                return (jsp.linalg.cho_solve((S_sqrt, True), H) @ P_sqrt[0] @ P_sqrt[0].T).T

            x, P_sqrt, y, R_sqrt = state["x"], state["P_sqrt"], state["y"], state["R_sqrt"]

            y_hat = H @ x.ravel()  # [L]
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
                "diffrax_state": state["diffrax_state"],
                "eps": state["eps"],
                "P_sqrt": P_sqrt_corrected,
                "Q_sqrt": state["Q_sqrt"],
                "gamma_sqrt": state["gamma_sqrt"],
                "y": state["y"],
                "y_hat": y_hat[None, :],
                "R_sqrt": state["R_sqrt"],
                "S_sqrt": S_sqrt[None, :, :],
            }
            return next_state

        return correct
