from typing import Callable, Dict, Tuple

from jax import Array
from jax import scipy as jsp

from src.covariance_functions.covariance_function import CovarianceFunction
from src.filters.filter import FilterBuilder, FilterCorrect, FilterPredict
from src.solvers.solver import Solver
from src.utils import const_diag, value_and_jacfwd


class EKF(FilterBuilder):
    """Extended Kalman Filter."""

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
            "P": (1, N * D, N * D),
            "Q": (N * D, N * D),
            "y": (L,),
            "y_hat": (1, L),
            "R": (L, L),
            "S": (1, L, L),
        }

    def build_cov_fn(self) -> CovarianceFunction:
        return self.cov_fn_builder.build()

    def build_predict(self) -> FilterPredict:
        def predict(
            solver: Solver, cov_fn: CovarianceFunction, state: Dict[str, Array]
        ) -> Dict[str, Array]:
            t, x, P, Q = state["t"], state["x"], state["P"], state["Q"]
            solver_state = {"t": t, "x": x}

            next_solver_state, solver_jacs = value_and_jacfwd(solver, solver_state)

            t_next = next_solver_state["t"]  # [1]
            x_next = next_solver_state["x"]  # [1, N, D]
            eps = next_solver_state["eps"]  # [1, N, D]
            jac = solver_jacs["x"]["x"].reshape(x.size, x.size)  # [N*D, N*D]

            P_next = (jac @ P[0] @ jac.T + Q + cov_fn(eps.ravel()))[None, :, :]  # [1, N*D, N*D]

            next_state = {
                "t": t_next,
                "x": x_next,
                "P": P_next,
                "Q": state["Q"],
                "y": state["y"],
                "y_hat": state["y_hat"],
                "R": state["R"],
                "S": state["S"],
            }
            return next_state

        return predict

    def build_correct(self) -> FilterCorrect:
        def correct(
            measurement_fn: Callable[[Array], Array], state: Dict[str, Array]
        ) -> Dict[str, Array]:
            x, P, y, R = state["x"], state["P"], state["y"], state["R"]

            y_hat, H = value_and_jacfwd(measurement_fn, x.ravel())  # [L], [L, N*D]
            y_delta = y - y_hat  # [L]

            S = (H @ P[0] @ H.T) + R + const_diag(R.shape[-1], 1e-8)  # [L, L]
            S_cho = jsp.linalg.cho_factor(S, lower=True)  # [L, L]
            K = jsp.linalg.cho_solve(S_cho, H @ P[0]).T  # [N*D, L]

            x_corrected = x + (K @ y_delta).reshape(*x.shape)  # [1, N, D]
            P_corrected = P - (K @ S @ K.T)[None, :, :]  # [1, N*D, N*D]

            next_state = {
                "t": state["t"],
                "x": x_corrected,
                "P": P_corrected,
                "Q": state["Q"],
                "y": state["y"],
                "y_hat": y_hat[None, :],
                "R": state["R"],
                "S": S[None, :, :],
            }
            return next_state

        return correct
