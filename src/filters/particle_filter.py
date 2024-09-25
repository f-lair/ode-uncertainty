from typing import Dict, Tuple

import jax
from jax import Array
from jax import numpy as jnp
from jax import random

from src.covariance_update_functions import DiagonalCovarianceUpdate
from src.covariance_update_functions.covariance_update_function import (
    CovarianceUpdateFunction,
    CovarianceUpdateFunctionBuilder,
)
from src.filters.filter import FilterBuilder, FilterPredict
from src.solvers.solver import Solver


class ParticleFilter(FilterBuilder):
    """Particle Filter."""

    def __init__(
        self,
        cov_update_fn_builder: CovarianceUpdateFunctionBuilder = DiagonalCovarianceUpdate(),
        num_particles: int = 100,
    ) -> None:
        super().__init__(cov_update_fn_builder)
        self.M = num_particles

    def state_def(self, N: int, D: int, L: int) -> Dict[str, Tuple[int, ...]]:
        """
        Defines the solver state.

        Args:
            N (int): ODE order.
            D (int): Latent dimension.
            L (int): Measurement dimension.

        Returns:
            Dict[str, Tuple[int, ...]]: State definition.
        """

        return {"t": (self.M,), "x": (self.M, N, D), "Q": (N * D, N * D), "prng_key": ()}

    def build_cov_update_fn(self) -> CovarianceUpdateFunction:
        return jax.vmap(self.cov_update_fn_builder.build())

    def build_predict(self) -> FilterPredict:
        def predict(
            solver: Solver, cov_update_fn: CovarianceUpdateFunction, state: Dict[str, Array]
        ) -> Dict[str, Array]:
            t, x, Q, prng_key = state["t"], state["x"], state["Q"], state["prng_key"]
            solver_state = {"t": t, "x": x}
            M, N, D = x.shape
            prng_key, prng_key_next = random.split(prng_key)

            next_solver_state = solver(solver_state)
            t_next = next_solver_state["t"]  # [M]
            x_next = next_solver_state["x"]  # [M, N, D]
            eps = next_solver_state["eps"]  # [M, N, D]

            p = (
                random.multivariate_normal(
                    prng_key,
                    jnp.zeros((M, N * D)),
                    cov_update_fn(
                        jnp.broadcast_to(Q[None, :, :], (M, N * D, N * D)), eps.reshape(M, N * D)
                    ),
                    method="svd",
                )
                .reshape(M, N, D)
                .at[0]
                .set(0.0)
            )  # [M, N, D]
            x_next = x_next + p  # [M, N, D]

            next_state = {
                "t": t_next,
                "x": x_next,
                "Q": state["Q"],
                "prng_key": prng_key_next,
            }
            return next_state

        return predict
