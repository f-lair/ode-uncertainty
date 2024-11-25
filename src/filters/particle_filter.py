from typing import Dict

import jax
from jax import Array
from jax import numpy as jnp
from jax import random, tree

from src.covariance_update_functions import (
    DiagonalCovarianceUpdate,
    StaticDiagonalCovarianceUpdate,
)
from src.covariance_update_functions.covariance_update_function import (
    CovarianceUpdateFunction,
    CovarianceUpdateFunctionBuilder,
)
from src.covariance_update_functions.static_covariance_update_function import (
    StaticCovarianceUpdateFunction,
    StaticCovarianceUpdateFunctionBuilder,
)
from src.filters.filter import FilterBuilder, FilterPredict
from src.solvers.solver import Solver


class ParticleFilter(FilterBuilder):
    """Particle Filter."""

    def __init__(
        self,
        cov_update_fn_builder: CovarianceUpdateFunctionBuilder = DiagonalCovarianceUpdate(),
        static_cov_update_fn_builder: StaticCovarianceUpdateFunctionBuilder = StaticDiagonalCovarianceUpdate(),
        num_particles: int = 100,
    ) -> None:
        super().__init__(cov_update_fn_builder, static_cov_update_fn_builder)
        self.M = num_particles

    def init_state(self, solver_state: Dict[str, Array], prng_key: Array) -> Dict[str, Array]:
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

        filter_state["t"] = jnp.broadcast_to(filter_state["t"][None], (self.M))
        filter_state["x"] = jnp.broadcast_to(
            filter_state["x"][None, :, :], (self.M,) + filter_state["x"].shape
        )
        filter_state["eps"] = jnp.broadcast_to(
            filter_state["eps"][None, :, :], (self.M,) + filter_state["eps"].shape
        )
        filter_state["diffrax_state"] = tree.map(
            lambda _x: jnp.broadcast_to(_x[None, ...], (self.M,) + _x.shape),
            filter_state["diffrax_state"],
        )
        filter_state["prng_key"] = prng_key

        return filter_state

    def build_cov_update_fn(self) -> CovarianceUpdateFunction:
        return jax.vmap(self.cov_update_fn_builder.build())

    def build_static_cov_update_fn(self) -> StaticCovarianceUpdateFunction:
        return jax.vmap(self.static_cov_update_fn_builder.build(), in_axes=(None, 0, 0))

    def build_predict(self) -> FilterPredict:
        def predict(
            solver: Solver, cov_update_fn: CovarianceUpdateFunction, state: Dict[str, Array]
        ) -> Dict[str, Array]:
            t, x, diffrax_state, prng_key = (
                state["t"],
                state["x"],
                state["diffrax_state"],
                state["prng_key"],
            )
            solver_state = {"t": t, "x": x, "diffrax_state": diffrax_state}
            M, N, D = x.shape
            prng_key, prng_key_next = random.split(prng_key)

            next_solver_state = solver(solver_state)
            t_next = next_solver_state["t"]  # [M]
            x_next = next_solver_state["x"]  # [M, N, D]
            eps = next_solver_state["eps"]  # [M, N, D]
            diffrax_state_next = next_solver_state["diffrax_state"]

            p = (
                random.multivariate_normal(
                    prng_key,
                    jnp.zeros((M, N * D)),
                    cov_update_fn(
                        jnp.zeros((M, N * D, N * D)),
                        eps.reshape(M, N * D),
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
                "eps": eps,
                "diffrax_state": diffrax_state_next,
                "prng_key": prng_key_next,
            }
            return next_state

        return predict
