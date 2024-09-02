from functools import partial
from typing import Callable, Tuple

import jax
from jax import Array
from jax import numpy as jnp
from jax import random

from src.filters.filter import Filter
from src.filters.perturbation_fns import PerturbationFn
from src.filters.sigma_fns import SigmaFn
from src.ode.ode import ODE
from src.solvers.rksolver import RKSolver


class ParticleFilter(Filter):
    """Particle Filter."""

    def __init__(self, num_particles: int = 100, seed: int = 7) -> None:
        self.num_particles = num_particles
        self.prng_key = random.key(seed)  # []

    def setup(
        self, rk_solver: RKSolver, P0: Array, perturbation_fn: PerturbationFn, sigma_fn: SigmaFn
    ) -> None:
        """
        Setups filter.
        D: Latent dimension.
        N: ODE order.

        Args:
            rk_solver (RKSolver): RK solver.
            P0 (Array): Initial covariance [N*D, N*D].
            sigma_fn (SigmaFn): Sigma function.
        """

        self.rk_solver = rk_solver
        self.t = rk_solver.t0
        self.m = jnp.broadcast_to(rk_solver.x0, (self.num_particles + 1,) + rk_solver.x0.shape[1:])
        self._P = P0
        self.perturbation_fn = perturbation_fn
        self.sigma_fn_vmap = jax.vmap(sigma_fn)

    def batch_dim(self) -> int:
        """
        Batch dimension.

        Returns:
            int: Batch dimension.
        """

        return self.num_particles + 1

    @staticmethod
    @partial(jax.jit, static_argnums=[0, 1, 2, 3])
    def _predict_jit(
        step_fn: Callable[[Array, Array], Tuple[Array, Array, Array, Array]],
        ode: ODE,
        perturbation_fn: PerturbationFn,
        sigma_fn_vmap: SigmaFn,
        t: Array,
        x: Array,
        prng_key: Array,
    ) -> Tuple[Array, Array, Array, Array]:
        """
        Jitted predict function of particle filter.
        D: Latent dimension.
        M: Number of particles.
        N: ODE order.

        Args:
            step_fn (Callable[[Array, Array], Tuple[Array, Array, Array, Array]]):
                RK-solver step function.
            sigma_fn (SigmaFn): Sigma function.
            t (Array): Time [1].
            x (Array): Particles [M+1, N, D].

        Returns:
            Tuple[Array, Array, Array]: Time [1], particles [M, N, D], sigma [M, N*D, N*D].
        """

        dx_dts = ode(t, x)  # [M, N, D]
        t_next, x_next, eps, _ = step_fn(t, x)  # [M], [M, N, D], [M, N, D]
        p, sigma = perturbation_fn(
            sigma_fn_vmap, dx_dts, eps, prng_key
        )  # [M, N, D], [M, N*D, N*D]
        p.at[0].set(0.0)
        x_next = x_next + p  # [M, N, D]

        return t_next[0:1], x_next, dx_dts, sigma

    def _predict(self) -> Tuple[Array, Array, Array, Array]:
        """
        Predicts state after performing one step of the ODE solver.
        D: Latent dimension.
        M: Number of particles.
        N: ODE order.

        Returns:
            Tuple[Array, Array, Array, Array]: Time [1], particles [M, N, D], particles derivative
                [M, N, D], sigma [M, N*D, N*D].
        """

        self.prng_key, split_prng_key = random.split(self.prng_key)
        self.t, self.m, dx_dts, sigma = self._predict_jit(
            self.rk_solver.step,
            self.rk_solver.fn,
            self.perturbation_fn,
            self.sigma_fn_vmap,
            self.t,
            self.m,
            split_prng_key,
        )

        return self.t, self.m, dx_dts, sigma

    @staticmethod
    def results_spec() -> Tuple[str, ...]:
        """
        Results specification.

        Returns:
            Tuple[str, ...]: Results keys.
        """

        return "ts", "xs", "dx_dts", "Sigmas"
