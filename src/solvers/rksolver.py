from functools import partial
from typing import Callable, Tuple

import jax

jax.config.update("jax_enable_x64", True)
from jax import Array, lax
from jax import numpy as jnp

from src.ode.ode import ODE


class RKSolver:
    """Abstract base class for explicit embedded Runge-Kutta solvers."""

    def __init__(self, step_size: float) -> None:
        """
        Initialization.

        Args:
            step_size (float): Step size h.
        """

        self.h = step_size

        # Coefficients
        self.A = self._A()  # [S, S]
        self.b = self._b()  # [2, S]
        self.hc = self.h * self._c()  # [S]

        # Stage
        self.s = self.A.shape[0]

    @staticmethod
    def _A() -> Array:
        """
        Defines coefficient matrix A in the Butcher tableau.
        S: Stage.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete RK solver.

        Returns:
            Array: Coefficient matrix A [S, S].
        """

        raise NotImplementedError

    @staticmethod
    def _b() -> Array:
        """
        Defines coefficient matrix b in the Butcher tableau.
        First row should be higher-order than second row.
        S: Stage.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete RK solver.

        Returns:
            Array: Coefficient matrix b [2, S].
        """

        raise NotImplementedError

    @staticmethod
    def _c() -> Array:
        """
        Defines coefficient vector c in the Butcher tableau.
        S: Stage.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete RK solver.

        Returns:
            Array: Coefficient vector c [S].
        """

        raise NotImplementedError

    @staticmethod
    def _compute_node(
        fn: ODE,
        ts: Array,
        x: Array,
        h: float,
        A: Array,
        idx: int,
        ks: Array,
    ) -> Array:
        """
        Computes single node k_i.
        D: Latent dimension.
        N: ODE order.
        S: Stage.
        ...: Batch dimension(s).

        Args:
            fn (ODE): ODE RHS.
            ts (Array): Time points to evaluate ODE at [S].
            x (Array): State [..., N, D].
            h (float): Step size h.
            A (Array): Coefficient matrix A [S, S].
            idx (int): Node index i.
            ks (Array): Node vector [S].

        Returns:
            Array: Updated node vector [..., N, D, S].
        """

        k = fn(ts[idx], x + jnp.broadcast_to(h * ks @ A[idx], x.shape))  # [..., N, D]
        return ks.at[..., idx].set(k)  # [..., N, D, S]

    @staticmethod
    @partial(jax.jit, static_argnums=[0, 1])
    def _step(
        compute_node: Callable[..., Array],
        fn: ODE,
        A: Array,
        b: Array,
        hc: Array,
        h: float,
        s: int,
        t: Array,
        x: Array,
    ) -> Tuple[Array, Array]:
        """
        Jitted step function of RK solver.
        D: Latent dimension.
        N: ODE order.
        S: Stage.
        ...: Batch dimension(s).

        Args:
            compute_node (Callable[..., Array]): Subroutine for single node computation.
            fn (ODE): ODE RHS.
            A (Array): Coefficient matrix A [S, S].
            b (Array): Coefficient matrix b [2, S].
            hc (Array): Step size-adjusted coefficient vector c [S].
            h (float): Step size.
            s (int): Stage.
            t (Array): Time [...].
            x (Array): State [..., N, D].

        Returns:
            Tuple[Array, Array]: Next state [..., N, D], local truncation error [..., N, D].
        """

        ks = jnp.zeros(x.shape + hc.shape)  # [..., N, D, S]
        ts = t + hc  # [S]

        compute_node_p = partial(compute_node, fn, ts, x, h, A)

        ks = lax.fori_loop(0, s, compute_node_p, ks)  # [..., N, D, S]
        x_next = x + h * ks @ b[1]  # [..., N, D]
        eps = x + h * ks @ jnp.abs((b[1] - b[0]))  # [..., N, D]

        return x_next, eps

    def step(self, fn: ODE, t: Array, x: Array) -> Tuple[Array, Array]:
        """
        Performs single integration step.
        D: Latent dimension.
        N: ODE order.
        ...: Batch dimension(s).

        Args:
            fn (ODE): ODE RHS.
            t (Array): Time [...].
            x (Array): State [..., N, D].

        Returns:
            Tuple[Array, Array]: Next state [..., N, D], local truncation error [..., N, D].
        """

        return self._step(self._compute_node, fn, self.A, self.b, self.hc, self.h, self.s, t, x)
