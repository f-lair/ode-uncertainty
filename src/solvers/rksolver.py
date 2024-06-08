from functools import partial
from typing import Callable, Tuple

import jax

jax.config.update("jax_enable_x64", True)
from jax import Array, lax
from jax import numpy as jnp

from src.ode.ode import ODE


class RKSolver:
    """Abstract base class for explicit embedded Runge-Kutta solvers."""

    # Conservative safety factors to prevent divergent behavior of adaptive step size control
    # cf. E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential Equations I: Nonstiff
    # Problems", Sec. II.4.
    SAFETY_FACTOR = 0.9
    MIN_FACTOR = 0.2
    MAX_FACTOR = 1.5

    def __init__(
        self,
        fn: ODE,
        t0: Array,
        x0: Array,
        step_size: Array | None = None,
        adaptive_control: bool = False,
        max_step_size: float = jnp.inf,
        rtol: float | Array = 1e-6,
        atol: float | Array = 1e-8,
    ) -> None:
        """
        Initialization.
        D: Latent dimension.
        N: ODE order.
        S: Stage.
        ...: Batch dimension(s).

        Args:
            fn (ODE): ODE RHS.
            t0 (Array): Initial time [...].
            x0 (Array): Initial state [..., N, D].
            step_size (Array | None, optional): (Initial) step size. Chosen automatically,
                if None. Defaults to None.
            adaptive_control (bool, optional): Activates adaptive step size control. Defaults to
                False.
            max_step_size (float, optional): Upper bound for step size. Defaults to jnp.inf.
            rtol (float | Array, optional): Relative tolerance in local error estimates for
                adaptive step size control [..., N, D]. Defaults to 1e-3.
            atol (float | Array, optional): Absolute tolerance in local error estimates for
                adaptive step size control [..., N, D]. Defaults to 1e-6.
        """

        # IVP definition
        self.fn = fn
        self.t = t0  # [...]
        self.x = x0  # [..., N, D]

        # Adaptive step size control
        self.adaptive_control = adaptive_control
        self.max_h = max_step_size
        self.rtol = rtol
        self.atol = atol

        # Coefficients
        self.A = self._A()  # [S, S]
        self.b = self._b()  # [2, S]
        self.c = self._c()  # [S]

        # Stage
        self.s = self.A.shape[0]

        # Error exponent
        self.err_exp = -1 / (self._q() + 1)

        # Step size
        if step_size is None:
            self.h = self.initial_step_size()
        else:
            self.h = step_size

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
        Second row is used to compute the actual next state, first row is only for local truncation
        error estimation.
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
    def _q() -> int:
        """
        Specifies order corresponding to the RK method which has lower order.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete RK solver.

        Returns:
            int: Lower order q of the embedded RK solver.
        """

        raise NotImplementedError

    @staticmethod
    def _compute_node(
        fn: ODE,
        ts: Array,
        x: Array,
        h: Array,
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
            ts (Array): Time points to evaluate ODE at [..., S].
            x (Array): State [..., N, D].
            h (Array): Step size h [...].
            A (Array): Coefficient matrix A [S, S].
            idx (int): Node index i.
            ks (Array): Node vectors [..., N, D, S].

        Returns:
            Array: Updated node vectors [..., N, D, S].
        """

        k = fn(ts[idx], x + h[..., None, None] * (ks @ A[idx]))  # [..., N, D]
        return ks.at[..., idx].set(k)  # [..., N, D, S]

    @staticmethod
    @partial(jax.jit, static_argnums=[0, 1])
    def _step(
        compute_node: Callable[..., Array],
        fn: ODE,
        A: Array,
        b: Array,
        c: Array,
        h: Array,
        s: int,
        t: Array,
        x: Array,
    ) -> Array:
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
            c (Array): Coefficient vector c [S].
            h (Array): Step size [...].
            s (int): Stage.
            t (Array): Time [...].
            x (Array): State [..., N, D].

        Returns:
            Array: Next state (embedded solution) [..., N, D, 2].
        """

        ks = jnp.zeros(x.shape + c.shape)  # [..., N, D, S]
        ts = t[..., None] + h[..., None] * c  # [..., S]

        # Iterate over nodes, while reusing previously computed nodes
        compute_node_p = partial(compute_node, fn, ts, x, h, A)
        ks = lax.fori_loop(0, s, compute_node_p, ks)  # [..., N, D, S]

        return x[..., None] + h[..., None, None, None] * (ks @ b.T)  # [..., N, D, 2]

    @staticmethod
    @jax.jit
    def _eps_scale(x: Array, atol: float | Array, rtol: float | Array) -> Tuple[Array, Array]:
        """
        Computes local truncation error and error scale.
        D: Latent dimension.
        N: ODE order.
        ...: Batch dimension(s).

        Args:
            x (Array): State (embedded solution) [..., N, D, 2].
            atol (float | Array): Relative error tolerance [..., N, D].
            rtol (float | Array): Absolute error tolerance [..., N, D].

        Returns:
            Tuple[Array, Array]: Local truncation error [..., N, D], error scale [..., N, D].
        """

        eps = jnp.abs((x[..., 0] - x[..., 1]))  # [..., N, D]
        scale = atol + rtol * jnp.maximum(jnp.abs(x[..., 0]), jnp.abs(x[..., 1]))  # [..., N, D]

        return eps, scale

    @staticmethod
    @jax.jit
    def _L2_error(x: Array, scale: Array) -> Array:
        """
        Computes error in L2-norm.
        D: Latent dimension.
        N: ODE order.
        ...: Batch dimension(s).

        Args:
            x (Array): State (embedded solution) [..., N, D, 2].
            scale (Array): Error scale [..., N, D].

        Returns:
            Array: L2 error [...].
        """

        ND = x.shape[-3] * x.shape[-2]
        return jnp.sqrt(
            1 / ND * jnp.sum(((x[..., 0] - x[..., 1]) / scale) ** 2, axis=(-1, -2))
        )  # [...]

    def initial_step_size(self) -> Array:
        """
        Determines initial step size heuristically [1].
        ...: Batch dimension(s).

        [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential Equations I:
            Nonstiff Problems", Sec. II.4.

        Returns:
            Array: Initial step size [...].
        """

        l2_norm = lambda x, scale: jnp.sqrt(
            1 / (x.shape[-2] * x.shape[-1]) * jnp.sum((x / scale) ** 2, axis=(-1, -2))
        )

        scale = self.atol + jnp.abs(self.x) * self.rtol  # [..., N, D]
        x_prime = self.fn(self.t, self.x)  # [..., N, D]

        d0 = l2_norm(self.x, scale)  # [...]
        d1 = l2_norm(x_prime, scale)  # [...]

        # First guess for initial step size
        h0 = jnp.where(jnp.logical_or(d0 < 1e-5, d1 < 1e-5), 1e-6, 0.01 * d0 / d1)  # [...]

        # Explicit Euler step
        x_next = self.x + h0 * x_prime  # [..., N, D]
        x_prime_next = self.fn(self.t + h0, x_next)  # [..., N, D]

        # Estimate of second derivative
        d2 = l2_norm(x_prime_next - x_prime, scale) / h0  # [...]

        # Final initial step size
        h1 = jnp.where(
            jnp.logical_and(d1 <= 1e-15, d2 <= 1e-15),
            jnp.maximum(1e-6, 1e-3 * h0),
            (0.01 / jnp.maximum(d1, d2)) ** (-self.err_exp),
        )  # [...]
        return jnp.minimum(100 * h0, h1)  # [...]

    def step(self) -> Tuple[Array, Array, Array]:
        """
        Performs single integration step and adapts step size [1], if needed.
        D: Latent dimension.
        N: ODE order.
        ...: Batch dimension(s).

        [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential Equations I:
            Nonstiff Problems", Sec. II.4.

        Args:
            t (Array): Time [...].
            x (Array): State [..., N, D].

        Returns:
            Tuple[Array, Array, Array]: Next time [...], next state [..., N, D],
                local truncation error [..., N, D].
        """

        t_next = self.t + self.h  # [...]

        if self.adaptive_control:
            # Initially, no step is accepted or rejected, since no step has happened yet
            step_accepted = jnp.zeros(self.x.shape[:-2], dtype=jnp.bool)  # [...]
            step_rejected = jnp.zeros_like(step_accepted)  # [...]

            # Loop until all batched steps are accepted
            while not jnp.all(step_accepted):
                # Perform step
                x_next = self._step(
                    self._compute_node,
                    self.fn,
                    self.A,
                    self.b,
                    self.c,
                    self.h,
                    self.s,
                    self.t,
                    self.x,
                )  # [..., N, D, 2]

                # Compute local truncation error
                eps, scale = self._eps_scale(
                    x_next, self.atol, self.rtol
                )  # [..., N, D], [..., N, D]
                err = self._L2_error(x_next, scale)  # [...]

                # Determine scaling factor for adaptive step size control based on error
                factor = jnp.minimum(
                    self.MAX_FACTOR,
                    jnp.maximum(self.MIN_FACTOR, self.SAFETY_FACTOR * err**self.err_exp),
                )  # [...]
                # After rejection, the factor should be upper bounded by 1
                factor = jnp.where(step_rejected, jnp.minimum(1.0, factor), factor)  # [...]

                # Rescale step size
                self.h = jnp.minimum(
                    self.max_h, jnp.where(step_accepted, self.h, factor * self.h)
                )  # [...]

                # Update step acception and rejection flags
                step_accepted = jnp.where(
                    jnp.logical_or(step_accepted, err <= 1.0), True, False
                )  # [...]
                step_rejected = jnp.where(
                    jnp.logical_and(~step_accepted, err > 1.0), True, False
                )  # [...]

                # Update next time point
                t_next = jnp.where(step_accepted, t_next, self.t + self.h)  # [...]
        else:
            x_next = self._step(
                self._compute_node, self.fn, self.A, self.b, self.c, self.h, self.s, self.t, self.x
            )  # [..., N, D, 2]

            eps, _ = self._eps_scale(x_next, self.atol, self.rtol)  # [..., N, D]

        # Update internal state
        self.t = t_next  # [...]
        self.x = x_next[..., 1]  # [..., N, D]

        return self.t, self.x, eps  # [...], [..., N, D], [..., N, D]
