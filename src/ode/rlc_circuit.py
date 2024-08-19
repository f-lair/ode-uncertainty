from jax import Array
from jax import numpy as jnp

from src.ode.ode import ODE


class RLCCircuit(ODE):
    """RLC circuit ODE (second-order)."""

    def __init__(
        self,
        resistance: float = 1.0,
        inductance: float = 1.0,
        capacitance: float = 1.0,
    ) -> None:
        """
        Initialization for RLC circuit model.

        Args:
            resistance (float, optional): Resistance R (Ohm). Defaults to 1.0.
            inductance (float, optional): Inductance L (Henry). Defaults to 1.0.
            capacitance (float, optional): Capacitance C (Farad). Defaults to 1.0.
        """

        self.R = jnp.array(resistance)
        self.L = jnp.array(inductance)
        self.C = jnp.array(capacitance)

        self.delta = 0.5 * self.R / self.L
        self.omega0 = (self.L * self.C) ** (-0.5)
        self.omega = (self.omega0**2 - self.delta**2) ** 0.5
        self.lambda_ = (self.delta**2 - self.omega0**2) ** 0.5

    def fn(self, t: Array, x: Array) -> Array:
        """
        RHS of ODE.
        D=1: Latent dimension.
        N=2: ODE order.
        ...: Batch dimension(s).

        Args:
            t (Array): Time [...].
            x (Array): State [..., N, D].

        Returns:
            Array: d/dt State [..., N, D].
        """

        x_prev = x[..., 0, :]  # [..., D]
        dx_dt_prev = x[..., 1, :]  # [..., D]

        dx_dt_next = dx_dt_prev  # [..., D]
        d2x_dt2_next = -self.R / self.L * dx_dt_prev - 1 / (self.L * self.C) * x_prev  # [..., D]

        return jnp.stack([dx_dt_next, d2x_dt2_next], axis=-2)  # [..., N, D]

    def Fn(self, t: Array, x0: Array) -> Array:
        """
        Analytic ODE solution.
        D=1: Latent dimension.
        N=2: ODE order.
        T: Time dimension.
        ...: Batch dimension(s).

        Args:
            t (Array): Time [..., T].
            x0 (Array): Initial state [N, D].

        Returns:
            Array: Function value [..., T, D].
        """

        b_shape = t.shape + x0.shape[-1:]
        b_x0 = jnp.broadcast_to(x0[..., 0, :], b_shape)  # [..., T, D]
        b_t = jnp.broadcast_to(t[..., None], b_shape)  # [..., T, D]

        # Underdamped
        if self.omega0**2 - self.delta**2 > 1e-6:
            return (
                b_x0
                * (jnp.cos(self.omega * b_t) + self.delta / self.omega * jnp.sin(self.omega * b_t))
                * jnp.exp(-self.delta * b_t)
            )  # [..., T, D]
        # Overdamped
        elif self.delta**2 - self.omega0**2 > 1e-6:
            return (
                0.5
                * b_x0
                / self.lambda_
                * (
                    (self.lambda_ + self.delta) * jnp.exp(self.lambda_ * b_t)
                    + (self.lambda_ - self.delta) * jnp.exp(-self.lambda_ * b_t)
                )
                * jnp.exp(-self.delta * b_t)
            )  # [..., T, D]
        # Critically damped
        else:
            return b_x0 * (1.0 + self.delta * b_t) * jnp.exp(-self.delta * b_t)  # [..., T, D]
