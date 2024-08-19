from jax import Array
from jax import numpy as jnp

from src.ode.ode import ODE


class VanDerPol(ODE):
    """Van der Pol ODE (second-order)."""

    def __init__(self, damping: float = 5.0) -> None:
        """
        Initialization for Van der Pol oscillator.

        Args:
            damping (float, optional): Nonlinearity and strength of the damping. Defaults to 5.0.
        """

        self.mu = jnp.array(damping)

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
        d2x_dt2_next = self.mu * (1 - x_prev**2) * dx_dt_prev - x_prev  # [..., D]

        return jnp.stack([dx_dt_next, d2x_dt2_next], axis=-2)  # [..., N, D]
