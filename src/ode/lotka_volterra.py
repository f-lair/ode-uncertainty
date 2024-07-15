from jax import Array
from jax import numpy as jnp

from src.ode.ode import ODE


class LotkaVolterra(ODE):
    """Lotka-Volterra ODE (first-order)."""

    def __init__(
        self,
        alpha: Array = jnp.array(1.5),
        beta: Array = jnp.array(1.0),
        gamma: Array = jnp.array(3.0),
        delta: Array = jnp.array(1.0),
    ) -> None:
        """
        Initialization for Lotka-Volterra ODE.

        Args:
            alpha (Array, optional): Coefficient alpha. Defaults to jnp.array(1.5).
            beta (Array, optional): Coefficient beta. Defaults to jnp.array(1.0).
            gamma (Array, optional): Coefficient gamma. Defaults to jnp.array(3.0).
            delta (Array, optional): Coefficient delta. Defaults to jnp.array(1.0).
        """

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def fn(self, t: Array, x: Array) -> Array:
        """
        RHS of ODE.
        D=2: Latent dimension.
        N=1: ODE order.
        ...: Batch dimension(s).

        Args:
            t (Array): Time [...].
            x (Array): State [..., N, D].

        Returns:
            Array: d/dt State [..., N, D].
        """

        dx_dt_next = [
            self.alpha * x[..., 0, 0] - self.beta * x[..., 0, 0] * x[..., 0, 1],
            -self.gamma * x[..., 0, 1] + self.delta * x[..., 0, 0] * x[..., 0, 1],
        ]  # [..., N, D]

        return jnp.stack(dx_dt_next, axis=-1)  # [..., N, D]
