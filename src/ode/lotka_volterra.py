import numpy as np
from jax import Array
from jax import numpy as jnp

from src.ode.ode import ODE


class LotkaVolterra(ODE):
    """Lotka-Volterra ODE (first-order)."""

    def __init__(
        self,
        alpha: float = 1.5,
        beta: float = 1.0,
        gamma: float = 3.0,
        delta: float = 1.0,
    ) -> None:
        """
        Initialization for Lotka-Volterra ODE.

        Args:
            alpha (float, optional): Coefficient alpha. Defaults to 1.5.
            beta (float, optional): Coefficient beta. Defaults to 1.0.
            gamma (float, optional): Coefficient gamma. Defaults to 3.0.
            delta (float, optional): Coefficient delta. Defaults to 1.0.
        """

        self.alpha = jnp.array(alpha)
        self.beta = jnp.array(beta)
        self.gamma = jnp.array(gamma)
        self.delta = jnp.array(delta)

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
            self.alpha * x[..., :, 0] - self.beta * x[..., :, 0] * x[..., :, 1],
            -self.gamma * x[..., :, 1] + self.delta * x[..., :, 0] * x[..., :, 1],
        ]  # [..., N, D]

        return jnp.stack(dx_dt_next, axis=-1)  # [..., N, D]
