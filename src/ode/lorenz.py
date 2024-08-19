from jax import Array
from jax import numpy as jnp

from src.ode.ode import ODE


class Lorenz(ODE):
    """Lorenz ODE (first-order)."""

    def __init__(
        self,
        sigma: float = 10.0,
        beta: float = 8.0 / 3,
        rho: float = 28.0,
    ) -> None:
        """
        Initialization for Lorenz system.

        Args:
            sigma (float, optional): Parameter proportional to Prandtl number. Defaults to 10.0.
            beta (float, optional): Parameter proportional to Rayleigh number. Defaults to 8.0 / 3.
            rho (float, optional): Parameter proportional to certain physical dimensions of the
                layer itself. Defaults to 28.0.
        """

        self.sigma = jnp.array(sigma)
        self.beta = jnp.array(beta)
        self.rho = jnp.array(rho)

    def fn(self, t: Array, x: Array) -> Array:
        """
        RHS of ODE.
        D=3: Latent dimension.
        N=1: ODE order.
        ...: Batch dimension(s).

        Args:
            t (Array): Time [...].
            x (Array): State [..., N, D].

        Returns:
            Array: d/dt State [..., N, D].
        """

        dx_dt_next = [
            self.sigma * (x[..., 1] - x[..., 0]),
            x[..., 0] * (self.rho - x[..., 2]) - x[..., 1],
            x[..., 0] * x[..., 1] - self.beta * x[..., 2],
        ]

        return jnp.stack(dx_dt_next, axis=-1)  # [..., N, D]
