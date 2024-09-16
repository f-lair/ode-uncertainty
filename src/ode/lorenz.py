from typing import Dict

from jax import Array
from jax import numpy as jnp

from src.ode.ode import ODE, ODEBuilder


class Lorenz(ODEBuilder):
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

        super().__init__(sigma=sigma, beta=beta, rho=rho)

    def build(self) -> ODE:
        def ode(t: Array, x: Array, params: Dict[str, Array]) -> Array:
            """
            RHS of ODE.
            D=3: Latent dimension.
            N=1: ODE order.

            Args:
                t (Array): Time [].
                x (Array): State [N, D].
                params (Dict[str, Array]): Parameters.

            Returns:
                Array: d/dt State [N, D].
            """

            dx_dt_next = [
                params["sigma"] * (x[:, 1] - x[:, 0]),
                x[:, 0] * (params["rho"] - x[:, 2]) - x[:, 1],
                x[:, 0] * x[:, 1] - params["beta"] * x[:, 2],
            ]

            return jnp.stack(dx_dt_next, axis=-1)  # [N, D]

        return ode
