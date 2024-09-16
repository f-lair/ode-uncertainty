from typing import Dict

from jax import Array
from jax import numpy as jnp

from src.ode.ode import ODE, ODEBuilder


class Pendulum(ODEBuilder):
    """Pendulum ODE (second-order)."""

    def __init__(self, length: float = 3.0) -> None:
        """
        Initialization for pendulum.

        Args:
            length (float, optional): Pendulum length in m. Defaults to 3.0.
        """

        super().__init__(length=length)

    def build(self) -> ODE:
        def ode(t: Array, x: Array, params: Dict[str, Array]) -> Array:
            """
            RHS of ODE.
            D=1: Latent dimension.
            N=2: ODE order.

            Args:
                t (Array): Time [].
                x (Array): State [N, D].
                params (Dict[str, Array]): Parameters.

            Returns:
                Array: d/dt State [N, D].
            """

            x_prev = x[0]  # [D]
            dx_dt_prev = x[1]  # [D]

            dx_dt_next = dx_dt_prev  # [D]
            d2x_dt2_next = -9.81 / params["length"] * jnp.sin(x_prev)  # [D]

            return jnp.stack([dx_dt_next, d2x_dt2_next], axis=-2)  # [N, D]

        return ode
