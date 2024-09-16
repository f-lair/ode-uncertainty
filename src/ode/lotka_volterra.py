from typing import Dict

from jax import Array
from jax import numpy as jnp

from src.ode.ode import ODE, ODEBuilder


class LotkaVolterra(ODEBuilder):
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

        super().__init__(alpha=alpha, beta=beta, gamma=gamma, delta=delta)

    def build(self) -> ODE:
        def ode(t: Array, x: Array, params: Dict[str, Array]) -> Array:
            """
            RHS of ODE.
            D=2: Latent dimension.
            N=1: ODE order.

            Args:
                t (Array): Time [].
                x (Array): State [N, D].
                params (Dict[str, Array]): Parameters.

            Returns:
                Array: d/dt State [N, D].
            """

            dx_dt_next = [
                params["alpha"] * x[:, 0] - params["beta"] * x[:, 0] * x[:, 1],
                -params["gamma"] * x[:, 1] + params["delta"] * x[:, 0] * x[:, 1],
            ]  # [N, D]

            return jnp.stack(dx_dt_next, axis=-1)  # [N, D]

        return ode
