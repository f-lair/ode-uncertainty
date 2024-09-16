from typing import Dict

from jax import Array

from src.ode.ode import ODE, ODEBuilder


class Logistic(ODEBuilder):
    """Logistic ODE (first-order)."""

    def __init__(self, growth_rate: float = 1.0, carrying_capacity: float = 1.0) -> None:
        """
        Initialization for population-growth model.

        Args:
            growth_rate (float, optional): Population growth rate. Defaults to 1.0.
            carrying_capacity (float, optional): Maximum population size. Defaults to 1.0.
        """

        super().__init__(growth_rate=growth_rate, carrying_capacity=carrying_capacity)

    def build(self) -> ODE:
        def ode(t: Array, x: Array, params: Dict[str, Array]) -> Array:
            """
            RHS of ODE.
            D=1: Latent dimension.
            N=1: ODE order.

            Args:
                t (Array): Time [].
                x (Array): State [N, D].
                params (Dict[str, Array]): Parameters.

            Returns:
                Array: d/dt State [N, D].
            """

            return params["growth_rate"] * x * (1 - x / params["carrying_capacity"])  # [N, D]

        return ode

    # def Fn(self, t: Array, x0: Array) -> Array:
    #     """
    #     Analytic ODE solution.
    #     D=1: Latent dimension.
    #     N=1: ODE order.
    #     T: Time dimension.
    #     ...: Batch dimension(s).

    #     Args:
    #         t (Array): Time [..., T].
    #         x0 (Array): Initial state [..., N, D].

    #     Returns:
    #         Array: Function value [..., T, D].
    #     """

    #     b_shape = t.shape + x0.shape[-1:]
    #     b_x0 = jnp.broadcast_to(x0[..., 0, :], b_shape)  # [..., T, D]
    #     b_t = jnp.broadcast_to(t[..., None], b_shape)  # [..., T, D]

    #     return self.params[1] / (
    #         1.0 + ((self.params[1] - b_x0) / b_x0) * jnp.exp(-self.params[0] * b_t)
    #     )  # [..., T, D]
