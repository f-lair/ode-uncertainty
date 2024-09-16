from typing import Dict

from jax import Array

from src.ode.ode import ODE, ODEBuilder


class Exponential(ODEBuilder):
    """Exponential ODE (first-order)."""

    def __init__(self, growth_factor: float = 1.0) -> None:
        """
        Initialization for exponential growth/decay model.

        Args:
            growth_factor (float, optional): Growth factor. Defaults to 1.0.
        """

        super().__init__(growth_factor=growth_factor)

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
                Array: d/dt State [ N, D].
            """

            return params["growth_factor"] * x  # [N, D]

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

    #     return jnp.exp(self.params[0] * b_t) + b_x0 - 1.0  # [..., T, D]
