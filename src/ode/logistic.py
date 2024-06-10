from jax import Array
from jax import numpy as jnp

from src.ode.ode import ODE


class Logistic(ODE):
    """Logistic ODE (first-order)."""

    def __init__(
        self, growth_rate: Array = jnp.array(1.0), carrying_capacity: Array = jnp.array(1.0)
    ) -> None:
        """
        Initialization for population-growth model.
        ...: Batch dimension(s).

        Args:
            growth_rate (Array, optional): Population growth rate [...]. Defaults to
                jnp.array(1.0).
            carrying_capacity (Array, optional): Maximum population size [...]. Defaults to
                jnp.array(1.0).
        """

        self.r = growth_rate
        self.K = carrying_capacity

    def fn(self, t: Array, x: Array) -> Array:
        """
        RHS of ODE.
        D=1: Latent dimension.
        N=1: ODE order.
        ...: Batch dimension(s).

        Args:
            t (Array): Time [...].
            x (Array): State [..., N, D].

        Returns:
            Array: d/dt State [..., N, D].
        """

        return self.r * x * (1 - x / self.K)  # [..., N, D]

    def Fn(self, t: Array, x0: Array) -> Array:
        """
        Analytic ODE solution.
        D=1: Latent dimension.
        N=1: ODE order.
        T: Time dimension.
        ...: Batch dimension(s).

        Args:
            t (Array): Time [..., T].
            x0 (Array): Initial state [..., N, D].

        Returns:
            Array: Function value [..., T, D].
        """

        b_shape = t.shape + x0.shape[-1:]
        b_x0 = jnp.broadcast_to(x0[..., 0, :], b_shape)  # [..., T, D]
        b_t = jnp.broadcast_to(t[..., None], b_shape)  # [..., T, D]

        return self.K / (1.0 + ((self.K - b_x0) / b_x0) * jnp.exp(-self.r * b_t))  # [..., T, D]
