from jax import Array
from jax import numpy as jnp

from src.ode.ode import ODE


class Exponential(ODE):
    """Exponential ODE (first-order)."""

    def __init__(self, growth_factor: Array = jnp.array(1.0)) -> None:
        """
        Initialization for exponential growth/decay model.
        ...: Batch dimension(s).

        Args:
            growth_factor (Array, optional): Growth factor [...]. Defaults to
                jnp.array(1.0).
        """

        self.a = growth_factor

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

        return self.a * x  # [..., N, D]

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

        return jnp.exp(self.a * b_t) + b_x0 - 1.0  # [..., T, D]
