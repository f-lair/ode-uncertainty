from jax import Array


class ODE:
    """Abstract base class for explicit order-N ODEs."""

    def fn(self, t: Array, x: Array) -> Array:
        """
        RHS of ODE.
        D: Latent dimension.
        N: ODE order.
        ...: Batch dimension(s).

        Args:
            t (Array): Time [...].
            x (Array): State [..., N, D].

        Raises:
            NotImplementedError: Needs to be implemented for a concrete ODE.

        Returns:
            Array: d/dt State [..., N, D].
        """

        raise NotImplementedError

    def __call__(self, t: Array, x: Array) -> Array:
        """
        Shorthand call for RHS of ODE.

        D: Latent dimension.
        N: ODE order.
        ...: Batch dimension(s).

        Args:
            t (Array): Time [...].
            x (Array): State [..., N, D].

        Raises:
            NotImplementedError: Needs to be implemented for a concrete ODE.

        Returns:
            Array: d/dt State [..., N, D].
        """

        return self.fn(t, x)
