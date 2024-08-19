from jax import Array
from jax import numpy as jnp


class SigmaFn:
    """Abstract base class for sigma functions."""

    def __call__(self, eps: Array) -> Array:
        """
        Evaluates function.
        D: Latent dimension.
        N: ODE order.

        Args:
            eps (Array): Local error estimate [N*D].

        Raises:
            NotImplementedError: Needs to be implemented for a concrete function.

        Returns:
            Array: Sigma [N*D, N*D].
        """

        raise NotImplementedError

    @staticmethod
    def sqrt(eps: Array) -> Array:
        """
        Evaluates matrix square root of function.

        Args:
            eps (Array): Local error estimate [N*D].

        Raises:
            NotImplementedError: Needs to be implemented for a concrete function.

        Returns:
            Array: Sigma^(1/2) [N*D, N*D].
        """

        raise NotImplementedError


class DiagonalSigma(SigmaFn):
    """Diagonal sigma."""

    def __call__(self, eps: Array) -> Array:
        """
        Evaluates function.
        D: Latent dimension.
        N: ODE order.

        Args:
            eps (Array): Local error estimate [N*D].

        Returns:
            Array: Sigma [N*D, N*D].
        """

        return jnp.diag(eps**2)

    @staticmethod
    def sqrt(eps: Array) -> Array:
        """
        Evaluates matrix square root of function.

        Args:
            eps (Array): Local error estimate [N*D].

        Returns:
            Array: Sigma^(1/2) [N*D, N*D].
        """

        return jnp.diag(eps)


class OuterSigma(SigmaFn):
    """Outer product sigma."""

    def __call__(self, eps: Array) -> Array:
        """
        Evaluates function.
        D: Latent dimension.
        N: ODE order.

        Args:
            eps (Array): Local error estimate [N*D].

        Returns:
            Array: Sigma [N*D, N*D].
        """

        return jnp.outer(eps, eps)

    @staticmethod
    def sqrt(eps: Array) -> Array:
        """
        Evaluates matrix square root of function.

        Args:
            eps (Array): Local error estimate [N*D].

        Returns:
            Array: Sigma^(1/2) [N*D, N*D].
        """

        return jnp.outer(eps, eps) / jnp.sqrt(jnp.dot(eps, eps))
