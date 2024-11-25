from typing import Callable

from jax import Array
from jax import numpy as jnp

StaticCovarianceUpdateFunction = Callable[[Array, Array, Array], Array]


class StaticCovarianceUpdateFunctionBuilder:
    """Abstract builder base class for static covariance functions."""

    def __init__(self, scale: float = 1.0) -> None:
        """
        Initializes covariance update function.

        Args:
            scale (float, optional): Scale factor. Defaults to 1.0.
        """

        self.scale = jnp.array(scale)

    def build(self) -> StaticCovarianceUpdateFunction:
        """
        Builds covariance function.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete covariance function.

        Returns:
            StaticCovarianceUpdateFunction: Covariance function.
        """

        raise NotImplementedError

    def build_sqrt(self) -> StaticCovarianceUpdateFunction:
        """
        Builds covariance square-root function.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete covariance function.

        Returns:
            StaticCovarianceUpdateFunction: Covariance square-root function.
        """

        raise NotImplementedError
