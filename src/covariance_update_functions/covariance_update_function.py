from typing import Callable

from jax import Array

CovarianceUpdateFunction = Callable[[Array, Array], Array]


class CovarianceUpdateFunctionBuilder:
    """Abstract builder base class for covariance functions."""

    def build(self) -> CovarianceUpdateFunction:
        """
        Builds covariance function.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete covariance function.

        Returns:
            CovarianceUpdateFunction: Covariance function.
        """

        raise NotImplementedError

    def build_sqrt(self) -> CovarianceUpdateFunction:
        """
        Builds covariance square-root function.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete covariance function.

        Returns:
            CovarianceUpdateFunction: Covariance square-root function.
        """

        raise NotImplementedError
