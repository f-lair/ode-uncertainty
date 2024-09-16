from typing import Callable

from jax import Array

CovarianceFunction = Callable[[Array], Array]


class CovarianceFunctionBuilder:
    """Abstract builder base class for covariance functions."""

    def build(self) -> CovarianceFunction:
        """
        Builds covariance function.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete covariance function.

        Returns:
            CovarianceFunction: Covariance function.
        """

        raise NotImplementedError

    def build_sqrt(self) -> CovarianceFunction:
        """
        Builds covariance square-root function.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete covariance function.

        Returns:
            CovarianceFunction: Covariance square-root function.
        """

        raise NotImplementedError
