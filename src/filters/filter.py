from functools import partial
from typing import Callable, Dict, Tuple

from jax import Array
from jax.tree_util import Partial

from src.covariance_functions import DiagonalCovariance
from src.covariance_functions.covariance_function import (
    CovarianceFunction,
    CovarianceFunctionBuilder,
)
from src.ode.ode import ODE
from src.solvers.solver import ParametrizedSolver, Solver

# FilterPredict::(Solver:solver, CovarianceFunction:cov_fn, Dict[str, Array]:state) ->
# (Dict[str, Array]:next_state)
FilterPredict = Callable[[Solver, CovarianceFunction, Dict[str, Array]], Dict[str, Array]]
# ParametrizedFilterPredict::(Solver:solver, CovarianceFunction:cov_fn, ODE:ode,
# Dict[str,Array]:params, Dict[str, Array]:state) -> (Dict[str, Array]:next_state)
ParametrizedFilterPredict = Callable[
    [ParametrizedSolver, CovarianceFunction, ODE, Dict[str, Array], Dict[str, Array]],
    Dict[str, Array],
]
# FilterCorrect::(Callable[[Array], Array]:measurement_fn, Dict[str, Array]:state) ->
# (Dict[str, Array]:next_state)
FilterCorrect = Callable[[Callable[[Array], Array], Dict[str, Array]], Dict[str, Array]]


class FilterBuilder:
    """Abstract builder base class for filters, used for ODE solving."""

    def __init__(self, cov_fn_builder: CovarianceFunctionBuilder = DiagonalCovariance()) -> None:
        self.cov_fn_builder = cov_fn_builder

    def state_def(self, N: int, D: int, L: int) -> Dict[str, Tuple[int, ...]]:
        """
        Defines the solver state.

        Args:
            N (int): ODE order.
            D (int): Latent dimension.
            L (int): Measurement dimension.

        Raises:
            NotImplementedError: Needs to be defined for a concrete filter.

        Returns:
            Dict[str, Tuple[int, ...]]: State definition.
        """

        raise NotImplementedError

    def build_cov_fn(self) -> CovarianceFunction:
        """
        Builds covariance function.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete filter.

        Returns:
            CovarianceFunction: Covariance function.
        """

        raise NotImplementedError

    def build_predict(self) -> FilterPredict:
        """
        Builds filter's predict function.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete filter.

        Returns:
            FilterPredict: Predict function.
        """

        raise NotImplementedError

    def build_parametrized_predict(self) -> ParametrizedFilterPredict:
        """
        Builds filter's parametrized predict function.

        Returns:
            ParametrizedFilterPredict: Parametrized predict function.
        """

        def parametrized_predict(
            solver: ParametrizedSolver,
            cov_fn: CovarianceFunction,
            ode: ODE,
            params: Dict[str, Array],
            state: Dict[str, Array],
        ) -> Dict[str, Array]:
            predict = self.build_predict()
            return predict(partial(solver, ode, params), cov_fn, state)

        return parametrized_predict

    def build_correct(self) -> FilterCorrect:
        """
        Builds filter's correct function.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete filter.

        Returns:
            FilterCorrect: Correct function.
        """

        raise NotImplementedError
