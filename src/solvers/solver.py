from typing import Callable, Dict, Set, Tuple

from jax import Array

from src.ode.ode import ODE

# Solver::(Dict[str, Array]:state) -> (Dict[str, Array]:next_state)
Solver = Callable[[Dict[str, Array]], Dict[str, Array]]
# ParametrizedSolver::(ODE:ode, Dict[str,Array]:params, Dict[str, Array]:state) ->
# (Dict[str, Array]:next_state)
ParametrizedSolver = Callable[[ODE, Dict[str, Array], Dict[str, Array]], Dict[str, Array]]


class SolverBuilder:
    """Abstract builder base class for ODE solvers."""

    def __init__(self, step_size: float = 0.1) -> None:
        """
        Initializes solver.

        Args:
            step_size (float, optional): Step size h. Defaults to 0.1.
        """

        self.h = step_size

    def setup(self, *args, **kwargs) -> None:
        """
        Setups solver with arguments not ready at initialization.
        """

        pass

    def state_def(self, N: int, D: int) -> Dict[str, Tuple[int, ...]]:
        """
        Defines the solver state.

        Args:
            N (int): ODE order.
            D (int): Latent dimension.

        Returns:
            Dict[str, Tuple[int, ...]]: State definition.
        """

        return {"t": (), "x": (N, D), "eps": (N, D)}

    def build(self) -> Solver:
        """
        Builds ODE solver's transition function.

        Raises:
            NotImplementedError: Needs to be defined for a concrete solver.

        Returns:
            Solver: Transition function.
        """

        raise NotImplementedError

    def build_parametrized(self) -> ParametrizedSolver:
        """
        Builds ODE solver's parametrized transition function.

        Raises:
            NotImplementedError: Needs to be defined for a concrete solver.

        Returns:
            ParametrizedSolver: Parametrized transition function.
        """

        raise NotImplementedError
