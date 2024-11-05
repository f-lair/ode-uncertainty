from typing import Callable, Dict, Set, Tuple

from jax import Array
from jax import numpy as jnp

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

    def init_state(self, t0: Array, x0: Array) -> Dict[str, Array]:
        """
        Initializes the solver state.
        D: Latent dimension.
        N: ODE order.

        Args:
            t0 (Array): Initial time [].
            x0 (Array): Initial state [N, D].

        Returns:
            Dict[str, Array]: Initial state.
        """

        return {"t": t0, "x": x0, "eps": jnp.zeros_like(x0), "diffrax_state": jnp.zeros(())}

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
