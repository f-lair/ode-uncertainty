from functools import partial
from typing import Dict, Tuple

from jax import Array, lax
from jax import numpy as jnp

from src.ode.ode import ODE
from src.solvers.solver import ParametrizedSolver, Solver, SolverBuilder


class RKSolverBuilder(SolverBuilder):
    """Abstract builder base class for explicit embedded Runge-Kutta solvers."""

    def __init__(self, step_size: float = 0.1) -> None:
        """
        Initializes solver.

        Args:
            step_size (float, optional): Step size h. Defaults to 0.1.
        """

        super().__init__(step_size=step_size)

        # Coefficients
        self.A = self.build_A()  # [S, S]
        self.b = self.build_b()  # [2, S]
        self.c = self.build_c()  # [S]

        # Stage
        self.s = self.A.shape[0]

    def setup(self, ode: ODE, params: Dict[str, Array], *args, **kwargs) -> None:
        """
        Setups solver with arguments not ready at initialization.

        Args:
            ode (ODE): ODE RHS.
            params (Dict[str, Array]): ODE parameters.
        """

        self.ode = ode
        self.params = params

    @classmethod
    def build_A(cls) -> Array:
        """
        Builds coefficient matrix A in the Butcher tableau.
        S: Stage.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete RK solver.

        Returns:
            Array: Coefficient matrix A [S, S].
        """

        raise NotImplementedError

    @classmethod
    def build_b(cls) -> Array:
        """
        Builds coefficient matrix b in the Butcher tableau.
        Second row is used to compute the actual next state, first row is only for local truncation
        error estimation.
        S: Stage.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete RK solver.

        Returns:
            Array: Coefficient matrix b [2, S].
        """

        raise NotImplementedError

    @classmethod
    def build_c(cls) -> Array:
        """
        Builds coefficient vector c in the Butcher tableau.
        S: Stage.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete RK solver.

        Returns:
            Array: Coefficient vector c [S].
        """

        raise NotImplementedError

    def build(self) -> Solver:
        """
        Builds RK solver's transition function.

        Returns:
            Solver: Transition function.
        """

        if not hasattr(self, "ode") or not hasattr(self, "params"):
            raise AttributeError("Setup solver before usage!")

        solve = partial(self.build_parametrized(), self.ode, self.params)
        return solve

    def build_parametrized(self) -> ParametrizedSolver:
        """
        Builds RK solver's parametrized transition function.

        Returns:
            ParametrizedSolver: Parametrized transition function.
        """

        def parametrized_solve(
            ode: ODE, params: Dict[str, Array], state: Dict[str, Array]
        ) -> Dict[str, Array]:
            """
            Performs single step.
            D: Latent dimension.
            N: ODE order.
            S: Stage.

            Args:
                ode (ODE): ODE RHS.
                params (Dict[str, Array]): ODE parameters.
                state (Dict[str, Array]):
                    t (Array): Time [],
                    x (Array): State [N, D].

            Returns:
                Dict[str, Array]:
                    t (Array): Next time [],
                    x (Array): Next state [N, D],
                    eps (Array): Local error estimate [N, D].
            """

            t, x = state["t"], state["x"]

            ks = jnp.zeros(x.shape + self.c.shape)  # [N, D, S]
            ts = t[None] + self.h * self.c  # [S]

            # Iterate over nodes, while reusing previously computed nodes
            compute_node_p = partial(compute_node, ode, params, ts, x, self.A, self.h)
            (_, ks), _ = lax.scan(compute_node_p, (0, ks), length=self.s)  # [N, D, S]

            t_next = t + self.h  # []
            x_next = x[:, :, None] + self.h * (ks @ self.b.T)  # [N, D, 2]
            eps = jnp.abs((x_next[:, :, 0] - x_next[:, :, 1]))  # [N, D]

            next_state = {
                "t": t_next,
                "x": x_next[:, :, 1],
                "eps": eps,
                "diffrax_state": jnp.zeros(()),
            }
            return next_state

        return parametrized_solve


def compute_node(
    ode: ODE,
    params: Dict[str, Array],
    ts: Array,
    x: Array,
    A: Array,
    h: float,
    state: Tuple[int, Array],
    x_: None,
) -> Tuple[Tuple[int, Array], None]:
    """
    Computes single node k_i.
    D: Latent dimension.
    N: ODE order.
    S: Stage.

    Args:
        ode (ODE): ODE RHS.
        params (Dict[str, Array]): ODE parameters.
        ts (Array): Time points to evaluate ODE at [S].
        x (Array): State [N, D].
        A (Array): Coefficient matrix A [S, S].
        h (float): Step size.
        state (Tuple[int, Array]): Node index i, node vectors [N, D, S].
        x_ (None): Unused scan value.

    Returns:
        Tuple[Tuple[int, Array], None]:
            Next node index i+1, updated node vectors [N, D, S];
            Unused scan value.
    """

    idx, ks = state
    k = ode(ts[idx], x + h * (ks @ A[idx]), params)  # [N, D]
    return (idx + 1, ks.at[:, :, idx].set(k)), None  # [N, D, S]
