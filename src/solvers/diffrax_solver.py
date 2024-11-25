import importlib
from functools import partial
from typing import Dict

import jax
from diffrax import AbstractImplicitSolver, DirectAdjoint, ODETerm, SaveAt, diffeqsolve
from jax import Array
from jax import numpy as jnp
from lineax import SVD
from optimistix import Newton

from src.ode.ode import ODE
from src.solvers.solver import ParametrizedSolver, Solver, SolverBuilder


class DiffraxSolverBuilder(SolverBuilder):
    """Abstract builder base class for ODE solvers."""

    def __init__(self, name: str = "ImplicitEuler", step_size: float = 0.1) -> None:
        """
        Initializes solver.

        Args:
            name (str, optional): Diffrax solver name. Defaults to ImplicitEuler.
            step_size (float, optional): Step size h. Defaults to 0.1.
        """

        super().__init__(step_size)

        solver_cls = getattr(importlib.import_module("diffrax"), name)
        if issubclass(solver_cls, AbstractImplicitSolver):
            self.solver = solver_cls(root_finder=Newton(rtol=1e-8, atol=1e-8), root_find_max_steps=500)  # type: ignore
        else:
            self.solver = solver_cls()

    def setup(self, ode: ODE, params: Dict[str, Array], *args, **kwargs) -> None:
        """
        Setups solver with arguments not ready at initialization.

        Args:
            ode (ODE): ODE RHS.
            params (Dict[str, Array]): ODE parameters.
        """

        self.ode = ode
        self.params = params

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

        term = ODETerm(self.ode)  # type: ignore
        diffrax_state = self.solver.init(term, t0, t0 + self.h, x0, self.params)
        return {"t": t0, "x": x0, "eps": jnp.zeros_like(x0), "diffrax_state": diffrax_state}

    def build(self) -> Solver:
        """
        Builds diffrax solver's transition function.

        Returns:
            Solver: Transition function.
        """

        if not hasattr(self, "ode") or not hasattr(self, "params"):
            raise AttributeError("Setup solver before usage!")

        solve = partial(self.build_parametrized(), self.ode, self.params)
        return solve

    def build_parametrized(self) -> ParametrizedSolver:
        """
        Builds diffrax solver's parametrized transition function.

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

            term = ODETerm(ode)  # type: ignore
            t, x, diffrax_state = state["t"], state["x"], state["diffrax_state"]
            t_next = t + self.h

            solution = diffeqsolve(
                term,
                self.solver,
                t0=t,
                t1=t_next,
                dt0=self.h,
                y0=x,
                args=params,
                adjoint=DirectAdjoint(),
                saveat=SaveAt(t1=True),
                solver_state=diffrax_state,
            )

            x_next = solution.ys[0]  # type: ignore
            diffrax_state_next = self.solver.init(term, t_next, t_next + self.h, x_next, params)
            eps = jnp.zeros_like(x_next)  # type: ignore

            next_state = {
                "t": t_next,
                "x": x_next,
                "eps": eps,
                "diffrax_state": diffrax_state_next,
            }
            return next_state

        return parametrized_solve
