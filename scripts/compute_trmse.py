import math
import sys
from ast import literal_eval
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Tuple

import jax

sys.path.append("../")
jax.config.update("jax_enable_x64", True)

import h5py
from jax import Array, lax
from jax import numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree
from jax_tqdm import scan_tqdm
from jsonargparse import CLI
from tqdm import trange

from src.ode import *
from src.ode.ode import ODE, ODEBuilder
from src.solvers import *
from src.solvers.solver import ParametrizedSolver, SolverBuilder
from src.utils import store_data


def main(
    output: str,
    solver_builder: SolverBuilder = Dopri65(),
    ode_builder: ODEBuilder = LotkaVolterra(),
    x0: str = "[[1.0, 1.0]]",
    t0: float = 0.0,
    tN: float = 80.0,
    noise_var: float = 0.0,
    save_interval: int = 1,
    parameter_estimates_input: str | None = None,
    seed: int = 7,
    disable_pbar: bool = False,
) -> None:
    """
    Runs ODE solver.
    D: Latent dimension.
    N: ODE order.

    Args:
        output (str): Path to H5 results file.
        solver_builder (SolverBuilder, optional): ODE solver builder. Defaults to Dopri65.
        ode_builder (ODEBuilder, optional): ODE builder. Defaults to LotkaVolterra().
        x0 (str, optional): Initial value [N, D]. Defaults to "[[1.0, 1.0]]".
        t0 (float, optional): Start time. Defaults to 0.0.
        tN (float, optional): End time. Defaults to 80.0.
        noise_var (float, optional): Noise variance (only added before saving). Defaults to 0.0.
        save_interval (int, optional): Interval in which results are saved. Defaults to 1.
        seed (int, optional): PRNG seed. Defaults to 7.
        disable_pbar (bool, optional): Disables progress bar. Defaults to False.
    """

    t0_arr = jnp.array(t0)
    x0_arr = jnp.array(literal_eval(x0))

    if parameter_estimates_input is None:
        raise ValueError("No input file provided!")

    params_estimated, params_name = retrieve_data(parameter_estimates_input)
    num_runs = params_estimated.shape[0]
    _, unravel_fn = ravel_pytree(ode_builder.params)

    params_estimated_l = []
    for idx1 in range(num_runs):
        params_estimated_i = deepcopy(ode_builder.params)
        for idx2, params_name_i in enumerate(params_name):
            if isinstance(params_estimated_i[params_name_i], list):
                params_estimated_i[params_name_i].append(params_estimated[idx1, idx2])
            else:
                params_estimated_i[params_name_i] = [params_estimated[idx1, idx2]]
        params_estimated_i = {
            k: jnp.stack(v).squeeze() if isinstance(v, list) else v
            for k, v in params_estimated_i.items()
        }
        params_estimated_i, _ = ravel_pytree(params_estimated_i)
        params_estimated_l.append(params_estimated_i)
    params_estimated = jnp.stack(params_estimated_l)

    ode = ode_builder.build()
    step_size = solver_builder.h
    solver_builder.setup(ode, ode_builder.params)
    solver = solver_builder.build_parametrized()
    num_steps = int(math.ceil((tN - t0) / step_size))
    unroll_p = jax.jit(
        partial(
            unroll,
            num_steps,
            solver,
            ode,
            ode_builder.build_initial_value,
            solver_builder.init_state,
        )
    )

    traj_true = unroll_p(t0_arr, x0_arr, ode_builder.params)

    @scan_tqdm(num_runs, disable=disable_pbar)
    def scan_wrapper(state: None, idx: Array) -> Tuple[None, Array]:
        params_estimated_dict = unravel_fn(params_estimated[idx])
        traj_estimated = unroll_p(t0_arr, x0_arr, params_estimated_dict)
        return None, compute_trmse(traj_true, traj_estimated)

    _, trmses = lax.scan(scan_wrapper, None, jnp.arange(num_runs, dtype=int))
    trmse_mean = jnp.mean(trmses)
    trmse_std = jnp.std(trmses, ddof=1)
    print(f"tRMSE={trmse_mean:.2f}±{trmse_std:.2f}")


def retrieve_data(path: str):
    with h5py.File(path) as h5f:
        params_estimated = jnp.asarray(h5f["params_optims"])[:, -1, :]
        params_name = list(h5f["params_name"].asstr())

    return params_estimated, params_name


def compute_trmse(traj_true: Array, traj_estimated: Array) -> Array:
    T = traj_true.shape[0]
    return jnp.sqrt(
        (1 / T)
        * jnp.sum(
            jnp.linalg.norm((traj_estimated - traj_true).reshape(T, -1), ord=2, axis=-1) ** 2
        )
    )


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def unroll(
    num_steps: int,
    solver: ParametrizedSolver,
    ode: ODE,
    build_initial_value: Callable,
    init_state: Callable,
    t0: Array,
    x0: Array,
    params: Dict[str, Array],
) -> Array:
    """
    Unrolls trajectory.

    Args:
        solver (Solver): ODE solver.
        initial_state (Dict[str, Array]): Initial state.
        num_steps (int): Number of steps.
        save_interval (int): Interval in which results are saved.
        disable_pbar (bool): Disables progress bar.

    Returns:
        Dict[str, Array]: Trajectory states.
    """

    def scan_wrapper(
        state: Dict[str, Array], idx: Array
    ) -> Tuple[Dict[str, Array], Dict[str, Array]]:
        state = solver(ode, params, state)
        return state, state

    x0_arr_built = build_initial_value(x0, params)
    initial_state = init_state(t0, x0_arr_built)
    _, traj_states = lax.scan(scan_wrapper, initial_state, jnp.arange(num_steps, dtype=int))
    return traj_states["x"]


if __name__ == "__main__":
    CLI(main, as_positional=False)
