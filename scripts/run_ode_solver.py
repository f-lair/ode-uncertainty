import math
import sys
from ast import literal_eval
from pathlib import Path
from typing import Dict, Tuple

import jax

sys.path.append("../")
jax.config.update("jax_enable_x64", True)

import h5py
from jax import Array, lax
from jax import numpy as jnp
from jax import random
from jax_tqdm import scan_tqdm
from jsonargparse import CLI

from src.ode import *
from src.ode.ode import ODEBuilder
from src.solvers import *
from src.solvers.solver import Solver, SolverBuilder
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

    ode = ode_builder.build()
    step_size = solver_builder.h
    solver_builder.setup(ode, ode_builder.params)
    solver = solver_builder.build()

    num_steps = int(math.ceil((tN - t0) / step_size))
    x0_arr_built = ode_builder.build_initial_value(x0_arr, ode_builder.params)
    state_def = solver_builder.state_def(*x0_arr_built.shape)
    initial_state = {k: jnp.zeros(v) for k, v in state_def.items()}
    initial_state["t"] = t0_arr
    initial_state["x"] = x0_arr_built

    traj_states = unroll(solver, initial_state, num_steps, save_interval, disable_pbar)

    prng_key = random.key(seed)
    p = noise_var**0.5 * random.normal(prng_key, traj_states["x"].shape)
    traj_states["x"] = traj_states["x"] + p

    store_data(traj_states, output)


def unroll(
    solver: Solver,
    initial_state: Dict[str, Array],
    num_steps: int,
    save_interval: int,
    disable_pbar: bool,
) -> Dict[str, Array]:
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

    @scan_tqdm(num_steps, disable=disable_pbar)
    def scan_wrapper(
        state: Dict[str, Array], idx: Array
    ) -> Tuple[Dict[str, Array], Dict[str, Array]]:
        state = solver(state)
        return state, state

    _, traj_states = lax.scan(scan_wrapper, initial_state, jnp.arange(num_steps, dtype=int))
    traj_states = {
        k: jnp.concat([initial_state[k][None, ...], traj_states[k]])[::save_interval]
        for k in traj_states
    }

    return traj_states


if __name__ == "__main__":
    CLI(main, as_positional=False)
