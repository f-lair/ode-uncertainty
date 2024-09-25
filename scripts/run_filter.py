import math
import sys
from ast import literal_eval
from typing import Callable, Dict, Tuple

import jax

sys.path.append("../")
jax.config.update("jax_enable_x64", True)

import h5py
from jax import Array, lax
from jax import numpy as jnp
from jax_tqdm import scan_tqdm
from jsonargparse import CLI

from src.covariance_update_functions import *
from src.covariance_update_functions.covariance_update_function import (
    CovarianceUpdateFunction,
)
from src.filters import *
from src.filters.filter import FilterBuilder, FilterCorrect, FilterPredict
from src.ode import *
from src.ode.ode import ODEBuilder
from src.solvers import *
from src.solvers.solver import Solver, SolverBuilder
from src.utils import const_diag, store_data, sync_times


def main(
    output: str,
    filter_builder: FilterBuilder = SQRT_EKF(),
    solver_builder: SolverBuilder = Dopri65(),
    ode_builder: ODEBuilder = LotkaVolterra(),
    x0: str = "[[1.0, 1.0]]",
    P0: str | None = None,
    t0: float = 0.0,
    tN: float = 80.0,
    y_path: str | None = None,
    measurement_matrix: str | None = None,
    obs_noise_var: float = 1e-3,
    seed: int = 7,
    save_interval: int = 1,
    disable_pbar: bool = False,
) -> None:
    """
    Runs ODE filter.
    D: Latent dimension.
    N: ODE order.
    L: Observation dimension.

    Args:
        output (str): Path to H5 results file.
        filter_builder (FilterBuilder, optional): ODE filter builder. Defaults to SQRT_EKF().
        solver_builder (SolverBuilder, optional): ODE solver builder. Defaults to Dopri65().
        ode_builder (ODEBuilder, optional): ODE builder. Defaults to LotkaVolterra().
        x0 (str, optional): Initial value [N, D]. Defaults to "[[1.0, 1.0]]".
        P0 (str | None, optional): Initial covariance [N*D, N*D]. Defaults to None.
        t0 (float, optional): Start time. Defaults to 0.0.
        tN (float, optional): End time. Defaults to 80.0.
        y_path (str | None, optional): Path to H5 observations file. Defaults to None.
        measurement_matrix (str | None, optional): Measurement matrix [L, N*D]. Defaults to None.
        obs_noise_var (float, optional): Observation noise variance. Defaults to 1e-3.
        seed (int, optional): PRNG seed. Defaults to 7.
        save_interval (int, optional): Interval in which results are saved. Defaults to 1.
        disable_pbar (bool, optional): Disables progress bar. Defaults to False.
    """

    t0_arr = jnp.array([t0])
    x0_arr = jnp.array([literal_eval(x0)])
    x0_arr_built = ode_builder.build_initial_value(x0_arr, ode_builder.params)
    P0_sqrt_arr = (
        const_diag(x0_arr_built.size, 1e-12)[None, :, :]
        if P0 is None
        else jnp.linalg.cholesky(jnp.array(literal_eval(P0)))[None, :, :]
    )

    ode = ode_builder.build()
    step_size = solver_builder.h
    solver_builder.setup(ode, ode_builder.params)
    solver = jax.jit(jax.vmap(solver_builder.build()))
    filter_predict = jax.jit(filter_builder.build_predict(), static_argnums=(0, 1))
    cov_update_fn = jax.jit(filter_builder.build_cov_update_fn())

    num_steps = int(math.ceil((tN - t0) / step_size))

    if y_path is not None and measurement_matrix is not None:
        with h5py.File(y_path) as h5f:
            ts_y = jnp.asarray(h5f["t"])
            ts_x = jnp.arange(t0, tN + step_size, step_size)

            x_indices, y_indices = sync_times(ts_x, ts_y)
            x_flags = jnp.zeros(ts_x.shape, dtype=bool)
            x_flags = x_flags.at[x_indices].set(True)

            ys = jnp.asarray(h5f["x"])[y_indices]

        H = jnp.array(literal_eval(measurement_matrix))
        L = H.shape[0]
        assert H.shape[1] == P0_sqrt_arr.shape[-1], "Invalid measurement matrix!"
        measurement_fn = lambda x: H @ x
        ys = jnp.einsum("ij,tj->ti", H, ys.reshape(-1, H.shape[1]))

        filter_correct = jax.jit(filter_builder.build_correct(), static_argnums=(0,))
    else:
        print("Prediction only")
        L = 0
        x_flags = jnp.zeros(num_steps, dtype=bool)
        ys = jnp.zeros((1, L))
        measurement_fn = lambda x: x
        filter_correct = lambda _, x: x

    state_def = filter_builder.state_def(x0_arr_built.shape[-2], x0_arr_built.shape[-1], L)
    initial_state = {k: jnp.zeros(v) for k, v in state_def.items()}
    initial_state["t"] = jnp.broadcast_to(t0_arr, initial_state["t"].shape)
    initial_state["x"] = jnp.broadcast_to(x0_arr_built, initial_state["x"].shape)
    if "P_sqrt" in initial_state:
        initial_state["P_sqrt"] = jnp.broadcast_to(P0_sqrt_arr, initial_state["P_sqrt"].shape)
    if "R_sqrt" in initial_state:
        initial_state["R_sqrt"] = const_diag(L, obs_noise_var**0.5)
    if "prng_key" in initial_state:
        initial_state["prng_key"] = jax.random.key(seed)

    traj_states = unroll(
        filter_predict,
        filter_correct,
        solver,
        cov_update_fn,
        measurement_fn,
        initial_state,
        ys,
        x_flags,
        num_steps,
        save_interval,
        disable_pbar,
    )

    store_data(traj_states, output)


def unroll(
    filter_predict: FilterPredict,
    filter_correct: FilterCorrect,
    solver: Solver,
    cov_update_fn: CovarianceUpdateFunction,
    measurement_fn: Callable[[Array], Array],
    initial_state: Dict[str, Array],
    ys: Array,
    correct_flags: Array,
    num_steps: int,
    save_interval: int,
    disable_pbar: bool,
) -> Dict[str, Array]:
    """
    Unrolls trajectory.

    Args:
        filter_predict (FilterPredict): Predict function of ODE filter.
        filter_correct (FilterCorrect): Correct function of ODE filter.
        solver (Solver): ODE solver.
        cov_update_fn (CovarianceFunction): Covariance function.
        measurement_fn (Callable[[Array], Array]): Measurement function.
        initial_state (Dict[str, Array]): Initial state.
        ys (Array): Observations.
        correct_flags (Array): Flags indicating availability of observations.
        num_steps (int): Number of steps.
        save_interval (int): Interval in which results are saved.
        disable_pbar (bool): Disables progress bar.

    Returns:
        Dict[str, Array]: Trajectory states.
    """

    cond_true_correct = lambda state: filter_correct(measurement_fn, state)
    cond_false_correct = lambda state: state

    @scan_tqdm(num_steps, disable=disable_pbar)
    def scan_wrapper(
        state: Dict[str, Array], idx: Array
    ) -> Tuple[Dict[str, Array], Dict[str, Array]]:
        correct_flag = correct_flags[idx]
        state["y"] = ys.at[idx].get()
        state_predicted = filter_predict(solver, cov_update_fn, state)
        state_corrected = lax.cond(
            correct_flag, cond_true_correct, cond_false_correct, state_predicted
        )

        return state_corrected, state_corrected

    _, traj_states = lax.scan(scan_wrapper, initial_state, jnp.arange(num_steps, dtype=int))
    traj_states = {
        k: jnp.concat([initial_state[k][None, ...], traj_states[k]])[::save_interval]
        for k in traj_states
    }

    return traj_states


if __name__ == "__main__":
    CLI(main, as_positional=False)
