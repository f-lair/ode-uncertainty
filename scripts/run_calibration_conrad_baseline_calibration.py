import math
import sys
from ast import literal_eval
from functools import partial
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
from src.utils import const_diag, negative_log_gaussian_sqrt, store_data, sync_times


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
    obs_noise_var: float = 0.0,
    min_noise_log: float = -3.0,
    max_noise_log: float = 1.0,
    num_noise_levels: int = 100,
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

    t0_arr = jnp.array(t0)
    x0_arr = jnp.array(literal_eval(x0))
    x0_arr_built = ode_builder.build_initial_value(x0_arr, ode_builder.params)
    P0_sqrt_arr = (
        const_diag(x0_arr_built.size, 1e-12)
        if P0 is None
        else jnp.linalg.cholesky(jnp.array(literal_eval(P0)))
    )

    ode = ode_builder.build()
    step_size = solver_builder.h
    solver_builder.setup(ode, ode_builder.params)
    solver = jax.jit(jax.vmap(solver_builder.build()))
    filter_predict = jax.jit(filter_builder.build_predict(), static_argnums=(0, 1))
    cov_update_fn = jax.jit(filter_builder.build_cov_update_fn())

    num_steps = int(math.ceil((tN - t0) / step_size))

    if y_path is None:
        raise ValueError("No observations provided!")
    if measurement_matrix is None:
        H = jnp.eye(P0_sqrt_arr.shape[0])
    else:
        H = jnp.array(literal_eval(measurement_matrix), dtype=float)

    with h5py.File(y_path) as h5f:
        ts_y = jnp.asarray(h5f["t"])
        ts_x = jnp.arange(t0 + step_size, tN + step_size, step_size)
        x_indices, y_indices = sync_times(ts_x, ts_y)
        x_flags = jnp.zeros(ts_x.shape, dtype=bool)
        x_flags = x_flags.at[x_indices].set(True)
        xy_index_map = jnp.zeros(ts_x.shape, dtype=int)
        xy_index_map = xy_index_map.at[x_indices].set(y_indices)
        ys = jnp.asarray(h5f["x"])

    L = H.shape[0]
    assert H.shape[1] == P0_sqrt_arr.shape[-1], "Invalid measurement matrix!"
    ys = jnp.einsum("ij,tj->ti", H, ys.reshape(-1, H.shape[1]))

    filter_correct = jax.jit(filter_builder.build_correct())

    solver_state = solver_builder.init_state(t0_arr, x0_arr_built)
    if isinstance(filter_builder, SQRT_EKF):
        initial_state = filter_builder.init_state(
            solver_state,
            P0_sqrt_arr,
            jnp.zeros_like(P0_sqrt_arr),
            jnp.zeros(()),
            const_diag(L, obs_noise_var**0.5),
        )
        static_cov_update_fn = jax.jit(filter_builder.build_static_cov_update_fn())
    else:
        raise ValueError("Unsupported filter builder:", type(filter_builder))

    noise_levels = jnp.logspace(min_noise_log, max_noise_log, num_noise_levels, endpoint=True)

    @scan_tqdm(len(noise_levels), disable=disable_pbar)
    def scan_wrapper(state: None, idx: Array) -> Tuple[None, Array]:
        nll_val = nll(
            num_steps,
            filter_predict,
            filter_correct,
            solver,
            partial(static_cov_update_fn, noise_levels.at[idx].get()),
            H,
            initial_state,
            ys,
            x_flags,
            xy_index_map,
        )
        return None, nll_val

    _, nlls_conrad = lax.scan(scan_wrapper, None, jnp.arange(num_noise_levels, dtype=int))
    nll_ours = nll(
        num_steps,
        filter_predict,
        filter_correct,
        solver,
        cov_update_fn,
        H,
        initial_state,
        ys,
        x_flags,
        xy_index_map,
    )

    nll_data = {"noise_levels": noise_levels, "nll_conrad": nlls_conrad, "nll_ours": nll_ours}

    store_data(nll_data, output)


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def nll(
    num_steps: int,
    filter_predict: FilterPredict,
    filter_correct: FilterCorrect,
    solver: Solver,
    cov_update_fn: CovarianceUpdateFunction,
    measurement_matrix: Array,
    initial_state: Dict[str, Array],
    ys: Array,
    correct_flags: Array,
    xy_index_map: Array,
) -> Array:
    """
    Unrolls trajectory.

    Args:
        filter_predict (FilterPredict): Predict function of ODE filter.
        filter_correct (FilterCorrect): Correct function of ODE filter.
        solver (Solver): ODE solver.
        cov_update_fn (CovarianceFunction): Covariance function.
        measurement_matrix (Array): Measurement matrix.
        initial_state (Dict[str, Array]): Initial state.
        ys (Array): Observations.
        correct_flags (Array): Flags indicating availability of observations.
        index_map (Array): Prediction -> observation index map.
        num_steps (int): Number of steps.
        save_interval (int): Interval in which results are saved.
        disable_pbar (bool): Disables progress bar.

    Returns:
        Dict[str, Array]: Trajectory states.
    """

    def cond_true_correct(state: Dict[str, Array]) -> Tuple[Dict[str, Array], Array]:
        state_corrected = filter_correct(measurement_matrix, state)
        nlg = negative_log_gaussian_sqrt(
            state_corrected["y"], state_corrected["y_hat"][0], state_corrected["S_sqrt"][0]
        )
        return state_corrected, nlg

    def cond_false_correct(state: Dict[str, Array]) -> Tuple[Dict[str, Array], Array]:
        nlg = jnp.zeros(())
        return state, nlg

    def nll_step(state: Dict[str, Array], idx: Array) -> Tuple[Dict[str, Array], Array]:
        correct_flag = correct_flags[idx]
        state["y"] = ys.at[xy_index_map[idx]].get()
        state_predicted = filter_predict(solver, cov_update_fn, state)
        state_corrected, nlg = lax.cond(
            correct_flag, cond_true_correct, cond_false_correct, state_predicted
        )

        return state_corrected, nlg

    _, nlls = lax.scan(nll_step, initial_state, jnp.arange(num_steps, dtype=int))

    nlls = jnp.nan_to_num(nlls)
    q95 = jnp.percentile(nlls, jnp.array(95))
    nlls = jnp.clip(nlls, 0.0, q95)

    out = nlls.sum()

    return out


if __name__ == "__main__":
    CLI(main, as_positional=False)
