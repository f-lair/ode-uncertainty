import math
import sys
from ast import literal_eval
from copy import deepcopy
from functools import partial
from typing import Callable, Dict, Tuple

import jax

sys.path.append("../")
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

import h5py
from jax import Array, lax
from jax import numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree
from jax.tree_util import Partial
from jaxopt import LBFGSB
from jsonargparse import CLI
from tqdm import tqdm, trange

from src.covariance_functions import *
from src.covariance_functions.covariance_function import CovarianceFunction
from src.filters import *
from src.filters.filter import FilterBuilder, FilterCorrect, ParametrizedFilterPredict
from src.noise_schedules import *
from src.ode import *
from src.ode.ode import ODE, ODEBuilder
from src.solvers import *
from src.solvers.solver import ParametrizedSolver, SolverBuilder
from src.utils import (
    const_diag,
    inv_normalize,
    negative_log_gaussian,
    normalize,
    store_data,
    sync_times,
)


def optimize(
    output: str,
    filter_builder: FilterBuilder = EKF(),
    solver_builder: SolverBuilder = Dopri65(),
    ode_builder: ODEBuilder = LotkaVolterra(),
    x0: str = "[[1.0, 1.0]]",
    P0: str | None = None,
    t0: float = 0.0,
    tN: float = 80.0,
    y_path: str | None = None,
    measurement_matrix: str | None = None,
    params_range: Dict[str, Tuple[float, float]] | None = None,
    params_optimized: Dict[str, bool] | None = None,
    num_tempering_steps: int = 10,
    obs_noise_var: float = 0.1,
    gamma_noise_schedule: NoiseSchedule = ExponentialDecaySchedule(),
    lbfgs_maxiter: int = 200,
    num_random_runs: int = 0,
    num_param_evals: Dict[str, int] | None = None,
    seed: int = 7,
    disable_pbar: bool = False,
    verbose: bool = False,
) -> None:

    t0_arr = jnp.array([t0])
    x0_arr = jnp.array([literal_eval(x0)])
    P0_arr = (
        jnp.zeros((1, x0_arr.size, x0_arr.size)) if P0 is None else jnp.array([literal_eval(P0)])
    )

    ode = ode_builder.build()
    step_size = solver_builder.h
    solver_builder.setup(ode, ode_builder.params)
    solver = jax.jit(
        jax.vmap(solver_builder.build_parametrized(), (None, None, 0)), static_argnums=(0, 1)
    )
    filter_predict = jax.jit(filter_builder.build_parametrized_predict(), static_argnums=(0, 1, 2))
    filter_correct = jax.jit(filter_builder.build_correct(), static_argnums=(0,))
    cov_fn = jax.jit(filter_builder.build_cov_fn())

    num_steps = int(math.ceil((tN - t0) / step_size))

    if y_path is None:
        raise ValueError("Observation data is required!")
    if measurement_matrix is None:
        raise ValueError("Measurement matrix is required!")
    if params_range is None:
        raise ValueError("Parameter ranges are required!")
    if params_optimized is None:
        params_optimized = {k: True for k in ode_builder.params}

    with h5py.File(y_path) as h5f:
        ts_y = jnp.asarray(h5f["t"])
        ts_x = jnp.arange(t0, tN + step_size, step_size)

        x_indices, y_indices = sync_times(ts_x, ts_y)
        x_flags = jnp.zeros(ts_x.shape, dtype=bool)
        x_flags = x_flags.at[x_indices].set(True)
        xy_index_map = jnp.zeros(ts_x.shape, dtype=int)
        xy_index_map = xy_index_map.at[x_indices].set(y_indices)

        ys = jnp.asarray(h5f["x"])[y_indices]

    H = jnp.array(literal_eval(measurement_matrix))
    assert H.shape[1] == P0_arr.shape[-1], "Invalid measurement matrix!"
    measurement_fn = lambda x: H @ x
    ys = jnp.einsum("ij,tj->ti", H, ys.reshape(-1, P0_arr.shape[-1]))

    params_min = {k: jnp.array(v[0]) for k, v in params_range.items()}
    params_max = {k: jnp.array(v[1]) for k, v in params_range.items()}
    params_optimized_arr = {k: jnp.array(v) for k, v in params_optimized.items()}
    assert len(params_min) == len(ode_builder.params), "Invalid parameter ranges!"
    assert len(params_max) == len(ode_builder.params), "Invalid parameter ranges!"
    assert len(params_optimized) == len(ode_builder.params), "Invalid optimization flags!"

    state_def = filter_builder.state_def(x0_arr.shape[-2], x0_arr.shape[-1], H.shape[0])
    initial_state = {k: jnp.zeros(v) for k, v in state_def.items()}
    initial_state["t"] = jnp.broadcast_to(t0_arr, initial_state["t"].shape)
    initial_state["x"] = jnp.broadcast_to(x0_arr, initial_state["x"].shape)
    initial_state["P"] = jnp.broadcast_to(P0_arr, initial_state["P"].shape)
    initial_state["R"] = const_diag(H.shape[0], obs_noise_var)

    if num_random_runs > 0:
        prng_key = random.split(random.key(seed), len(ode_builder.params))
        params_norms = {
            k: (
                random.uniform(prng_key[idx], shape=(num_random_runs,))
                if params_optimized[k]
                else jnp.broadcast_to(
                    normalize(ode_builder.params[k], params_min[k], params_max[k])[None],  # type: ignore
                    (num_random_runs,),
                )
            )
            for idx, k in enumerate(ode_builder.params.keys())
        }
    else:
        params_norms = {
            k: normalize(ode_builder.params[k], params_min[k], params_max[k])[None]  # type: ignore
            for k in ode_builder.params
        }
        num_random_runs = 1

    nll_p = partial(
        nll,
        num_steps,
        filter_predict,
        filter_correct,
        solver,
        ode,
        cov_fn,
        measurement_fn,
    )
    lbfgsb = LBFGSB(nll_p, maxiter=lbfgs_maxiter, jit=True, verbose=False)
    bounds = ({k: 0.0 for k in ode_builder.params}, {k: 1.0 for k in ode_builder.params})

    params_inits = []
    params_optims = []
    nll_optims = []
    num_lbfgs_iters = []
    for run_idx in trange(num_random_runs, desc=f"Runs", disable=disable_pbar):
        params_norm = {k: params_norms[k][run_idx] for k in params_norms}
        params_inits.append(ravel_pytree(inv_normalize(params_norm, params_min, params_max))[0])
        params_optims.append([])
        nll_optims.append([])
        num_lbfgs_iters.append(jnp.zeros(()))

        if verbose:
            print("Parameters [0]:", inv_normalize(params_norm, params_min, params_max))
        for tempering_idx in range(1, num_tempering_steps + 1):
            gamma = gamma_noise_schedule.step(tempering_idx)
            if tempering_idx == num_tempering_steps:
                gamma = jnp.zeros(())
            initial_state["Q"] = const_diag(H.shape[0], gamma.item())

            params_norm, lbfgsb_state = lbfgsb.run(
                init_params=params_norm,
                bounds=bounds,
                initial_state=deepcopy(initial_state),
                ys=ys,
                correct_flags=x_flags,
                index_map=xy_index_map,
                params_min=params_min,
                params_max=params_max,
                params_optimized=params_optimized_arr,
                params_default=ode_builder.params,
            )
            params_optim = inv_normalize(params_norm, params_min, params_max)
            params_optims[run_idx].append(ravel_pytree(params_optim)[0])
            nll_optims[run_idx].append(lbfgsb_state.value)
            num_lbfgs_iters[run_idx] = num_lbfgs_iters[run_idx] + lbfgsb_state.iter_num

            if verbose:
                print(f"Parameters [{tempering_idx}]:", params_optim)
                print(f"LBFGSB state [{tempering_idx}]:", lbfgsb_state)
        params_optims[run_idx] = jnp.stack(params_optims[run_idx])
        nll_optims[run_idx] = jnp.stack(nll_optims[run_idx])

    params_inits = jnp.stack(params_inits)
    params_optims = jnp.stack(params_optims)
    nll_optims = jnp.stack(nll_optims)
    num_lbfgs_iters = jnp.stack(num_lbfgs_iters)
    results = {
        "params_inits": params_inits,
        "params_optims": params_optims,
        "nll_optims": nll_optims,
        "num_lbfgs_iters": num_lbfgs_iters,
    }

    store_data(results, output, mode="a")


def evaluate(
    output: str,
    filter_builder: FilterBuilder = EKF(),
    solver_builder: SolverBuilder = Dopri65(),
    ode_builder: ODEBuilder = LotkaVolterra(),
    x0: str = "[[1.0, 1.0]]",
    P0: str | None = None,
    t0: float = 0.0,
    tN: float = 80.0,
    y_path: str | None = None,
    measurement_matrix: str | None = None,
    params_range: Dict[str, Tuple[float, float]] | None = None,
    params_optimized: Dict[str, bool] | None = None,
    num_tempering_steps: int = 10,
    obs_noise_var: float = 0.1,
    gamma_noise_schedule: NoiseSchedule = ExponentialDecaySchedule(),
    lbfgs_maxiter: int = 200,
    num_random_runs: int = 0,
    num_param_evals: Dict[str, int] | None = None,
    seed: int = 7,
    disable_pbar: bool = False,
    verbose: bool = False,
) -> None:

    t0_arr = jnp.array([t0])
    x0_arr = jnp.array([literal_eval(x0)])
    P0_arr = (
        jnp.zeros((1, x0_arr.size, x0_arr.size)) if P0 is None else jnp.array([literal_eval(P0)])
    )

    ode = ode_builder.build()
    step_size = solver_builder.h
    solver_builder.setup(ode, ode_builder.params)
    solver = jax.vmap(solver_builder.build_parametrized(), (None, None, 0))
    filter_predict = filter_builder.build_parametrized_predict()
    filter_correct = filter_builder.build_correct()
    cov_fn = filter_builder.build_cov_fn()

    num_steps = int(math.ceil((tN - t0) / step_size))

    if y_path is None:
        raise ValueError("Observation data is required!")
    if measurement_matrix is None:
        raise ValueError("Measurement matrix is required!")
    if params_range is None:
        raise ValueError("Parameter ranges are required!")
    if params_optimized is None:
        params_optimized = {k: True for k in ode_builder.params}
    if num_param_evals is None:
        raise ValueError("Parameter evaluation counts are required!")

    with h5py.File(y_path) as h5f:
        ts_y = jnp.asarray(h5f["t"])
        ts_x = jnp.arange(t0, tN + step_size, step_size)

        x_indices, y_indices = sync_times(ts_x, ts_y)
        x_flags = jnp.zeros(ts_x.shape, dtype=bool)
        x_flags = x_flags.at[x_indices].set(True)
        xy_index_map = jnp.zeros(ts_x.shape, dtype=int)
        xy_index_map = xy_index_map.at[x_indices].set(y_indices)

        ys = jnp.asarray(h5f["x"])[y_indices]

    H = jnp.array(literal_eval(measurement_matrix))
    assert H.shape[1] == P0_arr.shape[-1], "Invalid measurement matrix!"
    measurement_fn = lambda x: H @ x
    ys = jnp.einsum("ij,tj->ti", H, ys.reshape(-1, P0_arr.shape[-1]))

    params_min = {k: jnp.array(v[0]) for k, v in params_range.items()}
    params_max = {k: jnp.array(v[1]) for k, v in params_range.items()}
    params_optimized_arr = {k: jnp.array(v) for k, v in params_optimized.items()}
    assert len(params_min) == len(ode_builder.params), "Invalid parameter ranges!"
    assert len(params_max) == len(ode_builder.params), "Invalid parameter ranges!"

    state_def = filter_builder.state_def(x0_arr.shape[-2], x0_arr.shape[-1], H.shape[0])
    initial_state = {k: jnp.zeros(v) for k, v in state_def.items()}
    initial_state["t"] = jnp.broadcast_to(t0_arr, initial_state["t"].shape)
    initial_state["x"] = jnp.broadcast_to(x0_arr, initial_state["x"].shape)
    initial_state["P"] = jnp.broadcast_to(P0_arr, initial_state["P"].shape)
    initial_state["R"] = const_diag(H.shape[0], obs_noise_var)

    param_eval_arr = [
        jnp.linspace(params_min[k], params_max[k], num_param_evals[k]) for k in ode_builder.params
    ]
    param_eval_arr = jnp.stack(jnp.meshgrid(*param_eval_arr, indexing="ij"), axis=-1).reshape(
        -1, len(ode_builder.params)
    )
    _, unravel_fn = ravel_pytree(ode_builder.params)

    nll_p = partial(
        nll,
        num_steps,
        filter_predict,
        filter_correct,
        solver,
        ode,
        cov_fn,
        measurement_fn,
    )
    nll_evals = []
    gammas = []
    for tempering_idx in trange(
        1, num_tempering_steps + 1, desc=f"Tempering stages", disable=disable_pbar
    ):
        gamma = gamma_noise_schedule.step(tempering_idx)
        if tempering_idx == num_tempering_steps:
            gamma = jnp.zeros(())
        initial_state["Q"] = const_diag(H.shape[0], gamma.item())

        nll_evals.append([])
        gammas.append(gamma)

        for eval_idx in trange(param_eval_arr.shape[0], desc=f"Evaluations", disable=disable_pbar):
            params_norm = normalize(unravel_fn(param_eval_arr[eval_idx]), params_min, params_max)
            nll_eval = nll_p(
                params_norm,
                initial_state,
                ys,
                x_flags,
                xy_index_map,
                params_min,
                params_max,
                params_optimized_arr,
                ode_builder.params,
            )
            nll_evals[-1].append(nll_eval)
        nll_evals[-1] = jnp.stack(nll_evals[-1])

    nll_evals = jnp.stack(nll_evals)
    gammas = jnp.stack(gammas)
    results = {
        "param_evals": param_eval_arr,
        "nll_evals": nll_evals,
        "gammas": gammas,
    }

    store_data(results, output)


# @partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5, 6))
def nll(
    num_steps: int,
    filter_predict: ParametrizedFilterPredict,
    filter_correct: FilterCorrect,
    solver: ParametrizedSolver,
    ode: ODE,
    cov_fn: CovarianceFunction,
    measurement_fn: Callable[[Array], Array],
    params_norm: Dict[str, Array],
    initial_state: Dict[str, Array],
    ys: Array,
    correct_flags: Array,
    index_map: Array,
    params_min: Dict[str, Array],
    params_max: Dict[str, Array],
    params_optimized: Dict[str, Array],
    params_default: Dict[str, Array],
) -> Array:
    params = jax.tree.map(
        jnp.where,
        params_optimized,
        inv_normalize(params_norm, params_min, params_max),
        params_default,
    )

    def cond_true_correct(state: Dict[str, Array]) -> Tuple[Dict[str, Array], Array]:
        state_corrected = filter_correct(measurement_fn, state)
        nlg = negative_log_gaussian(
            state_corrected["y"], state_corrected["y_hat"][0], state_corrected["S"][0]
        )
        return state_corrected, nlg

    def cond_false_correct(state: Dict[str, Array]) -> Tuple[Dict[str, Array], Array]:
        nlg = jnp.zeros(())
        return state, nlg

    # @jax.jit
    def nll_step(state: Dict[str, Array], idx: Array) -> Tuple[Dict[str, Array], Array]:
        correct_flag = correct_flags[idx]
        state["y"] = ys.at[index_map[idx]].get()
        state_predicted = filter_predict(solver, cov_fn, ode, params, state)

        if correct_flag:
            state_corrected, nlg = cond_true_correct(state_predicted)
        else:
            state_corrected, nlg = cond_false_correct(state_predicted)

        # state_corrected, nlg = lax.cond(
        #     correct_flag, cond_true_correct, cond_false_correct, state_predicted
        # )

        return state_corrected, nlg

    # _, nlls = lax.scan(nll_step, initial_state, jnp.arange(num_steps, dtype=int))
    _, nlls = scan_debug(nll_step, initial_state, jnp.arange(num_steps, dtype=int))
    return nlls.sum()


def scan_debug(f, init, xs):
    carry = init
    ys = []
    for x in tqdm(xs):
        carry, y = f(carry, x)
        ys.append(y)
    return carry, jnp.stack(ys)


if __name__ == "__main__":
    CLI([optimize, evaluate], as_positional=False)
