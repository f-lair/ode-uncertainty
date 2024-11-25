import itertools
import math
import sys
from ast import literal_eval
from copy import deepcopy
from functools import partial
from time import perf_counter_ns
from typing import Callable, Dict, Tuple

import jax

sys.path.append("../")
jax.config.update("jax_enable_x64", True)

import h5py
import multiprocess
from jax import Array, lax
from jax import numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree
from jaxopt import ScipyBoundedMinimize
from jsonargparse import CLI
from p_tqdm import p_umap
from tqdm import trange

from src.ode import *
from src.ode.ode import ODE, ODEBuilder
from src.solvers import *
from src.solvers.solver import ParametrizedSolver, SolverBuilder
from src.utils import (
    const_diag,
    inv_normalize,
    negative_log_gaussian_sqrt,
    normalize,
    store_data,
    sync_times,
)


def optimize(
    output: str,
    solver_builder: SolverBuilder = Dopri65(),
    ode_builder: ODEBuilder = LotkaVolterra(),
    x0: str = "[[1.0, 1.0]]",
    t0: float = 0.0,
    tN: float = 80.0,
    y_path: str | None = None,
    measurement_matrix: str | None = None,
    params_range: Dict[str, Tuple[float, float]] | None = None,
    params_optimized: Dict[str, bool] | None = None,
    obs_noise_var: float = 0.1,
    initial_state_parametrized: bool = False,
    lbfgs_maxiter: int = 200,
    num_random_runs: int = 0,
    num_param_evals: Dict[str, int] | None = None,
    seed: int = 7,
    num_processes: int = 4,
    disable_pbar: bool = False,
    verbose: bool = False,
) -> None:
    """
    Runs ODE parameter optimization (RK baseline).
    D: Latent dimension.
    N: ODE order.
    L: Observation dimension.

    Args:
        output (str): Path to H5 results file.
        solver_builder (SolverBuilder, optional): ODE solver builder. Defaults to Dopri65().
        ode_builder (ODEBuilder, optional): ODE builder. Defaults to LotkaVolterra().
        x0 (str, optional): Initial value [N, D]. Defaults to "[[1.0, 1.0]]".
        t0 (float, optional): Start time. Defaults to 0.0.
        tN (float, optional): End time. Defaults to 80.0.
        y_path (str | None, optional): Path to H5 observations file. Defaults to None.
        measurement_matrix (str | None, optional): Measurement matrix [L, N*D]. Defaults to None.
        params_range (Dict[str, Tuple[float, float]] | None, optional): Parameter range. Defaults to None.
        params_optimized (Dict[str, bool] | None, optional): Parameters to be optimized. Defaults to None.
        obs_noise_var (float, optional): Observation noise variance. Defaults to 0.1.
        lbfgs_maxiter (int, optional): Max number of LBFGS iterations per tempering stage. Defaults to 200.
        num_random_runs (int, optional): Number of random runs. Defaults to 0.
        num_param_evals (Dict[str, int] | None, optional): Number of evaluations per parameter (unused). Defaults to None.
        seed (int, optional): PRNG seed. Defaults to 7.
        num_processes (int, optional): Number of parallel executed processes. Defaults to 4.
        disable_pbar (bool, optional): Disables progress bar. Defaults to False.
        verbose (bool, optional): Activates verbose output. Defaults to False.
    """

    t0_arr = jnp.array(t0)
    x0_arr = jnp.array(literal_eval(x0))
    x0_arr_built = ode_builder.build_initial_value(x0_arr, ode_builder.params)

    step_size = solver_builder.h
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
        ts_x = jnp.arange(t0 + step_size, tN + step_size, step_size)
        x_indices, y_indices = sync_times(ts_x, ts_y)
        x_flags = jnp.zeros(ts_x.shape, dtype=bool)
        x_flags = x_flags.at[x_indices].set(True)
        xy_index_map = jnp.zeros(ts_x.shape, dtype=int)
        xy_index_map = xy_index_map.at[x_indices].set(y_indices)
        ys = jnp.asarray(h5f["x"])

    H = jnp.array(literal_eval(measurement_matrix), dtype=float)
    L = H.shape[0]
    assert H.shape[1] == x0_arr_built.size, "Invalid measurement matrix!"
    ys = jnp.einsum("ij,tj->ti", H, ys.reshape(-1, H.shape[1]))

    params_min = {k: jnp.full(ode_builder.params[k].shape, v[0]) for k, v in params_range.items()}
    params_max = {k: jnp.full(ode_builder.params[k].shape, v[1]) for k, v in params_range.items()}
    params_optimized_arr = {
        k: jnp.full(ode_builder.params[k].shape, v) for k, v in params_optimized.items()
    }
    assert len(params_min) == len(ode_builder.params), "Invalid parameter ranges!"
    assert len(params_max) == len(ode_builder.params), "Invalid parameter ranges!"
    assert len(params_optimized) == len(ode_builder.params), "Invalid optimization flags!"

    ode = jax.jit(ode_builder.build())
    solver_builder.setup(ode, ode_builder.params)
    initial_state = solver_builder.init_state(t0_arr, x0_arr_built)
    R_sqrt = const_diag(L, obs_noise_var**0.5)

    if num_random_runs > 0:
        prng_key = random.split(random.key(seed), len(ode_builder.params))
        params_norms = {
            k: (
                random.uniform(
                    prng_key[idx], shape=(num_random_runs,) + ode_builder.params[k].shape[-1:]
                )
                if params_optimized[k]
                else jnp.broadcast_to(
                    normalize(
                        ode_builder.params[k][
                            tuple(0 for _ in range(ode_builder.params[k].ndim - 1))
                            + (
                                (slice(0, ode_builder.params[k].shape[-1]),)
                                if ode_builder.params[k].ndim > 0
                                else tuple()
                            )
                        ],
                        params_min[k],
                        params_max[k],
                    )[
                        None
                    ],  # type: ignore
                    (num_random_runs,) + ode_builder.params[k].shape[-1:],
                )
            )
            for idx, k in enumerate(ode_builder.params.keys())
        }
    else:
        params_norms = {
            k: normalize(
                ode_builder.params[k][
                    tuple(0 for _ in range(ode_builder.params[k].ndim - 1))
                    + (
                        (slice(0, ode_builder.params[k].shape[-1]),)
                        if ode_builder.params[k].ndim > 0
                        else tuple()
                    )
                ],
                params_min[k],
                params_max[k],
            )[
                None
            ]  # type: ignore
            for k in ode_builder.params
        }
        num_random_runs = 1

    solver = jax.jit(solver_builder.build_parametrized(), static_argnums=(0,))
    nll_p = jax.jit(
        partial(
            nll,
            num_steps,
            initial_state_parametrized,
            solver,
            ode,
            ode_builder.build_initial_value,
        ),
    )

    optimize_run_p = partial(
        optimize_run,
        nll_p,
        ode_builder,
        params_norms,
        initial_state,
        x0_arr,
        H,
        ys,
        R_sqrt,
        x_flags,
        xy_index_map,
        params_min,
        params_max,
        params_optimized_arr,
        lbfgs_maxiter,
        verbose,
    )

    if num_random_runs > 1:
        results = p_umap(
            optimize_run_p,
            range(num_random_runs),
            num_cpus=num_processes,
            desc="Runs",
            disable=disable_pbar,
        )
    else:
        results = [optimize_run_p(0)]

    params_default, unravel_fn = ravel_pytree(
        {k: v for k, v in ode_builder.params.items() if params_optimized[k]}
    )
    params_name = list(
        itertools.chain(
            *[
                [param_name] * ode_builder.params[param_name].size
                for param_name in unravel_fn(params_default).keys()
            ]
        )
    )

    params_inits, params_optims, nll_optims, num_lbfgs_iters, num_nll_evals, num_nll_jac_evals = (
        zip(*results)
    )
    params_inits = jnp.stack(params_inits)
    params_optims = jnp.stack(params_optims)
    nll_optims = jnp.stack(nll_optims)
    num_lbfgs_iters = jnp.stack(num_lbfgs_iters)
    num_nll_evals = jnp.stack(num_nll_evals)
    num_nll_jac_evals = jnp.stack(num_nll_jac_evals)
    results = {
        "params_inits": params_inits,
        "params_optims": params_optims,
        "params_default": params_default,
        "params_name": params_name,
        "nll_optims": nll_optims,
        "num_lbfgs_iters": num_lbfgs_iters,
        "num_nll_evals": num_nll_evals,
        "num_nll_jac_evals": num_nll_jac_evals,
    }

    store_data(results, output, mode="a")


def evaluate(
    output: str,
    solver_builder: SolverBuilder = Dopri65(),
    ode_builder: ODEBuilder = LotkaVolterra(),
    x0: str = "[[1.0, 1.0]]",
    t0: float = 0.0,
    tN: float = 80.0,
    y_path: str | None = None,
    measurement_matrix: str | None = None,
    params_range: Dict[str, Tuple[float, float]] | None = None,
    params_optimized: Dict[str, bool] | None = None,
    obs_noise_var: float = 0.1,
    initial_state_parametrized: bool = False,
    lbfgs_maxiter: int = 200,
    num_random_runs: int = 0,
    num_param_evals: Dict[str, int] | None = None,
    seed: int = 7,
    num_processes: int = 4,
    disable_pbar: bool = False,
    verbose: bool = False,
) -> None:
    """
    Runs NLL evaluation for ODE parameter estimation (RK baseline).
    D: Latent dimension.
    N: ODE order.
    L: Observation dimension.

    Args:
        output (str): Path to H5 results file.
        solver_builder (SolverBuilder, optional): ODE solver builder. Defaults to Dopri65().
        ode_builder (ODEBuilder, optional): ODE builder. Defaults to LotkaVolterra().
        x0 (str, optional): Initial value [N, D]. Defaults to "[[1.0, 1.0]]".
        t0 (float, optional): Start time. Defaults to 0.0.
        tN (float, optional): End time. Defaults to 80.0.
        y_path (str | None, optional): Path to H5 observations file. Defaults to None.
        measurement_matrix (str | None, optional): Measurement matrix [L, N*D]. Defaults to None.
        params_range (Dict[str, Tuple[float, float]] | None, optional): Parameter range. Defaults to None.
        params_optimized (Dict[str, bool] | None, optional): Parameters to be optimized. Defaults to None.
        obs_noise_var (float, optional): Observation noise variance. Defaults to 0.1.
        lbfgs_maxiter (int, optional): Max number of LBFGS iterations per tempering stage (unused). Defaults to 200.
        num_random_runs (int, optional): Number of random runs (unused). Defaults to 0.
        num_param_evals (Dict[str, int] | None, optional): Number of evaluations per parameter. Defaults to None.
        seed (int, optional): PRNG seed (unused). Defaults to 7.
        num_processes (int, optional): Number of parallel executed processes (unused). Defaults to 4.
        disable_pbar (bool, optional): Disables progress bar. Defaults to False.
        verbose (bool, optional): Activates verbose output (unused). Defaults to False.
    """

    t0_arr = jnp.array(t0)
    x0_arr = jnp.array(literal_eval(x0))
    x0_arr_built = ode_builder.build_initial_value(x0_arr, ode_builder.params)

    ode = jax.jit(ode_builder.build())
    step_size = solver_builder.h
    solver_builder.setup(ode, ode_builder.params)
    solver = jax.jit(solver_builder.build_parametrized(), static_argnums=(0,))

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
        ts_x = jnp.arange(t0 + step_size, tN + step_size, step_size)
        x_indices, y_indices = sync_times(ts_x, ts_y)
        x_flags = jnp.zeros(ts_x.shape, dtype=bool)
        x_flags = x_flags.at[x_indices].set(True)
        xy_index_map = jnp.zeros(ts_x.shape, dtype=int)
        xy_index_map = xy_index_map.at[x_indices].set(y_indices)
        ys = jnp.asarray(h5f["x"])

    H = jnp.array(literal_eval(measurement_matrix), dtype=float)
    L = H.shape[0]
    assert H.shape[1] == x0_arr_built.size, "Invalid measurement matrix!"
    ys = jnp.einsum("ij,tj->ti", H, ys.reshape(-1, H.shape[1]))

    params_min = {
        k: jnp.full(ode_builder.params[k].shape[-1:], v[0]) for k, v in params_range.items()
    }
    params_max = {
        k: jnp.full(ode_builder.params[k].shape[-1:], v[1]) for k, v in params_range.items()
    }
    params_min_reduced = {k: v for k, v in params_min.items() if params_optimized[k]}
    params_max_reduced = {k: v for k, v in params_max.items() if params_optimized[k]}
    params_optimized_arr = {
        k: jnp.full(ode_builder.params[k].shape, v) for k, v in params_optimized.items()
    }
    params_optimized_indices = jnp.flatnonzero(ravel_pytree(params_optimized_arr)[0])
    assert len(params_min) == len(ode_builder.params), "Invalid parameter ranges!"
    assert len(params_max) == len(ode_builder.params), "Invalid parameter ranges!"

    initial_state = solver_builder.init_state(t0_arr, x0_arr_built)
    R_sqrt = const_diag(L, obs_noise_var**0.5)

    param_eval_arr = [
        (
            jnp.linspace(params_min[k][idx], params_max[k][idx], num_param_evals[k])
            if ode_builder.params[k].ndim > 0
            else jnp.linspace(params_min[k], params_max[k], num_param_evals[k])
        )
        for k in sorted(ode_builder.params)
        for idx in range(ode_builder.params[k].size)
    ]

    param_eval_arr = jnp.stack(jnp.meshgrid(*param_eval_arr, indexing="ij"), axis=-1).reshape(
        -1, len(param_eval_arr)
    )
    _, unravel_fn = ravel_pytree(ode_builder.params)

    nll_p = jax.jit(
        partial(
            nll,
            num_steps,
            initial_state_parametrized,
            solver,
            ode,
            ode_builder.build_initial_value,
        )
    )
    nll_evals = []
    timings = []

    for eval_idx in trange(param_eval_arr.shape[0], desc=f"Evaluations", disable=disable_pbar):
        params_reg = unravel_fn(param_eval_arr[eval_idx])
        params_norm = normalize(params_reg, params_min, params_max)
        params_norm_reduced = {k: v for k, v in params_norm.items() if params_optimized[k]}  # type: ignore
        try:
            t1 = perf_counter_ns()
            nll_eval = nll_p(
                params_norm_reduced,
                deepcopy(initial_state),
                x0_arr,
                H,
                ys,
                R_sqrt,
                x_flags,
                xy_index_map,
                params_min_reduced,
                params_max_reduced,
                params_optimized_indices,
                ode_builder.params,
            )
            t2 = perf_counter_ns()
        except RuntimeError as err:
            if verbose:
                print(f"An error occured at evaluation {eval_idx+1}:", str(err))
            nll_eval = jnp.array(jnp.nan)
            t1, t2 = 0, 0
        nll_evals.append(nll_eval)  # type: ignore
        if eval_idx > 0:
            timings.append(t2 - t1)  # type: ignore

    nll_evals = jnp.stack(nll_evals)
    timings = jnp.stack(timings)
    results = {
        "param_evals": param_eval_arr[:, params_optimized_indices],
        "nll_evals": nll_evals,
        "timings": timings,
    }

    store_data(results, output, mode="a")


def optimize_run(
    nll_fn: Callable,
    ode_builder: ODEBuilder,
    params_norms: Dict[str, Array],
    initial_state: Dict[str, Array],
    x0: Array,
    H: Array,
    ys: Array,
    R_sqrt: Array,
    correct_flags: Array,
    xy_index_map: Array,
    params_min: Dict[str, Array],
    params_max: Dict[str, Array],
    params_optimized: Dict[str, Array],
    lbfgs_maxiter: int,
    verbose: bool,
    run_idx: int,
) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """
    Performs a single optimization run.

    Args:
        nll_fn (Callable): NLL function.
        ode_builder (ODEBuilder): ODE builder.
        params_norms (Dict[str, Array]): Normed parameters.
        initial_state (Dict[str, Array]): Initial state.
        x0 (Array): Initial value.
        H (Array): Measurement matrix.
        ys (Array): Observations.
        R_sqrt (Array): Observatation noise.
        correct_flags (Array): Flags indicating availability of observations.
        index_map (Array): Prediction -> observation index map.
        params_min (Dict[str, Array]): Minimum values per parameter.
        params_max (Dict[str, Array]): Maximum values per parameter.
        params_optimized (Dict[str, Array]): Parameters to be optimized.
        lbfgs_maxiter (int): Max number of LBFGS iterations tempering stage.
        verbose (bool): Activates verbose output.
        run_idx (int): Run index.

    Returns:
        Tuple[Array, Array, Array, Array]:
            Initial parameters,
            optimized parameters,
            NLL value,
            number of LBFGS iterations
            number of NLL evaluations,
            number of NLL Jacobian evaluations.
    """

    params_norms_reduced = {k: v for k, v in params_norms.items() if params_optimized[k].any()}
    params_min_reduced = {k: v for k, v in params_min.items() if params_optimized[k].any()}
    params_max_reduced = {k: v for k, v in params_max.items() if params_optimized[k].any()}
    params_optimized_indices = jnp.flatnonzero(ravel_pytree(params_optimized)[0])

    lbfgsb = ScipyBoundedMinimize(fun=nll_fn, method="L-BFGS-B", maxiter=lbfgs_maxiter, jit=True)
    bounds = (
        {k: jnp.zeros_like(v[run_idx]) for k, v in params_norms_reduced.items()},
        {k: jnp.ones_like(v[run_idx]) for k, v in params_norms_reduced.items()},
    )

    params_norm_reduced = {k: params_norms_reduced[k][run_idx] for k in params_norms_reduced}
    params_init_reduced = ravel_pytree(
        inv_normalize(params_norm_reduced, params_min_reduced, params_max_reduced)
    )[0]

    if verbose:
        print(
            "\nParameters [0]:",
            inv_normalize(params_norm_reduced, params_min_reduced, params_max_reduced),
        )

    try:
        params_norm_reduced, lbfgsb_state = lbfgsb.run(
            init_params=params_norm_reduced,
            bounds=bounds,
            initial_state=deepcopy(initial_state),
            x0=x0,
            measurement_matrix=H,
            ys=ys,
            R_sqrt=R_sqrt,
            correct_flags=correct_flags,
            xy_index_map=xy_index_map,
            params_min=params_min_reduced,
            params_max=params_max_reduced,
            params_optimized_indices=params_optimized_indices,
            params_default=ode_builder.params,
        )
        params_optim_reduced = inv_normalize(
            params_norm_reduced, params_min_reduced, params_max_reduced
        )

        if verbose:
            print(f"Parameters [1]:", params_optim_reduced)
            print(f"LBFGSB state [1]:", lbfgsb_state)
        jax.clear_caches()

        params_optim_reduced = ravel_pytree(params_optim_reduced)[0]
        nll_optim = lbfgsb_state.fun_val
        num_lbfgs_iters = lbfgsb_state.iter_num
        num_nll_evals = lbfgsb_state.num_fun_eval
        num_nll_jac_evals = lbfgsb_state.num_jac_eval
    except RuntimeError as err:
        if verbose:
            print(f"An error occured during optimization:", str(err))
        params_optim_reduced = ravel_pytree(
            inv_normalize(params_norm_reduced, params_min_reduced, params_max_reduced)
        )[0]
        nll_optim = jnp.zeros(())
        num_lbfgs_iters = jnp.zeros(())
        num_nll_evals = jnp.zeros(())
        num_nll_jac_evals = jnp.zeros(())

    return (
        params_init_reduced,
        params_optim_reduced,
        nll_optim,
        num_lbfgs_iters,
        num_nll_evals,
        num_nll_jac_evals,
    )


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def nll(
    num_steps: int,
    initial_state_parametrized: bool,
    solver: ParametrizedSolver,
    ode: ODE,
    ode_build_initial_value: Callable,
    params_norm: Dict[str, Array],
    initial_state: Dict[str, Array],
    x0: Array,
    measurement_matrix: Array,
    ys: Array,
    R_sqrt: Array,
    correct_flags: Array,
    xy_index_map: Array,
    params_min: Dict[str, Array],
    params_max: Dict[str, Array],
    params_optimized_indices: Array,
    params_default: Dict[str, Array],
) -> Array:
    """
    Computes negative log-likelihood.

    Args:
        num_steps (int): Number of steps.
        solver (ParametrizedSolver): Parametrized ODE solver.
        ode (ODE): ODE RHS.
        params_norm (Dict[str, Array]): Normalized ODE parameters.
        initial_state (Dict[str, Array]): Initial state.
        x0 (Array): Initial value.
        measurement_matrix (Array): Measurement matrix.
        ys (Array): Observations.
        R_sqrt (Array): Measurement noise.
        correct_flags (Array): Flags indicating availability of observations.
        index_map (Array): Prediction -> observation index map.
        params_min (Dict[str, Array]): Minimum values per parameter.
        params_max (Dict[str, Array]): Maximum values per parameter.
        params_optimized_indices (Array): Indices of parameters to be optimized.
        params_default (Dict[str, Array]): Default values per parameter.

    Returns:
        Array: NLL [].
    """

    params = inv_normalize(params_norm, params_min, params_max)
    params_flat, _ = ravel_pytree(params)
    default_params_flat, unravel_fn = ravel_pytree(params_default)
    params = unravel_fn(
        default_params_flat.at[params_optimized_indices].set(
            params_flat, indices_are_sorted=True, unique_indices=True
        )
    )
    if initial_state_parametrized:
        initial_state["x"] = jnp.broadcast_to(
            ode_build_initial_value(x0, params),  # type: ignore
            initial_state["x"].shape,
        )

    def cond_true_correct(state: Dict[str, Array], y: Array) -> Array:
        y_hat = measurement_matrix @ state["x"].ravel()
        nlg = negative_log_gaussian_sqrt(y, y_hat, R_sqrt)
        return nlg

    def cond_false_correct(state: Dict[str, Array], y: Array) -> Array:
        nlg = jnp.zeros(())
        return nlg

    def nll_step(state: Dict[str, Array], idx: Array) -> Tuple[Dict[str, Array], Array]:
        next_state = solver(ode, params, state)

        correct_flag = correct_flags[idx]
        y = ys.at[xy_index_map[idx]].get()

        nlg = lax.cond(correct_flag, cond_true_correct, cond_false_correct, next_state, y)

        return next_state, nlg

    _, nlls = lax.scan(nll_step, initial_state, jnp.arange(num_steps, dtype=int))
    out = nlls.sum()

    return out


if __name__ == "__main__":
    multiprocess.set_start_method("spawn")  # type: ignore
    CLI([optimize, evaluate], as_positional=False)
