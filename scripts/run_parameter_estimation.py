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

from src.covariance_update_functions import *
from src.covariance_update_functions.covariance_update_function import (
    CovarianceUpdateFunction,
)
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
    negative_log_gaussian_sqrt,
    normalize,
    store_data,
    sync_times,
)


def optimize(
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
    params_range: Dict[str, Tuple[float, float]] | None = None,
    params_optimized: Dict[str, bool] | None = None,
    num_tempering_stages: int = 10,
    obs_noise_var: float = 0.1,
    gamma_noise_schedule: NoiseSchedule = ExponentialDecaySchedule(),
    gamma_noise_weights: str | None = None,
    lbfgs_maxiter: int = 200,
    num_random_runs: int = 0,
    num_param_evals: Dict[str, int] | None = None,
    seed: int = 7,
    num_processes: int = 4,
    disable_pbar: bool = False,
    verbose: bool = False,
) -> None:
    """
    Runs ODE parameter optimization.
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
        params_range (Dict[str, Tuple[float, float]] | None, optional): Parameter range. Defaults to None.
        params_optimized (Dict[str, bool] | None, optional): Parameters to be optimized. Defaults to None.
        num_tempering_stages (int, optional): Number of tempering stages. Defaults to 10.
        obs_noise_var (float, optional): Observation noise variance. Defaults to 0.1.
        gamma_noise_schedule (NoiseSchedule, optional): Noise schedule used for tempering. Defaults to ExponentialDecaySchedule().
        gamma_noise_weights (str | None, optional): Gamma noise weight vector [N*D]. Defaults to None.
        lbfgs_maxiter (int, optional): Max number of LBFGS iterations per tempering stage. Defaults to 200.
        num_random_runs (int, optional): Number of random runs. Defaults to 0.
        num_param_evals (Dict[str, int] | None, optional): Number of evaluations per parameter (unused). Defaults to None.
        seed (int, optional): PRNG seed. Defaults to 7.
        num_processes (int, optional): Number of parallel executed processes. Defaults to 4.
        disable_pbar (bool, optional): Disables progress bar. Defaults to False.
        verbose (bool, optional): Activates verbose output. Defaults to False.
    """

    t0_arr = jnp.array([t0])
    x0_arr = jnp.array([literal_eval(x0)])
    x0_arr_built = ode_builder.build_initial_value(x0_arr[0], ode_builder.params)[None]
    P0_sqrt_arr = (
        const_diag(x0_arr_built.size, 1e-12)[None, :, :]
        if P0 is None
        else jnp.linalg.cholesky(jnp.array(literal_eval(P0)))[None, :, :]
    )

    step_size = solver_builder.h
    num_steps = int(math.ceil((tN - t0) / step_size))

    if y_path is None:
        raise ValueError("Observation data is required!")
    if measurement_matrix is None:
        raise ValueError("Measurement matrix is required!")
    if gamma_noise_weights is None:
        raise ValueError("Gamma noise weight vector is required!")
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

    H = jnp.array(literal_eval(measurement_matrix))
    L = H.shape[0]
    assert H.shape[1] == P0_sqrt_arr.shape[-1], "Invalid measurement matrix!"
    ys = jnp.einsum("ij,tj->ti", H, ys.reshape(-1, H.shape[1]))

    w = jnp.array(literal_eval(gamma_noise_weights))
    assert w.shape[0] == P0_sqrt_arr.shape[-1], "Invalid gamma noise weight vector!"

    params_min = {
        k: jnp.full(ode_builder.params[k].shape[-1:], v[0]) for k, v in params_range.items()
    }
    params_max = {
        k: jnp.full(ode_builder.params[k].shape[-1:], v[1]) for k, v in params_range.items()
    }
    params_optimized_arr = {
        k: jnp.full(ode_builder.params[k].shape[-1:], v) for k, v in params_optimized.items()
    }
    assert len(params_min) == len(ode_builder.params), "Invalid parameter ranges!"
    assert len(params_max) == len(ode_builder.params), "Invalid parameter ranges!"
    assert len(params_optimized) == len(ode_builder.params), "Invalid optimization flags!"

    state_def = filter_builder.state_def(x0_arr_built.shape[-2], x0_arr_built.shape[-1], L)
    initial_state = {k: jnp.zeros(v) for k, v in state_def.items()}
    initial_state["t"] = jnp.broadcast_to(t0_arr, initial_state["t"].shape)
    initial_state["P_sqrt"] = jnp.broadcast_to(P0_sqrt_arr, initial_state["P_sqrt"].shape)
    initial_state["R_sqrt"] = const_diag(L, obs_noise_var**0.5)

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

    ode = jax.jit(ode_builder.build())
    solver_builder.setup(ode, ode_builder.params)
    solver = jax.jit(
        jax.vmap(solver_builder.build_parametrized(), (None, None, 0)), static_argnums=(0,)
    )
    filter_predict = jax.jit(filter_builder.build_parametrized_predict(), static_argnums=(0, 1, 2))
    filter_correct = jax.jit(filter_builder.build_correct(), static_argnums=(0,))
    cov_update_fn = jax.jit(filter_builder.build_cov_update_fn())
    measurement_fn = lambda x: H @ x
    nll_p = jax.jit(
        partial(
            nll,
            num_steps,
            filter_predict,
            filter_correct,
            solver,
            ode,
            cov_update_fn,
            measurement_fn,
        ),
    )

    optimize_run_p = partial(
        optimize_run,
        nll_p,
        ode_builder,
        gamma_noise_schedule,
        params_norms,
        initial_state,
        num_tempering_stages,
        x0_arr,
        H,
        ys,
        w,
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
    filter_builder: FilterBuilder = SQRT_EKF(),
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
    num_tempering_stages: int = 10,
    obs_noise_var: float = 0.1,
    gamma_noise_schedule: NoiseSchedule = ExponentialDecaySchedule(),
    gamma_noise_weights: str | None = None,
    lbfgs_maxiter: int = 200,
    num_random_runs: int = 0,
    num_param_evals: Dict[str, int] | None = None,
    seed: int = 7,
    num_processes: int = 4,
    disable_pbar: bool = False,
    verbose: bool = False,
) -> None:
    """
    Runs NLL evaluation for ODE parameter estimation.
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
        params_range (Dict[str, Tuple[float, float]] | None, optional): Parameter range. Defaults to None.
        params_optimized (Dict[str, bool] | None, optional): Parameters to be optimized. Defaults to None.
        num_tempering_stages (int, optional): Number of tempering stages. Defaults to 10.
        obs_noise_var (float, optional): Observation noise variance. Defaults to 0.1.
        gamma_noise_schedule (NoiseSchedule, optional): Noise schedule used for tempering. Defaults to ExponentialDecaySchedule().
        gamma_noise_weights (str | None, optional): Gamma noise weight vector [N*D]. Defaults to None.
        lbfgs_maxiter (int, optional): Max number of LBFGS iterations per tempering stage (unused). Defaults to 200.
        num_random_runs (int, optional): Number of random runs (unused). Defaults to 0.
        num_param_evals (Dict[str, int] | None, optional): Number of evaluations per parameter. Defaults to None.
        seed (int, optional): PRNG seed (unused). Defaults to 7.
        num_processes (int, optional): Number of parallel executed processes (unused). Defaults to 4.
        disable_pbar (bool, optional): Disables progress bar. Defaults to False.
        verbose (bool, optional): Activates verbose output (unused). Defaults to False.
    """

    t0_arr = jnp.array([t0])
    x0_arr = jnp.array([literal_eval(x0)])
    x0_arr_built = x0_arr_built = ode_builder.build_initial_value(x0_arr[0], ode_builder.params)[
        None
    ]
    P0_sqrt_arr = (
        const_diag(x0_arr_built.size, 1e-12)[None, :, :]
        if P0 is None
        else jnp.linalg.cholesky(jnp.array(literal_eval(P0)))[None, :, :]
    )

    ode = ode_builder.build()
    step_size = solver_builder.h
    solver_builder.setup(ode, ode_builder.params)
    solver = jax.jit(
        jax.vmap(solver_builder.build_parametrized(), (None, None, 0)), static_argnums=(0,)
    )
    filter_predict = jax.jit(filter_builder.build_parametrized_predict(), static_argnums=(0, 1, 2))
    filter_correct = jax.jit(filter_builder.build_correct(), static_argnums=(0,))
    cov_update_fn = jax.jit(filter_builder.build_cov_update_fn())

    num_steps = int(math.ceil((tN - t0) / step_size))

    if y_path is None:
        raise ValueError("Observation data is required!")
    if measurement_matrix is None:
        raise ValueError("Measurement matrix is required!")
    if gamma_noise_weights is None:
        raise ValueError("Gamma noise weight vector is required!")
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

    H = jnp.array(literal_eval(measurement_matrix))
    L = H.shape[0]
    assert H.shape[1] == P0_sqrt_arr.shape[-1], "Invalid measurement matrix!"
    measurement_fn = lambda x: H @ x
    ys = jnp.einsum("ij,tj->ti", H, ys.reshape(-1, H.shape[1]))

    w = jnp.array(literal_eval(gamma_noise_weights))
    assert w.shape[0] == P0_sqrt_arr.shape[-1], "Invalid gamma noise weight vector!"

    params_min = {
        k: jnp.full(ode_builder.params[k].shape[-1:], v[0]) for k, v in params_range.items()
    }
    params_max = {
        k: jnp.full(ode_builder.params[k].shape[-1:], v[1]) for k, v in params_range.items()
    }
    params_min_reduced = {k: v for k, v in params_min.items() if params_optimized[k]}
    params_max_reduced = {k: v for k, v in params_max.items() if params_optimized[k]}
    params_optimized_arr = {
        k: jnp.full(ode_builder.params[k].shape[-1:], v) for k, v in params_optimized.items()
    }
    params_optimized_indices = jnp.flatnonzero(ravel_pytree(params_optimized_arr)[0])
    assert len(params_min) == len(ode_builder.params), "Invalid parameter ranges!"
    assert len(params_max) == len(ode_builder.params), "Invalid parameter ranges!"

    state_def = filter_builder.state_def(x0_arr_built.shape[-2], x0_arr_built.shape[-1], L)
    initial_state = {k: jnp.zeros(v) for k, v in state_def.items()}
    initial_state["t"] = jnp.broadcast_to(t0_arr, initial_state["t"].shape)
    initial_state["P_sqrt"] = jnp.broadcast_to(P0_sqrt_arr, initial_state["P_sqrt"].shape)
    initial_state["R_sqrt"] = const_diag(L, obs_noise_var**0.5)

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
            filter_predict,
            filter_correct,
            solver,
            ode,
            cov_update_fn,
            measurement_fn,
        )
    )
    nll_evals = []
    gammas = []
    timings = []
    for tempering_idx in trange(
        0, num_tempering_stages, desc=f"Tempering stages", disable=disable_pbar
    ):
        gamma = gamma_noise_schedule.step(tempering_idx)
        initial_state["Q_sqrt"] = const_diag(H.shape[1], gamma.item() ** 0.5) * w

        nll_evals.append([])
        gammas.append(gamma)

        for eval_idx in trange(param_eval_arr.shape[0], desc=f"Evaluations", disable=disable_pbar):
            params_reg = unravel_fn(param_eval_arr[eval_idx])
            initial_state["x"] = jnp.broadcast_to(
                ode_builder.build_initial_value(x0_arr[0], params_reg)[None],
                initial_state["x"].shape,
            )
            params_norm = normalize(params_reg, params_min, params_max)
            params_norm_reduced = {k: v for k, v in params_norm.items() if params_optimized[k]}  # type: ignore
            t1 = perf_counter_ns()
            nll_eval = nll_p(
                params_norm_reduced,
                deepcopy(initial_state),
                ys,
                x_flags,
                xy_index_map,
                params_min_reduced,
                params_max_reduced,
                params_optimized_indices,
                ode_builder.params,
            )
            t2 = perf_counter_ns()
            nll_evals[-1].append(nll_eval)  # type: ignore
            if not (tempering_idx == 0 and eval_idx == 0):
                timings.append(t2 - t1)  # type: ignore

        nll_evals[-1] = jnp.stack(nll_evals[-1])

    nll_evals = jnp.stack(nll_evals)
    timings = jnp.stack(timings)

    gammas = jnp.stack(gammas)
    results = {
        "param_evals": param_eval_arr[:, params_optimized_indices],
        "nll_evals": nll_evals,
        "gammas": gammas,
        "timings": timings,
    }

    store_data(results, output, mode="a")


def optimize_run(
    nll_fn: Callable,
    ode_builder: ODEBuilder,
    gamma_noise_schedule: NoiseSchedule,
    params_norms: Dict[str, Array],
    initial_state: Dict[str, Array],
    num_tempering_stages: int,
    x0: Array,
    H: Array,
    ys: Array,
    w: Array,
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
        gamma_noise_schedule (NoiseSchedule): Noise schedule used for tempering.
        params_norms (Dict[str, Array]): Normed parameters.
        initial_state (Dict[str, Array]): Initial state.
        num_tempering_stages (int): Number of tempering stages.
        x0 (Array): Initial value.
        H (Array): Measurement matrix.
        ys (Array): Observations.
        w (Array): Gamma noise weight vector.
        correct_flags (Array): Flags indicating availability of observations.
        index_map (Array): Prediction -> observation index map.
        params_min (Dict[str, Array]): Minimum values per parameter.
        params_max (Dict[str, Array]): Maximum values per parameter.
        params_optimized (Dict[str, Array]): Parameters to be optimized.
        lbfgs_maxiter (int): Max number of LBFGS iterations per tempering stage.
        verbose (bool): Activates verbose output.
        run_idx (int): Run index.

    Returns:
        Tuple[Array, Array, Array, Array, Array, Array]:
            Initial parameters,
            optimized parameters at tempering stages,
            NLL values at tempering stages,
            number of LBFGS iterations,
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
    params_optims_reduced = []
    nll_optims = []
    num_lbfgs_iters = []
    num_nll_evals = []
    num_nll_jac_evals = []

    if verbose:
        print(
            "\nParameters [0]:",
            inv_normalize(params_norm_reduced, params_min_reduced, params_max_reduced),
        )
    for tempering_idx in range(0, num_tempering_stages):
        gamma = gamma_noise_schedule.step(tempering_idx)
        initial_state["Q_sqrt"] = const_diag(H.shape[1], gamma.item() ** 0.5) * w

        params_reduced = inv_normalize(params_norm_reduced, params_min_reduced, params_max_reduced)
        params_reduced_flat, _ = ravel_pytree(params_reduced)
        default_params_flat, unravel_fn = ravel_pytree(ode_builder.params)
        params = unravel_fn(
            default_params_flat.at[params_optimized_indices].set(
                params_reduced_flat, indices_are_sorted=True, unique_indices=True
            )
        )
        initial_state["x"] = jnp.broadcast_to(
            ode_builder.build_initial_value(x0[0], params)[None],  # type: ignore
            initial_state["x"].shape,
        )

        params_norm_reduced, lbfgsb_state = lbfgsb.run(
            init_params=params_norm_reduced,
            bounds=bounds,
            initial_state=deepcopy(initial_state),
            ys=ys,
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
        params_optims_reduced.append(ravel_pytree(params_optim_reduced)[0])
        nll_optims.append(lbfgsb_state.fun_val)
        num_lbfgs_iters.append(lbfgsb_state.iter_num)
        num_nll_evals.append(lbfgsb_state.num_fun_eval)
        num_nll_jac_evals.append(lbfgsb_state.num_jac_eval)

        if verbose:
            print(f"Gamma [{tempering_idx+1}]:", gamma)
            print(f"Parameters [{tempering_idx+1}]:", params_optim_reduced)
            print(f"LBFGSB state [{tempering_idx+1}]:", lbfgsb_state)
        jax.clear_caches()
    params_optims_reduced = jnp.stack(params_optims_reduced)
    nll_optims = jnp.stack(nll_optims)
    num_lbfgs_iters = jnp.stack(num_lbfgs_iters)
    num_nll_evals = jnp.stack(num_nll_evals)
    num_nll_jac_evals = jnp.stack(num_nll_jac_evals)

    return (
        params_init_reduced,
        params_optims_reduced,
        nll_optims,
        num_lbfgs_iters,
        num_nll_evals,
        num_nll_jac_evals,
    )


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5, 6))
def nll(
    num_steps: int,
    filter_predict: ParametrizedFilterPredict,
    filter_correct: FilterCorrect,
    solver: ParametrizedSolver,
    ode: ODE,
    cov_update_fn: CovarianceUpdateFunction,
    measurement_fn: Callable[[Array], Array],
    params_norm: Dict[str, Array],
    initial_state: Dict[str, Array],
    ys: Array,
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
        filter_predict (ParametrizedFilterPredict): Parametrized predict function of ODE filter.
        filter_correct (FilterCorrect): Parametrized correct function of ODE filter.
        solver (ParametrizedSolver): Parametrized ODE solver.
        ode (ODE): ODE RHS.
        cov_update_fn (CovarianceFunction): Covariance function.
        measurement_fn (Callable[[Array], Array]): Measurement function.
        params_norm (Dict[str, Array]): Normalized ODE parameters.
        initial_state (Dict[str, Array]): Initial state.
        ys (Array): Observations.
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

    def cond_true_correct(state: Dict[str, Array]) -> Tuple[Dict[str, Array], Array]:
        state_corrected = filter_correct(measurement_fn, state)
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
        state_predicted = filter_predict(solver, cov_update_fn, ode, params, state)

        state_corrected, nlg = lax.cond(
            correct_flag, cond_true_correct, cond_false_correct, state_predicted
        )

        return state_corrected, nlg

    _, nlls = lax.scan(nll_step, initial_state, jnp.arange(num_steps, dtype=int))
    out = nlls.sum()

    return out


if __name__ == "__main__":
    multiprocess.set_start_method("spawn")  # type: ignore
    CLI([optimize, evaluate], as_positional=False)
