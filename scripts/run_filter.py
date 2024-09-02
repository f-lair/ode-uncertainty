import sys
from ast import literal_eval
from pathlib import Path
from typing import Dict, Type

sys.path.append("../")

import h5py
from jax import Array, lax
from jax import numpy as jnp
from jsonargparse import CLI
from tqdm import tqdm

from src.filters import *
from src.filters.filter import Filter
from src.filters.perturbation_fns import *
from src.filters.sigma_fns import *
from src.ode import *
from src.ode.ode import ODE
from src.solvers import *
from src.solvers.rksolver import RKSolver


def main(
    output: str,
    filter_: Filter = EKF(),
    solver_cls: Type[RKSolver] = RKF45,
    ode: ODE = LotkaVolterra(),
    perturbation_fn: PerturbationFn = Gaussian(),
    sigma_fn: SigmaFn = DiagonalSigma(),
    x0: str = "[[1.0, 1.0]]",
    P0: str | None = None,
    t0: float = 0.0,
    tN: float = 80.0,
    dt: float = 0.1,
    save_interval: int = 1,
    disable_pbar: bool = False,
) -> None:
    """
    Runs filter-based ODE solver.
    D: Latent dimension.
    N: ODE order.

    Args:
        output (str): Path to H5 results file.
        filter_ (Filter, optional): ODE filter. Defaults to EKF().
        solver_cls (Type[RKSolver], optional): ODE solver class. Defaults to RKF45.
        ode (ODE, optional): ODE. Defaults to LotkaVolterra().
        sigma_fn (SigmaFn, optional): Sigma function. Defaults to DiagonalSigma().
        x0 (str, optional): Initial value [N, D]. Defaults to "[[1.0, 1.0]]".
        P0 (str | None, optional): Initial covariance [N*D, N*D]. Defaults to None.
        t0 (float, optional): Start time. Defaults to 0.0.
        tN (float, optional): End time. Defaults to 80.0.
        dt (float, optional): Step size. Defaults to 0.1.
        save_interval (int, optional): Timestep interval in which results are saved. Defaults to 1.
        disable_pbar (bool, optional): Disables progress bar. Defaults to False.
    """

    x0_arr = jnp.array(literal_eval(x0))[None, :, :]
    P0_arr = (
        jnp.zeros((1, x0_arr.size, x0_arr.size))
        if P0 is None
        else jnp.array([literal_eval(P0)])[None, :, :]
    )
    t0_arr = jnp.array([t0])
    tN_arr = jnp.array([tN])
    dt_arr = jnp.array([dt])

    solver = solver_cls(ode, t0_arr, x0_arr, dt_arr)
    if isinstance(filter_, ParticleFilter):
        filter_.setup(solver, P0_arr, perturbation_fn, sigma_fn)
    else:
        filter_.setup(solver, P0_arr, sigma_fn)

    results = unroll(
        filter_,
        tN_arr,
        save_interval,
        disable_pbar,
    )
    store(results, output)


def unroll(
    filter_: Filter,
    tN: Array,
    save_interval: int,
    disable_pbar: bool,
) -> Dict[str, Array]:
    """
    Unrolls trajectory.
    D: Latent dimension.
    N: ODE order.
    T: Time dimension.

    Args:
        filter_ (Filter): ODE filter.
        tN (Array): End time.
        save_interval (int): Timestep interval in which results are saved.
        disable_pbar (bool): Disables progress bar.

    Returns:
        Dict[str, Array]: Results according to filter's results_spec.
    """

    results = {key: [] for key in filter_.results_spec()}
    results["ts"].append(filter_.t)
    results["xs"].append(
        lax.pad(
            filter_.m,
            0.0,
            [(0, filter_.batch_dim() - filter_.m.shape[0], 0), (0, 0, 0), (0, 0, 0)],
        )
    )
    if "Ps" in results:
        results["Ps"].append(
            lax.pad(
                filter_.P,
                0.0,
                [(0, filter_.batch_dim() - filter_.P.shape[0], 0), (0, 0, 0), (0, 0, 0)],
            )
        )
    dx_dts = filter_.rk_solver.fn(filter_.t, filter_.m)
    results["dx_dts"].append(
        lax.pad(
            dx_dts,
            0.0,
            [(0, filter_.batch_dim() - dx_dts.shape[0], 0), (0, 0, 0), (0, 0, 0)],
        )
    )

    counter = 0
    t = filter_.t
    pbar = tqdm(total=tN.item(), disable=disable_pbar, unit="sec")

    while t < tN:
        step_results = filter_.predict()
        t = step_results["ts"]

        if counter % save_interval == 0:
            for key in results.keys() & step_results.keys():
                results[key].append(step_results[key])
        counter += 1
        pbar.update(t.item() - pbar.n)

    pbar.update(tN.item() - pbar.n)
    pbar.close()

    return {key: jnp.stack(data) for key, data in results.items()}


def store(results: Dict[str, Array], out_filepath: str) -> None:
    """
    Stores results data in H5 file on disk.

    Args:
        results (Dict[str, Array]): Results.
        out_filepath (str): Path to H5 results file.
    """

    Path(out_filepath).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_filepath, "w") as h5f:
        for key, data in results.items():
            h5f.create_dataset(key, data=data)


if __name__ == "__main__":
    CLI(main, as_positional=False)
