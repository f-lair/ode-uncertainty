import sys
from ast import literal_eval
from pathlib import Path
from typing import Tuple, Type

sys.path.append("../")

import h5py
from jax import Array
from jax import numpy as jnp
from jsonargparse import CLI
from tqdm import tqdm

from src.ode import *
from src.ode.ode import ODE
from src.solvers import *
from src.solvers.rksolver import RKSolver


def main(
    output: str,
    solver_cls: Type[RKSolver] = RKF45,
    ode: ODE = LotkaVolterra(),
    x0: str = "[[1.0, 1.0]]",
    t0: float = 0.0,
    tN: float = 80.0,
    dt: float = 0.1,
    adaptive: bool = False,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    save_interval: int = 1,
    disable_pbar: bool = False,
) -> None:
    """
    Runs ODE solver.
    D: Latent dimension.
    N: ODE order.

    Args:
        output (str): Path to H5 results file.
        solver_cls (Type[RKSolver], optional): ODE solver class. Defaults to RKF45.
        ode (ODE, optional): ODE. Defaults to LotkaVolterra().
        x0 (str, optional): Initial value [N, D]. Defaults to "[[1.0, 1.0]]".
        t0 (float, optional): Start time. Defaults to 0.0.
        tN (float, optional): End time. Defaults to 80.0.
        dt (float, optional): Step size. Defaults to 0.1.
        adaptive (bool, optional): Activates adaptive step size control. Defaults to False.
        rtol (float, optional): Relative tolerance for adaptive step size control. Defaults to
            1e-6.
        atol (float, optional): Absolute tolerance for adaptive step size control. Defaults to
            1e-8.
        save_interval (int, optional): Timestep interval in which results are saved. Defaults to 1.
        disable_pbar (bool, optional): Disables progress bar. Defaults to False.
    """

    x0_arr = jnp.array([literal_eval(x0)])
    t0_arr = jnp.array([t0])
    tN_arr = jnp.array(tN)
    dt_arr = jnp.array(dt)

    solver = solver_cls(ode, t0_arr, x0_arr, dt_arr, adaptive, rtol=rtol, atol=atol)

    ts, xs = unroll(
        solver,
        tN_arr,
        save_interval,
        disable_pbar,
    )
    store(ts, xs, output)


def unroll(
    solver: RKSolver,
    tN: Array,
    save_interval: int,
    disable_pbar: bool,
) -> Tuple[Array, Array]:
    """
    Unrolls trajectory.
    D: Latent dimension.
    N: ODE order.
    T: Time dimension.

    Args:
        solver (RKSolver): ODE solver.
        tN (Array): End time.
        save_interval (int): Timestep interval in which results are saved.
        disable_pbar (bool): Disables progress bar.

    Returns:
        Tuple[Array, Array: Time [T], State [T, N, D].
    """

    ts = [solver.t0[0]]
    xs = [solver.x0[0]]

    counter = 0
    t = solver.t0
    x = solver.x0
    h = solver.h
    pbar = tqdm(total=tN.item(), disable=disable_pbar, unit="sec")

    while jnp.any(t < tN):
        if solver.adaptive_control:
            t, x, _, h = solver.step(t, x, h)  # [P, N, D]
        else:
            t, x, _, _ = solver.step(t, x)  # [P, N, D]

        if counter % save_interval == 0:
            ts.append(t[0])
            xs.append(x[0])
        counter += 1
        pbar.update(t.item() - pbar.n)

    pbar.update(tN.item() - pbar.n)
    pbar.close()

    ts = jnp.stack(ts)
    xs = jnp.stack(xs)

    return ts, xs


def store(
    ts: Array,
    xs: Array,
    out_filepath: str,
) -> None:
    """
    Saves results on disk.
    D: Latent dimension.
    N: ODE order.
    T: Time dimension.

    Args:
        ts (Array): Time [T].
        xs (Array): State [T, N, D].
        out_filepath (str): Path to H5 results file.
    """

    Path(out_filepath).parent.mkdir(parents=True, exist_ok=True)
    h5f = h5py.File(out_filepath, "w")
    h5f.create_dataset("ts", data=ts)
    h5f.create_dataset("xs", data=xs)
    h5f.close()


if __name__ == "__main__":
    CLI(main, as_positional=False)
