import sys
from ast import literal_eval
from pathlib import Path
from typing import Tuple, Type

sys.path.append("../")

import h5py
from jax import Array
from jax import numpy as jnp
from jax import random
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
    eps_perturbations: int = 0,
    eps_perturbations_mode: str = "Diagonal",
    const_perturbations: int = 0,
    const_perturbations_val: float = 0.1,
    seed: int = 7,
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
        eps_perturbations (int, optional): Number of perturbed sample paths computed using local
            truncation error. Defaults to 0.
        eps_perturbations_mode (str, optional): Mode used for eps-based perturbations: Diagonal,
            Outer. Defaults to "Diagonal".
        const_perturbations (int, optional): Number of perturbed sample paths computed using
            constant. Defaults to 0.
        const_perturbations_val (float, optional): Constant used for perturbations. Defaults to
            0.1.
        seed (int, optional): Seed used for random perturbations. Defaults to 7.
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

    prng_key = random.key(seed)
    solver = solver_cls(ode, t0_arr, x0_arr, dt_arr, adaptive, rtol=rtol, atol=atol)

    ts, xs, xs_eps_p, xs_const_p, epss, epss_eps_p, epss_const_p = unroll(
        solver,
        tN_arr,
        prng_key,
        eps_perturbations,
        eps_perturbations_mode,
        const_perturbations,
        const_perturbations_val,
        save_interval,
        disable_pbar,
    )
    store(ts, xs, xs_eps_p, xs_const_p, epss, epss_eps_p, epss_const_p, output)


def draw_eps_perturbations(eps: Array, prng_key: Array, eps_p_mode: str) -> Array:
    """
    Draws random perturbations based on local truncation error.
    ...: Batch dimension.
    D: Latent dimension.
    N: ODE order.

    Args:
        eps (Array): Local truncation error [..., N, D].
        prng_key (Array): PRNG key.
        eps_p_mode (str): Mode used for eps-based perturbations.

    Raises:
        ValueError: Unknown perturbation mode.

    Returns:
        Array: Random perturbations [..., N, D].
    """

    match eps_p_mode:
        case "Diagonal":
            return eps * random.normal(prng_key, shape=eps.shape)
        case "Outer":
            eps_sqrt = jnp.sqrt(jnp.reshape(eps, eps.shape[:-2] + (-1,)))
            mean = jnp.zeros(eps.shape[:-2] + (eps.shape[-2] * eps.shape[-1],))
            cov = jnp.einsum("...i, ...j -> ...ij", eps_sqrt, eps_sqrt)

            out = random.multivariate_normal(prng_key, mean, cov, method="svd").reshape(*eps.shape)
            return out
        case _:
            raise ValueError(f"Unknown mode: {eps_p_mode}")


def unroll(
    solver: RKSolver,
    tN: Array,
    prng_key: Array,
    n_eps_p: int,
    eps_p_mode: str,
    n_const_p: int,
    const_p_val: float,
    save_interval: int,
    disable_pbar: bool,
) -> Tuple[Array, Array, Array | None, Array | None, Array, Array | None, Array | None]:
    """
    Unrolls trajectory.
    D: Latent dimension.
    N: ODE order.
    P: Perturbation dimension.
    T: Time dimension.

    Args:
        solver (RKSolver): ODE solver.
        tN (Array): End time.
        prng_key (Array): PRNG key.
        n_eps_p (int): Number of perturbed sample paths computed using local truncation error.
        eps_p_mode (str): Mode used for eps-based perturbations.
        n_const_p (int): Number of perturbed sample paths computed using constant.
        const_p_val (float): Constant used for perturbations.
        save_interval (int): Timestep interval in which results are saved.
        disable_pbar (bool): Disables progress bar.

    Returns:
        Tuple[Array, Array, Array | None, Array | None, Array, Array | None, Array | None]:
            Time [T],
            State [T, N, D],
            Eps-perturbed state [T, P, N, D],
            Const-perturbed state [T, P, N, D],
            Eps [T, N, D],
            Eps (eps-perturbed) [T, P, N, D],
            Eps (const-perturbed) [T, P, N, D].
    """

    ts = [solver.t0[0]]
    xs = [solver.x0[0]]
    if n_eps_p > 0:
        xs_eps_p = [jnp.broadcast_to(solver.x0, (n_eps_p,) + solver.x0.shape[-2:])]
        epss_eps_p = [jnp.zeros((n_eps_p,) + solver.x0.shape[-2:])]
    if n_const_p > 0:
        xs_const_p = [jnp.broadcast_to(solver.x0, (n_const_p,) + solver.x0.shape[-2:])]
        epss_const_p = [jnp.zeros((n_const_p,) + solver.x0.shape[-2:])]
    epss = [jnp.zeros(solver.x0.shape[-2:])]

    counter = 0
    t = solver.t0
    x = solver.x0
    h = solver.h
    pbar = tqdm(total=tN.item(), disable=disable_pbar, unit="sec")

    while jnp.any(t < tN):
        if solver.adaptive_control:
            t, x, eps, h = solver.step(t, x, h)  # [P, N, D]
        else:
            t, x, eps, _ = solver.step(t, x)  # [P, N, D]
        prng_key, subkey_1 = random.split(prng_key)
        prng_key, subkey_2 = random.split(prng_key)
        p = jnp.concatenate(
            [
                jnp.zeros((1,) + eps.shape[-2:]),
                draw_eps_perturbations(
                    jnp.broadcast_to(eps, (n_eps_p + n_const_p + 1,) + eps.shape[-2:])[
                        1 : n_eps_p + 1, ...
                    ],
                    subkey_1,
                    eps_p_mode,
                ),
                const_p_val * random.normal(subkey_2, shape=(n_const_p,) + eps.shape[-2:]),
            ],
            axis=0,
        )  # [P, N, D]
        x = x + p  # [P, N, D]

        if counter % save_interval == 0:
            eps_b = jnp.broadcast_to(eps, (n_eps_p + n_const_p + 1,) + eps.shape[-2:])
            xs.append(x[0])
            if n_eps_p > 0:
                xs_eps_p.append(x[1 : n_eps_p + 1])
                epss_eps_p.append(eps_b[1 : n_eps_p + 1])
            if n_const_p > 0:
                xs_const_p.append(x[1 + n_eps_p :])
                epss_const_p.append(eps_b[1 + n_eps_p :])
            epss.append(eps_b[0])
            ts.append(t[0])
        counter += 1
        pbar.update(t.item() - pbar.n)

    pbar.update(tN.item() - pbar.n)
    pbar.close()

    ts = jnp.stack(ts)
    xs = jnp.stack(xs)
    xs_eps_p = jnp.stack(xs_eps_p) if n_eps_p > 0 else None
    xs_const_p = jnp.stack(xs_const_p) if n_const_p > 0 else None
    epss = jnp.stack(epss)
    epss_eps_p = jnp.stack(epss_eps_p) if n_eps_p > 0 else None
    epss_const_p = jnp.stack(epss_const_p) if n_const_p > 0 else None

    return ts, xs, xs_eps_p, xs_const_p, epss, epss_eps_p, epss_const_p


def store(
    ts: Array,
    xs: Array,
    xs_eps_p: Array | None,
    xs_const_p: Array | None,
    epss: Array,
    epss_eps_p: Array | None,
    epss_const_p: Array | None,
    out_filepath: str,
) -> None:
    """
    Saves results on disk.
    D: Latent dimension.
    N: ODE order.
    P: Perturbation dimension.
    T: Time dimension.

    Args:
        ts (Array): Time [T].
        xs (Array): State [T, N, D].
        xs_eps_p (Array | None): Eps-perturbed state [T, P, N, D].
        xs_const_p (Array | None): Const-perturbed state [T, P, N, D].
        epss (Array): Eps (unperturbed state) [T, N, D].
        epss_eps_p (Array | None): Eps (eps-perturbed states) [T, P, N, D].
        epss_const_p (Array | None): Eps (const-perturbed states) [T, P, N, D].
        out_filepath (str): Path to H5 results file.
    """

    Path(out_filepath).parent.mkdir(parents=True, exist_ok=True)
    h5f = h5py.File(out_filepath, "w")
    h5f.create_dataset("ts", data=ts)
    h5f.create_dataset("xs", data=xs)
    if xs_eps_p is not None:
        h5f.create_dataset("xs_eps_p", data=xs_eps_p)
    if xs_const_p is not None:
        h5f.create_dataset("xs_const_p", data=xs_const_p)
    h5f.create_dataset("epss", data=epss)
    if epss_eps_p is not None:
        h5f.create_dataset("epss_eps_p", data=epss_eps_p)
    if epss_const_p is not None:
        h5f.create_dataset("epss_const_p", data=epss_const_p)
    h5f.close()


if __name__ == "__main__":
    CLI(main, as_positional=False)
