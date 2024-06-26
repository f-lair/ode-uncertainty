import ast
import sys
from argparse import ArgumentParser
from typing import Tuple

sys.path.append("../")

import h5py
from jax import Array
from jax import numpy as jnp
from jax import random
from tqdm import tqdm

from src.ode import Lorenz, VanDerPol
from src.solvers import RKF45
from src.solvers.rksolver import RKSolver


def main() -> None:
    parser = ArgumentParser(
        "ODE Solver Script",
        description="Runs embedded RK ODE solver.\nN: ODE order\nD: Latent dimension",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="RKF45",
        help="Embedded RK solver: RKF45 (default).",
        choices=["RKF45"],
    )
    parser.add_argument(
        "--ode",
        type=str,
        default="Lorenz",
        help="ODE: Lorenz (default), VanDerPol.",
        choices=["Lorenz", "VanDerPol"],
    )
    parser.add_argument("--x0", type=str, required=True, help="Initial value [N, D].")
    parser.add_argument("--t0", type=float, required=True, help="Initial time.")
    parser.add_argument("--tN", type=float, required=True, help="Final time.")
    parser.add_argument("--dt", type=float, help="Final time.")
    parser.add_argument(
        "--eps-perturbations",
        type=int,
        default=0,
        help="Number of perturbed sample paths computed using local truncation error.",
    )
    parser.add_argument(
        "--const-perturbations",
        type=int,
        default=0,
        help="Number of perturbed sample paths computed using constant.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed used for random perturbations.",
    )
    parser.add_argument(
        '--adaptive', action="store_true", help="Activates adaptive step size control."
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance for adaptive step size control.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-8,
        help="Absolute tolerance for adaptive step size control.",
    )
    parser.add_argument(
        "--save-interval", type=int, default=1, help="Interval in which solutions are saved."
    )
    parser.add_argument("-o", "--output", type=str, required=True, help="Output filepath (.h5).")
    parser.add_argument('--disable-pbar', action="store_true", help="Deactivates progress bar.")
    args = parser.parse_args()

    match args.solver:
        case "RKF45":
            solver_cls = RKF45
        case _:
            raise ValueError(f"Unknown solver: {args.solver}")

    match args.ode:
        case "Lorenz":
            ode = Lorenz()
        case "VanDerPol":
            ode = VanDerPol(jnp.array(5))
        case _:
            raise ValueError(f"Unknown ODE: {args.ode}")

    x0 = jnp.array([ast.literal_eval(args.x0)])
    t0 = jnp.array([args.t0])
    tN = jnp.array(args.tN)
    dt = jnp.array(args.dt) if args.dt is not None else None
    prng_key = random.key(args.seed)
    n_eps_p = args.eps_perturbations
    n_const_p = args.const_perturbations
    adaptive = args.adaptive
    rtol = args.rtol
    atol = args.atol
    save_interval = args.save_interval
    out_filepath = args.output
    disable_pbar = args.disable_pbar

    solver = solver_cls(ode, t0, x0, dt, adaptive, rtol=rtol, atol=atol)
    ts, xs, xs_eps_p, xs_const_p, epss, epss_eps_p, epss_const_p = unroll(
        solver, tN, prng_key, n_eps_p, n_const_p, save_interval, disable_pbar
    )
    store(ts, xs, xs_eps_p, xs_const_p, epss, epss_eps_p, epss_const_p, out_filepath)


def unroll(
    solver: RKSolver,
    tN: Array,
    prng_key: Array,
    n_eps_p: int,
    n_const_p: int,
    save_interval: int,
    disable_pbar: bool,
) -> Tuple[Array, Array, Array | None, Array | None, Array, Array | None, Array | None]:
    ts = [solver.t[0]]
    xs = [solver.x[0]]
    if n_eps_p > 0:
        xs_eps_p = [jnp.broadcast_to(solver.x, (n_eps_p,) + solver.x.shape[-2:])]
        epss_eps_p = [jnp.zeros((n_eps_p,) + solver.x.shape[-2:])]
    if n_const_p > 0:
        xs_const_p = [jnp.broadcast_to(solver.x, (n_const_p,) + solver.x.shape[-2:])]
        epss_const_p = [jnp.zeros((n_const_p,) + solver.x.shape[-2:])]
    epss = [jnp.zeros(solver.x.shape[-2:])]

    counter = 0
    pbar = tqdm(total=tN.item(), disable=disable_pbar, unit="sec")

    while jnp.any(solver.t < tN):
        t, x, eps = solver.step()  # [P, N, D]
        prng_key, subkey_1 = random.split(prng_key)
        prng_key, subkey_2 = random.split(prng_key)
        p = jnp.concatenate(
            [
                jnp.zeros((1,) + eps.shape[-2:]),
                (eps * random.normal(subkey_1, shape=(n_eps_p + n_const_p + 1,) + eps.shape[-2:]))[
                    1 : n_eps_p + 1, ...
                ],
                0.1 * solver.h * random.normal(subkey_2, shape=(n_const_p,) + eps.shape[-2:]),
            ],
            axis=0,
        )  # [P, N, D]
        x_p = x + p  # [P, N, D]
        solver.x = x_p

        if counter % save_interval == 0:
            eps_b = jnp.broadcast_to(eps, (n_eps_p + n_const_p + 1,) + eps.shape[-2:])
            xs.append(x_p[0])
            if n_eps_p > 0:
                xs_eps_p.append(x_p[1 : n_eps_p + 1])
                epss_eps_p.append(eps_b[1 : n_eps_p + 1])
            if n_const_p > 0:
                xs_const_p.append(x_p[1 + n_eps_p :])
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
    main()
