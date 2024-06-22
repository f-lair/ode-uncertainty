import ast
import sys
from argparse import ArgumentParser
from typing import Tuple

sys.path.append("../")

import h5py
from jax import Array
from jax import numpy as jnp
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

    x0 = jnp.array(ast.literal_eval(args.x0))
    t0 = jnp.array(args.t0)
    tN = jnp.array(args.tN)
    dt = jnp.array(args.dt) if args.dt is not None else None
    adaptive = args.adaptive
    rtol = args.rtol
    atol = args.atol
    save_interval = args.save_interval
    out_filepath = args.output
    disable_pbar = args.disable_pbar

    solver = solver_cls(ode, t0, x0, dt, adaptive, rtol=rtol, atol=atol)
    ts, xs, epss = unroll(solver, tN, save_interval, disable_pbar)
    store(ts, xs, epss, out_filepath)


def unroll(
    solver: RKSolver, tN: Array, save_interval: int, disable_pbar: bool
) -> Tuple[Array, Array, Array]:
    ts = [solver.t]
    xs = [solver.x]
    epss = [jnp.zeros(solver.x.shape[-2:])]

    counter = 0
    pbar = tqdm(total=tN.item(), disable=disable_pbar, unit="sec")

    while solver.t < tN:
        t, x, eps = solver.step()  # [N, D]
        if counter % save_interval == 0:
            xs.append(x)
            epss.append(eps)
            ts.append(t)
        counter += 1
        pbar.update(t.item() - pbar.n)

    pbar.update(tN.item() - pbar.n)
    pbar.close()

    return jnp.stack(ts), jnp.stack(xs), jnp.stack(epss)


def store(ts: Array, xs: Array, epss: Array, out_filepath: str) -> None:
    h5f = h5py.File(out_filepath, "w")
    h5f.create_dataset("t", data=ts)
    h5f.create_dataset("x", data=xs)
    h5f.create_dataset("eps", data=epss)
    h5f.close()


if __name__ == "__main__":
    main()
