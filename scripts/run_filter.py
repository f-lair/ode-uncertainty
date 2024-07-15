import ast
import sys
from argparse import ArgumentParser
from typing import Tuple

sys.path.append("../")

import h5py
from jax import Array
from jax import numpy as jnp
from tqdm import tqdm

from src.filters import EKF, UKF, UKF_SQRT
from src.filters.filter import Filter
from src.filters.sigma_fns import DiagonalSigma, OuterSigma
from src.ode import LCAO, Lorenz, LotkaVolterra, VanDerPol
from src.solvers import RKF45


def main() -> None:
    parser = ArgumentParser(
        "ODE Filter Script",
        description="Runs embedded RK ODE filter.\nN: ODE order\nD: Latent dimension",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="EKF",
        help="Filter: EKF (default), UKF, UKF-SQRT.",
        choices=["EKF", "UKF", "UKF-SQRT"],
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
        help="ODE: Lorenz (default), VanDerPol, LCAO, LotkaVolterra.",
        choices=["Lorenz", "VanDerPol", "LCAO", "LotkaVolterra"],
    )
    parser.add_argument(
        "--sigma-fn",
        type=str,
        default="Diagonal",
        help="Sigma function: Diagonal (default), Outer.",
        choices=["Diagonal", "Outer"],
    )
    parser.add_argument("--x0", type=str, required=True, help="Initial value [N, D].")
    parser.add_argument(
        "--P0", type=str, default="", help="Initial uncertainty [N*D, N*D]. Defaults to zero."
    )
    parser.add_argument("--t0", type=float, required=True, help="Initial time.")
    parser.add_argument("--tN", type=float, required=True, help="Final time.")
    parser.add_argument("--dt", type=float, help="Final time.")
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
        case "LCAO":
            ode = LCAO()
        case "LotkaVolterra":
            ode = LotkaVolterra()
        case _:
            raise ValueError(f"Unknown ODE: {args.ode}")

    match args.sigma_fn:
        case "Diagonal":
            sigma_fn = DiagonalSigma()
        case "Outer":
            sigma_fn = OuterSigma()
        case _:
            raise ValueError(f"Unknown sigma function: {args.sigma_fn}")

    x0 = jnp.array([ast.literal_eval(args.x0)])
    if args.P0 == "":
        P0 = jnp.zeros((x0.size, x0.size))
    else:
        P0 = jnp.array([ast.literal_eval(args.P0)])
        assert x0.size == P0.shape[0] and x0.size == P0.shape[1]
    t0 = jnp.array([args.t0])
    tN = jnp.array(args.tN)
    dt = jnp.array(args.dt) if args.dt is not None else None
    save_interval = args.save_interval
    out_filepath = args.output
    disable_pbar = args.disable_pbar

    solver = solver_cls(ode, t0, x0, dt, adaptive_control=False)

    match args.filter:
        case "EKF":
            filter_ = EKF(solver, P0, sigma_fn)
        case "UKF":
            filter_ = UKF(solver, P0, sigma_fn)
        case "UKF-SQRT":
            filter_ = UKF_SQRT(solver, P0, sigma_fn)
        case _:
            raise ValueError(f"Unknown filter: {args.filter}")

    ts, xs, Ps, dx_dts, jacs, sigmas = unroll(
        filter_,
        tN,
        save_interval,
        disable_pbar,
    )
    store(ts, xs, Ps, dx_dts, jacs, sigmas, out_filepath)


def unroll(
    filter_: Filter,
    tN: Array,
    save_interval: int,
    disable_pbar: bool,
) -> Tuple[Array, Array, Array, Array, Array | None, Array | None]:
    ts = [filter_.t[0]]
    xs = [filter_.m[0]]
    Ps = [filter_.P]
    dx_dts = [filter_.rk_solver.fn(filter_.t, filter_.m)[0]]

    counter = 0
    t = filter_.t
    x = filter_.m
    P = filter_.P
    pbar = tqdm(total=tN.item(), disable=disable_pbar, unit="sec")

    while jnp.any(t < tN):
        t, x, P = filter_.predict()  # [1], [1, N, D], [N*D, N*D]
        dx_dt = filter_.rk_solver.fn(t, x)  # [1, N, D]

        if counter % save_interval == 0:
            xs.append(x[0])
            ts.append(t[0])
            Ps.append(P)
            dx_dts.append(dx_dt[0])
        counter += 1
        pbar.update(t.item() - pbar.n)

    pbar.update(tN.item() - pbar.n)
    pbar.close()

    ts = jnp.stack(ts)
    xs = jnp.stack(xs)
    Ps = jnp.stack(Ps)
    dx_dts = jnp.stack(dx_dts)
    if isinstance(filter_, EKF):
        jacs = jnp.stack(filter_.jac_buffer)
        sigmas = jnp.stack(filter_.sigma_buffer)
    else:
        jacs, sigmas = None, None

    return ts, xs, Ps, dx_dts, jacs, sigmas


def store(
    ts: Array,
    xs: Array,
    Ps: Array,
    dx_dts: Array,
    jacs: Array | None,
    sigmas: Array | None,
    out_filepath: str,
) -> None:
    h5f = h5py.File(out_filepath, "w")
    h5f.create_dataset("ts", data=ts)
    h5f.create_dataset("xs", data=xs)
    h5f.create_dataset("Ps", data=Ps)
    h5f.create_dataset("dx_dts", data=dx_dts)
    if jacs is not None:
        h5f.create_dataset("Jacs", data=jacs)
    if sigmas is not None:
        h5f.create_dataset("Sigmas", data=sigmas)
    h5f.close()


if __name__ == "__main__":
    main()
