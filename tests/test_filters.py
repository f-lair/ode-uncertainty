import pytest
from jax import numpy as jnp

from src.filters import EKF
from src.filters.sigma_fns import DiagonalSigma, OuterSigma
from src.ode import Logistic, RLCCircuit
from src.solvers import RKF45


def test_ekf_diag_rkf45_logistic():
    t0 = 0.0
    tN = 10.0
    dt = 0.1
    x0 = [[0.01]]
    P0 = [[0.0]]

    ode = Logistic()
    t = jnp.array(t0)
    x = jnp.array(x0)
    P = jnp.array(P0)
    solver = RKF45(ode, t, x, jnp.array(dt))
    filter_ = EKF(solver, P, DiagonalSigma())

    ts = jnp.arange(t0, tN, dt)

    xs = [jnp.array(x0)]
    Ps = [jnp.array(P0)]
    for _ in range(len(ts) - 1):
        t, x, P = filter_.predict()
        xs.append(x)
        Ps.append(P)
    xs = jnp.concatenate(xs)
    Ps = jnp.stack(Ps)

    xs_correct = ode.Fn(ts, jnp.array(x0))

    assert jnp.allclose(xs, xs_correct)


def test_ekf_diag_rlc_circuit_1():
    t0 = 0.0
    tN = 1.0
    dt = 0.01
    x0 = [[10.0], [0.0]]
    P0 = [[0.0, 0.0], [0.0, 0.0]]

    ode = RLCCircuit(
        resistance=jnp.array(2500), inductance=jnp.array(400), capacitance=jnp.array(2.5e-5)
    )
    t = jnp.array(t0)
    x = jnp.array(x0)
    P = jnp.array(P0)
    solver = RKF45(ode, t, x, jnp.array(dt))
    filter_ = EKF(solver, P, DiagonalSigma())

    ts = jnp.arange(t0, tN, dt)

    xs = [jnp.array(x0)]
    Ps = [jnp.array(P0)]
    for _ in range(len(ts) - 1):
        t, x, P = filter_.predict()
        xs.append(x)
        Ps.append(P)
    xs = jnp.stack(xs)[:, 0]
    Ps = jnp.stack(Ps)

    xs_correct = ode.Fn(ts, jnp.array(x0))

    assert jnp.allclose(xs, xs_correct, rtol=1e-4, atol=1e-7)
