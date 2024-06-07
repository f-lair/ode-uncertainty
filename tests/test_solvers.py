import pytest
from jax import numpy as jnp

from src.ode import Logistic, RLCCircuit
from src.solvers import RKF45


def test_rkf45_logistic():
    t0 = 0.0
    tN = 10.0
    dt = 0.1
    x0 = [[0.01]]

    ode = Logistic()
    solver = RKF45(dt)

    x0 = jnp.array(x0)
    ts = jnp.arange(t0, tN, dt)

    x = x0
    xs = [x]
    for t in ts[:-1]:
        x, _ = solver.step(ode, t, x)
        xs.append(x)
    xs = jnp.concatenate(xs)

    xs_correct = ode.Fn(ts, x0)

    assert jnp.allclose(xs, xs_correct)


def test_rkf45_rlc_circuit_1():
    t0 = 0.0
    tN = 1.0
    dt = 0.01
    x0 = [[10.0], [0.0]]

    ode = RLCCircuit(resistance=2500, inductance=400, capacitance=2.5e-5)
    solver = RKF45(dt)

    x0 = jnp.array(x0)
    ts = jnp.arange(t0, tN, dt)

    x = x0
    xs = [x]
    for t in ts[:-1]:
        x, _ = solver.step(ode, t, x)
        xs.append(x)
    xs = jnp.stack(xs)[:, 0]

    xs_correct = ode.Fn(ts, x0)

    assert jnp.allclose(xs, xs_correct, rtol=1e-4, atol=1e-7)


def test_rkf45_rlc_circuit_2():
    t0 = 0.0
    tN = 1.0
    dt = 0.01
    x0 = [[10.0], [0.0]]

    ode = RLCCircuit(resistance=4000, inductance=160, capacitance=4e-5)
    solver = RKF45(dt)

    x0 = jnp.array(x0)
    ts = jnp.arange(t0, tN, dt)

    x = x0
    xs = [x]
    for t in ts[:-1]:
        x, _ = solver.step(ode, t, x)
        xs.append(x)
    xs = jnp.stack(xs)[:, 0]

    xs_correct = ode.Fn(ts, x0)

    assert jnp.allclose(xs, xs_correct, rtol=1e-4, atol=1e-7)


def test_rkf45_rlc_circuit_3():
    t0 = 0.0
    tN = 1.0
    dt = 0.01
    x0 = [[10.0], [0.0]]

    ode = RLCCircuit(resistance=5000, inductance=160, capacitance=4e-5)
    solver = RKF45(dt)

    x0 = jnp.array(x0)
    ts = jnp.arange(t0, tN, dt)

    x = x0
    xs = [x]
    for t in ts[:-1]:
        x, _ = solver.step(ode, t, x)
        xs.append(x)
    xs = jnp.stack(xs)[:, 0]

    xs_correct = ode.Fn(ts, x0)

    assert jnp.allclose(xs, xs_correct, rtol=1e-4, atol=1e-7)
