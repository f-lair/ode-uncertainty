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
    solver = RKF45(ode, jnp.array(t0), jnp.array(x0), jnp.array(dt))

    ts = jnp.arange(t0, tN, dt)

    xs = [jnp.array(x0)]
    for _ in range(len(ts) - 1):
        _, x, _ = solver.step()
        xs.append(x)
    xs = jnp.concatenate(xs)

    xs_correct = ode.Fn(ts, jnp.array(x0))

    assert jnp.allclose(xs, xs_correct)


def test_rkf45_logistic_auto_adaptive():
    t0 = 0.0
    tN = 10.0
    x0 = [[0.01]]

    ode = Logistic()
    solver = RKF45(ode, jnp.array(t0), jnp.array(x0), adaptive_control=True)

    ts = [jnp.array(t0)]
    xs = [jnp.array(x0)]
    while jnp.any(solver.t < tN):
        t, x, _ = solver.step()
        xs.append(x)
        ts.append(t)
    xs = jnp.concatenate(xs)
    ts = jnp.stack(ts)

    xs_correct = ode.Fn(ts, jnp.array(x0))

    assert jnp.allclose(xs, xs_correct)


def test_rkf45_rlc_circuit_1():
    t0 = 0.0
    tN = 1.0
    dt = 0.01
    x0 = [[10.0], [0.0]]

    ode = RLCCircuit(resistance=2500, inductance=400, capacitance=2.5e-5)
    solver = RKF45(ode, jnp.array(t0), jnp.array(x0), jnp.array(dt))

    ts = jnp.arange(t0, tN, dt)

    xs = [jnp.array(x0)]
    for _ in range(len(ts) - 1):
        _, x, _ = solver.step()
        xs.append(x)
    xs = jnp.stack(xs)[:, 0]

    xs_correct = ode.Fn(ts, jnp.array(x0))

    assert jnp.allclose(xs, xs_correct, rtol=1e-4, atol=1e-7)


def test_rkf45_rlc_circuit_1_auto_adaptive():
    t0 = 0.0
    tN = 1.0
    x0 = [[10.0], [0.0]]

    ode = RLCCircuit(resistance=2500, inductance=400, capacitance=2.5e-5)
    solver = RKF45(ode, jnp.array(t0), jnp.array(x0), adaptive_control=True)

    ts = [jnp.array(t0)]
    xs = [jnp.array(x0)]
    while jnp.any(solver.t < tN):
        t, x, _ = solver.step()
        xs.append(x)
        ts.append(t)
    xs = jnp.stack(xs)[:, 0]
    ts = jnp.stack(ts)

    xs_correct = ode.Fn(ts, jnp.array(x0))

    assert jnp.allclose(xs, xs_correct, rtol=1e-3, atol=1e-6)


def test_rkf45_rlc_circuit_2():
    t0 = 0.0
    tN = 1.0
    dt = 0.01
    x0 = [[10.0], [0.0]]

    ode = RLCCircuit(resistance=4000, inductance=160, capacitance=4e-5)
    solver = RKF45(ode, jnp.array(t0), jnp.array(x0), jnp.array(dt))

    ts = jnp.arange(t0, tN, dt)

    xs = [jnp.array(x0)]
    for _ in range(len(ts) - 1):
        _, x, _ = solver.step()
        xs.append(x)
    xs = jnp.stack(xs)[:, 0]

    xs_correct = ode.Fn(ts, jnp.array(x0))

    assert jnp.allclose(xs, xs_correct, rtol=1e-4, atol=1e-7)


def test_rkf45_rlc_circuit_2_auto_adaptive():
    t0 = 0.0
    tN = 1.0
    x0 = [[10.0], [0.0]]

    ode = RLCCircuit(resistance=4000, inductance=160, capacitance=4e-5)
    solver = RKF45(ode, jnp.array(t0), jnp.array(x0), adaptive_control=True)

    ts = [jnp.array(t0)]
    xs = [jnp.array(x0)]
    while jnp.any(solver.t < tN):
        t, x, _ = solver.step()
        xs.append(x)
        ts.append(t)
    xs = jnp.stack(xs)[:, 0]
    ts = jnp.stack(ts)

    xs_correct = ode.Fn(ts, jnp.array(x0))

    assert jnp.allclose(xs, xs_correct, rtol=1e-3, atol=1e-6)


def test_rkf45_rlc_circuit_3():
    t0 = 0.0
    tN = 1.0
    dt = 0.01
    x0 = [[10.0], [0.0]]

    ode = RLCCircuit(resistance=5000, inductance=160, capacitance=4e-5)
    solver = RKF45(ode, jnp.array(t0), jnp.array(x0), jnp.array(dt))

    ts = jnp.arange(t0, tN, dt)

    xs = [jnp.array(x0)]
    for _ in range(len(ts) - 1):
        _, x, _ = solver.step()
        xs.append(x)
    xs = jnp.stack(xs)[:, 0]

    xs_correct = ode.Fn(ts, jnp.array(x0))

    assert jnp.allclose(xs, xs_correct, rtol=1e-4, atol=1e-7)


def test_rkf45_rlc_circuit_3_auto_adaptive():
    t0 = 0.0
    tN = 1.0
    x0 = [[10.0], [0.0]]

    ode = RLCCircuit(resistance=5000, inductance=160, capacitance=4e-5)
    solver = RKF45(ode, jnp.array(t0), jnp.array(x0), adaptive_control=True)

    ts = [jnp.array(t0)]
    xs = [jnp.array(x0)]
    while jnp.any(solver.t < tN):
        t, x, _ = solver.step()
        xs.append(x)
        ts.append(t)
    xs = jnp.stack(xs)[:, 0]
    ts = jnp.stack(ts)

    xs_correct = ode.Fn(ts, jnp.array(x0))

    assert jnp.allclose(xs, xs_correct, rtol=1e-3, atol=1e-6)


def test_rk_batched():
    t0 = 0.0
    tN = [10.0, 20.0]
    dt = [0.01, 0.02]
    x0 = [[[0.01]], [[0.01]]]

    ode = Logistic()
    solver = RKF45(ode, jnp.array(t0), jnp.array(x0), jnp.array(dt))

    ts = jnp.stack([jnp.arange(t0, tN[0], dt[0]), jnp.arange(t0, tN[1], dt[1])], axis=1)

    xs = [jnp.array(x0)]
    for _ in range(len(ts) - 1):
        _, x, _ = solver.step()
        xs.append(x)
    xs = jnp.stack(xs)[:, :, 0]

    xs_correct = ode.Fn(ts, jnp.array(x0))

    assert jnp.allclose(xs, xs_correct)
