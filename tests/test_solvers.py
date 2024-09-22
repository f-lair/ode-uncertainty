import jax
import pytest

jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

from src.ode import Logistic, RLCCircuit
from src.solvers import RKF45


def test_rkf45_logistic():
    t0 = 0.0
    tN = 10.0
    h = 0.1
    x0 = [[0.01]]

    t0_arr = jnp.array(t0)
    x0_arr = jnp.array(x0)

    ode_builder = Logistic()
    solver_builder = RKF45(step_size=h)

    ode = ode_builder.build()
    ode_solution = ode_builder.build_solution()
    solver_builder.setup(ode, ode_builder.params)
    solver = jax.jit(solver_builder.build())

    state_def = solver_builder.state_def(*x0_arr.shape)
    state = {k: jnp.zeros(v) for k, v in state_def.items()}
    state["t"] = t0_arr
    state["x"] = x0_arr

    ts = jnp.arange(t0, tN, h)
    xs = [x0_arr[:, 0]]

    for _ in range(len(ts) - 1):
        state = solver(state)
        xs.append(state["x"][:, 0])
    xs = jnp.stack(xs)

    xs_correct = ode_solution(ts, x0_arr, ode_builder.params)

    assert jnp.allclose(xs, xs_correct)


def test_rkf45_rlc_circuit_1():
    t0 = 0.0
    tN = 1.0
    h = 0.01
    x0 = [[10.0], [0.0]]

    t0_arr = jnp.array(t0)
    x0_arr = jnp.array(x0)

    ode_builder = RLCCircuit(resistance=2500.0, inductance=400.0, capacitance=2.5e-5)
    solver_builder = RKF45(step_size=h)

    ode = ode_builder.build()
    ode_solution = ode_builder.build_solution()
    solver_builder.setup(ode, ode_builder.params)
    solver = jax.jit(solver_builder.build())

    state_def = solver_builder.state_def(*x0_arr.shape)
    state = {k: jnp.zeros(v) for k, v in state_def.items()}
    state["t"] = t0_arr
    state["x"] = x0_arr

    ts = jnp.arange(t0, tN, h)
    xs = [x0_arr[0, :]]

    for _ in range(len(ts) - 1):
        state = solver(state)
        xs.append(state["x"][0, :])
    xs = jnp.stack(xs)

    xs_correct = ode_solution(ts, x0_arr, ode_builder.params)

    assert jnp.allclose(xs, xs_correct, rtol=1e-4, atol=1e-7)


def test_rkf45_rlc_circuit_2():
    t0 = 0.0
    tN = 1.0
    h = 0.01
    x0 = [[10.0], [0.0]]

    t0_arr = jnp.array(t0)
    x0_arr = jnp.array(x0)

    ode_builder = RLCCircuit(resistance=4000.0, inductance=160.0, capacitance=4e-5)
    solver_builder = RKF45(step_size=h)

    ode = ode_builder.build()
    ode_solution = ode_builder.build_solution()
    solver_builder.setup(ode, ode_builder.params)
    solver = jax.jit(solver_builder.build())

    state_def = solver_builder.state_def(*x0_arr.shape)
    state = {k: jnp.zeros(v) for k, v in state_def.items()}
    state["t"] = t0_arr
    state["x"] = x0_arr

    ts = jnp.arange(t0, tN, h)
    xs = [x0_arr[0, :]]

    for _ in range(len(ts) - 1):
        state = solver(state)
        xs.append(state["x"][0, :])
    xs = jnp.stack(xs)

    xs_correct = ode_solution(ts, x0_arr, ode_builder.params)

    assert jnp.allclose(xs, xs_correct, rtol=1e-4, atol=1e-7)


def test_rkf45_rlc_circuit_3():
    t0 = 0.0
    tN = 1.0
    h = 0.01
    x0 = [[10.0], [0.0]]

    t0_arr = jnp.array(t0)
    x0_arr = jnp.array(x0)

    ode_builder = RLCCircuit(resistance=5000.0, inductance=160.0, capacitance=4e-5)
    solver_builder = RKF45(step_size=h)

    ode = ode_builder.build()
    ode_solution = ode_builder.build_solution()
    solver_builder.setup(ode, ode_builder.params)
    solver = jax.jit(solver_builder.build())

    state_def = solver_builder.state_def(*x0_arr.shape)
    state = {k: jnp.zeros(v) for k, v in state_def.items()}
    state["t"] = t0_arr
    state["x"] = x0_arr

    ts = jnp.arange(t0, tN, h)
    xs = [x0_arr[0, :]]

    for _ in range(len(ts) - 1):
        state = solver(state)
        xs.append(state["x"][0, :])
    xs = jnp.stack(xs)

    xs_correct = ode_solution(ts, x0_arr, ode_builder.params)

    assert jnp.allclose(xs, xs_correct, rtol=1e-4, atol=1e-7)
