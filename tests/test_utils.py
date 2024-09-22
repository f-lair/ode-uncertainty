import jax
import pytest

jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
from jax import random
from jax import scipy as jsp

# from src.filters.deprecated.gmm_ekf import bmmT, merge_refit
from src.solvers import RKF45
from src.utils import *


@pytest.fixture
def rand_10x10():
    key = random.key(7)
    return random.normal(key, (10, 10))


@pytest.fixture
def rand_10x10_L(rand_10x10):
    return jnp.linalg.cholesky(rand_10x10 @ rand_10x10.T)


def test_sqrt_L_sum_qr(rand_10x10, rand_10x10_L):
    a = rand_10x10 @ rand_10x10_L
    b = jnp.diag(jnp.diag(rand_10x10) ** 2)
    c = sqrt_L_sum_qr(a, b**0.5)

    x = rand_10x10 @ (rand_10x10_L @ rand_10x10_L.T) @ rand_10x10.T + b
    c_correct = jnp.linalg.cholesky(x)

    assert jnp.allclose(c @ c.T, c_correct @ c_correct.T)


def test_sqrt_L_sum_qr_zero(rand_10x10, rand_10x10_L):
    a = rand_10x10_L
    b = jnp.zeros_like(a)
    c = sqrt_L_sum_qr(a, b)

    c_correct = jnp.linalg.cholesky(rand_10x10 @ rand_10x10.T)

    assert jnp.allclose(c @ c.T, c_correct @ c_correct.T)


def test_multivariate_normal_sqrt(rand_10x10, rand_10x10_L):
    x = rand_10x10[0, :]
    m = rand_10x10[1, :]
    S = rand_10x10_L
    p = multivariate_normal_sqrt(x, m, S)

    p_correct = jsp.stats.multivariate_normal.pdf(x, m, S @ S.T)

    assert jnp.allclose(p, p_correct)


def test_kl_divergence_gaussian_sqrt(rand_10x10, rand_10x10_L):
    m1 = rand_10x10[0, :]
    m2 = rand_10x10[1, :]
    S1 = rand_10x10_L
    S2 = rand_10x10_L

    kl_d = kl_divergence_gaussian_sqrt(m1, m2, S1, S2)

    n = m1.shape[0]
    P1 = S1 @ S1.T
    P2 = S2 @ S2.T
    P2_inv = jnp.linalg.inv(P2)
    kl_d_correct = 0.5 * (
        jnp.log(jnp.linalg.det(P2) / jnp.linalg.det(P1))
        - n
        + (m2 - m1) @ P2_inv @ (m2 - m1)
        + jnp.trace(P2_inv @ P1)
    )
    assert jnp.allclose(kl_d, kl_d_correct)
