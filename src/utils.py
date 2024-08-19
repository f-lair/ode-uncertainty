import jax
from jax import Array
from jax import numpy as jnp
from jax import scipy as jsp


@jax.jit
def bmmT(a: Array, b: Array) -> Array:
    """
    Computes batched matrix multiplication with second matrix transposed.

    Args:
        a (Array): First matrix [..., N, M].
        b (Array): Second matix [..., L, M].

    Returns:
        Array: Batched matrix product  [..., N, L].
    """

    return jnp.einsum("...ij,...kj->...ik", a, b)


@jax.jit
def sqrt_L_sum_qr(a: Array, b: Array) -> Array:
    """
    Computes lower-triangular (a+b)^(1/2), where a, b are lower-triangular matrix square-roots,
    using QR decomposition.

    Args:
        a (Array): Lower-triangular matrix square-root [..., N, N].
        b (Array): Lower-triangular matrix square-root [..., N, N].

    Returns:
        Array: Lower-triangular matrix square-root [..., N, N].
    """

    t_s = tuple(range(a.ndim - 2)) + (a.ndim - 1, a.ndim - 2)
    a_b, b_b = jnp.broadcast_arrays(a, b)

    r = jsp.linalg.qr(
        jnp.concatenate([a_b.transpose(*t_s), b_b.transpose(*t_s)], axis=-2),
        mode="economic",
    )[
        1
    ]  # [..., N, N]
    return r.transpose(*t_s)


@jax.jit
def multivariate_normal_sqrt(x: Array, m: Array, S: Array) -> Array:
    """
    Evaluates PDF of multivariate Gaussian efficiently using pre-computed covariance square-root.

    Args:
        x (Array): Input [..., N].
        m (Array): Mean [..., N].
        S (Array): Lower-triangular covariance square-root [..., N, N].

    Returns:
        Array: PDF evaluation result [...].
    """

    n = m.shape[-1]
    y = x - m
    b_shape = jnp.broadcast_shapes(y.shape[:-1], S.shape[:-2])
    y_b = jnp.broadcast_to(y, b_shape + (n,))
    S_b = jnp.broadcast_to(S, b_shape + (n, n))

    y = jsp.linalg.solve_triangular(S_b, y_b, lower=True)  # [..., N]
    log_pdf = (
        -1 / 2 * jnp.einsum('...i,...i->...', y, y)
        - n / 2 * jnp.log(2 * jnp.pi)
        - jnp.log(jnp.abs(S.diagonal(axis1=-1, axis2=-2))).sum(-1)
    )  # [...]
    return jnp.exp(log_pdf)


@jax.jit
def kl_divergence_gaussian_sqrt(m_p: Array, m_q: Array, S_p: Array, S_q: Array) -> Array:
    """
    Evaluates KL divergence KL(P||Q) between pairs of Gaussians P, Q with covariance square-root.

    Args:
        m_p (Array): Mean of Gaussian P [..., N].
        m_q (Array): Mean of Gaussian Q [..., N].
        S_p (Array): Square-root covariance of Gaussian P [..., N].
        S_q (Array): Square-root covariance of Gaussian Q [..., N].

    Returns:
        Array: KL divergence KL(P||Q) [...].
    """

    n = m_p.shape[-1]
    y = m_q - m_p
    b_shape = jnp.broadcast_shapes(y.shape[:-1], S_p.shape[:-2], S_q.shape[:-2])
    y_b = jnp.broadcast_to(y, b_shape + (n,))
    S_p_b = jnp.broadcast_to(S_p, b_shape + (n, n))
    S_q_b = jnp.broadcast_to(S_q, b_shape + (n, n))
    t_s = tuple(range(S_p.ndim - 2)) + (S_p.ndim - 1, S_p.ndim - 2)

    y = jsp.linalg.solve_triangular(S_q_b, y_b, lower=True)  # [..., N]
    tr_qp = (
        (jsp.linalg.cho_solve((S_q_b, True), S_p_b) @ S_p_b.transpose(*t_s))
        .diagonal(axis1=-1, axis2=-2)
        .sum(-1)
    )
    log_det_p = jnp.log(jnp.abs(S_p_b.diagonal(axis1=-1, axis2=-2)) + 1e-8).sum(-1)
    log_det_q = jnp.log(jnp.abs(S_q_b.diagonal(axis1=-1, axis2=-2)) + 1e-8).sum(-1)

    return 1 / 2 * (log_det_q - log_det_p - n + jnp.einsum('...i,...i->...', y, y) + tr_qp)


@jax.jit
def jeffrey_divergence_sqrt(m_1: Array, m_2: Array, S_1: Array, S_2: Array) -> Array:
    """Evaluates symmetric KL divergence (aka Jeffrey divergence) between pairs of Gaussians with
    covariance square-root.

    Args:
        m_1 (Array): Mean of Gaussian 1 [..., N].
        m_2 (Array): Mean of Gaussian 2 [..., N].
        S_1 (Array): Square-root covariance of Gaussian 1 [..., N].
        S_2 (Array): Square-root covariance of Gaussian 2 [..., N].

    Returns:
        Array: Jeffrey divergence [...].
    """

    return kl_divergence_gaussian_sqrt(m_1, m_2, S_1, S_2) + kl_divergence_gaussian_sqrt(
        m_2, m_1, S_2, S_1
    )
