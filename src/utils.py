from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import h5py
import jax
import optax
import optax.tree_utils as otu
from jax import Array
from jax import numpy as jnp
from jax import scipy as jsp
from jax.flatten_util import ravel_pytree


def run_lbfgs_projected(init_params, bounds, fun, opt, max_iter, tol, **kwargs):
    fun_p = partial(fun, **kwargs)
    value_and_grad_fun = optax.value_and_grad_from_state(fun_p)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(grad, state, params, value=value, grad=grad, value_fn=fun_p)
        params = optax.apply_updates(params, updates)
        params = optax.projections.projection_box(params, bounds[0], bounds[1])
        return params, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = jax.lax.while_loop(continuing_criterion, step, init_carry)
    return final_params, final_state


def const_diag(n: int, val: float) -> Array:
    """
    Constructs diagonal matrix filled with scalar value.

    Args:
        n (int): Matrix size.
        val (float): Value.

    Returns:
        Array: Diagonal matrix filled with value [n, n].
    """

    return jnp.diag(jnp.full(n, val))


def value_and_jacfwd(f: Callable, *args, argnum: int = 0) -> Tuple[Any, Any]:
    """
    Evaluates function and its Jacobian.

    Args:
        f (Callable): Function.
        argnum (int, optional): Which argument to differentiate against. Defaults to 0.

    Returns:
        Tuple[Any, Any]: Function value, Jacobian.
    """

    val = f(*args)
    jac = jax.jacfwd(f, argnums=argnum)(*args)

    return val, jac


def jmp_aux(
    f: Callable, aux_structure: Tuple[int | None, ...], primals: List[Array], tangents: List[Array]
) -> Tuple[Array, Array, Tuple[Array, ...]]:
    jvp_fn = partial(jax.jvp, f, has_aux=True)
    y, jmp, aux = jax.vmap(jvp_fn, in_axes=(None, 1), out_axes=(None, 1, aux_structure))(
        primals, tangents
    )
    return y, jmp, aux


def mjp_aux(
    f: Callable, primals: List[Array], tangents: List[Array]
) -> Tuple[Array, Array, Tuple[Array, ...]]:
    y, vjp_fn, aux = jax.vjp(f, *primals, has_aux=True)
    (mjp,) = jax.vmap(vjp_fn)(*tangents)
    return y, mjp, aux


def store_data(data: Dict[str, Array], out_filepath: str, mode="w") -> None:
    """
    Saves data in a H5 file.

    Args:
        data (Dict[str, Array]): Data to be saved.
        out_filepath (str): Path to H5 results file.
    """

    Path(out_filepath).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_filepath, mode) as h5f:
        for k, v in data.items():
            if k in h5f.keys():
                del h5f[k]
            if k in {"prng_key"}:
                continue
            h5f.create_dataset(k, data=v)


def negative_log_gaussian_sqrt(x: Array, m: Array, P_sqrt: Array) -> Array:
    """
    Evaluates negative log Gaussian using pre-computed covariance square-root.

    Args:
        x (Array): Input [..., N].
        m (Array): Mean [..., N].
        P_sqrt (Array): Lower-triangular covariance square-root [..., N, N].

    Returns:
        Array: Evaluation result [...].
    """

    n = m.shape[-1]
    y = jsp.linalg.solve_triangular(P_sqrt, x - m, lower=True)  # type: ignore
    return (
        1 / 2 * jnp.einsum('...i,...i->...', y, y)
        + n / 2 * jnp.log(2 * jnp.pi)
        + jnp.log(jnp.abs(P_sqrt.diagonal(axis1=-1, axis2=-2))).sum(-1)  # type: ignore
    )


def normalize(
    values: Dict[str, Array] | Array,
    mins: Dict[str, Array] | Array,
    maxs: Dict[str, Array] | Array,
) -> Dict[str, Array] | Array:
    """
    Normalizes values to [0, 1] ranges, according to min-max values.

    Args:
        values (Dict[str, Array] | Array): Values to be normalized.
        mins (Dict[str, Array] | Array): Minimum values.
        maxs (Dict[str, Array] | Array): Maximum values.

    Returns:
        Dict[str, Array] | Array: Normalized values.
    """

    values_flat, unravel_fn = ravel_pytree(values)
    mins_flat, _ = ravel_pytree(mins)
    maxs_flat, _ = ravel_pytree(maxs)

    out_flat = (values_flat - mins_flat) / (maxs_flat - mins_flat)
    return unravel_fn(out_flat)


def inv_normalize(
    values: Dict[str, Array] | Array,
    mins: Dict[str, Array] | Array,
    maxs: Dict[str, Array] | Array,
) -> Dict[str, Array] | Array:
    """
    Inverts normalization of values to [0, 1] ranges, according to min-max values.

    Args:
        values (Dict[str, Array] | Array): Normalized values.
        mins (Dict[str, Array] | Array): Minimum values.
        maxs (Dict[str, Array] | Array): Maximum values.

    Returns:
        Dict[str, Array] | Array: Unnormalized values.
    """

    values_flat, unravel_fn = ravel_pytree(values)
    mins_flat, _ = ravel_pytree(mins)
    maxs_flat, _ = ravel_pytree(maxs)

    out_flat = values_flat * (maxs_flat - mins_flat) + mins_flat
    return unravel_fn(out_flat)


def sync_times(ts_x: Array, ts_y: Array) -> Tuple[Array, Array]:
    x_indices = jnp.nonzero(isin_tolerance(ts_x, ts_y, 1e-8))[0]
    y_indices = jnp.nonzero(isin_tolerance(ts_y, ts_x[x_indices], 1e-8))[0]
    assert len(x_indices) == len(y_indices), f"{len(x_indices)} != {len(y_indices)}"

    return x_indices, y_indices


def isin_tolerance(elements: Array, test_elements: Array, tol: float) -> Array:
    """
    Implementation of jnp.isin for floating point arrays with tolerance.
    Assumes arrays to be sorted.
    cf. https://stackoverflow.com/a/51747164

    Args:
        elements (Array): Elements to be checked [N].
        test_elements (Array): Elements to be checked against [M].
        tol (float): Tolerance.

    Returns:
        Array: Boolean mask of elements being found in test_elements [N].
    """

    idx = jnp.searchsorted(test_elements, elements)

    linvalid_mask = idx == len(test_elements)
    idx = jnp.where(linvalid_mask, len(test_elements) - 1, idx)
    lval = test_elements[idx] - elements
    lval = jnp.where(linvalid_mask, -lval, lval)

    rinvalid_mask = idx == 0
    idx1 = jnp.where(rinvalid_mask, 0, idx - 1)
    rval = elements - test_elements[idx1]
    rval = jnp.where(rinvalid_mask, -rval, rval)
    return jnp.minimum(lval, rval) <= tol


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


def sqrt_L_sum_qr(a: Array, b: Array) -> Array:
    """
    Computes lower-triangular (a+b)^(1/2), where a, b are lower-triangular matrix square-roots,
    using QR decomposition.

    Args:
        a (Array): Lower-triangular matrix square-root [N, N].
        b (Array): Lower-triangular matrix square-root [N, N].

    Returns:
        Array: Lower-triangular matrix square-root [N, N].
    """

    r = jsp.linalg.qr(
        jnp.concatenate([a.T, b.T], axis=-2),
        mode="economic",
    )[
        1
    ]  # [..., N, N]
    return r.T


def sqrt_L_sum_qr_3(a: Array, b: Array, c: Array) -> Array:
    """
    Computes lower-triangular (a+b)^(1/2), where a, b are lower-triangular matrix square-roots,
    using QR decomposition.

    Args:
        a (Array): Lower-triangular matrix square-root [N, N].
        b (Array): Lower-triangular matrix square-root [N, N].

    Returns:
        Array: Lower-triangular matrix square-root [N, N].
    """

    r = jsp.linalg.qr(
        jnp.concatenate([a.T, b.T, c.T], axis=-2),
        mode="economic",
    )[
        1
    ]  # [..., N, N]
    return r.T


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
