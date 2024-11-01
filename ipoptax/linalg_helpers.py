from functools import partial

from jax import numpy as np
from jax import jit, lax, vmap


@jit
def _get_psd_eigenvalue_shift_ub(Q):
    """Returns a value k that guarantees that Q + kI is PSD.
    Relies on https://en.wikipedia.org/wiki/Gershgorin_circle_theorem.
    """
    D = np.diag(Q)

    # The radii of the circles.
    Rs = np.sum(np.abs(Q), axis=1) - np.abs(D)

    # The centers of the circles.
    Cs = D

    # The minimum shifts of each circle. Note: C - R + S > 0 and S >= 0 => S > R - C and S >= 0
    Ss = (Rs - Cs).clip(min=0.0)

    return np.max(Ss)


@jit
def _get_acceptable_psd_eigenvalue_shift(Q, k, delta):
    n, _ = Q.shape

    already_psd = is_positive_definite(Q, delta=delta)

    def continuation_criterion(k):
        return np.logical_and(
            np.logical_not(already_psd),
            is_positive_definite(Q + k * np.eye(n), delta=delta),
        )

    def body(k):
        return 0.5 * k

    k = 2.0 * lax.while_loop(
        continuation_criterion,
        body,
        k,
    )

    return lax.select(already_psd, 0.0, k)


@partial(jit, static_argnames=("use_lapack", "iterate"))
def project_psd_cone(Q, *, delta=0.0, use_lapack=True, iterate=True):
    """Projects to the cone of positive semi-definite matrices.

    Args:
      Q: [n, n] symmetric matrix.
      delta: minimum eigenvalue of the projection.

    Returns:
      [n, n] symmetric matrix projection of the input.
    """
    if use_lapack:
        S, V = np.linalg.eigh(Q)
        S = np.maximum(S, delta)
        Q_plus = np.matmul(V, np.matmul(np.diag(S), V.T))
        return 0.5 * (Q_plus + Q_plus.T)

    n = Q.shape[0]

    k = _get_psd_eigenvalue_shift_ub(Q)

    if iterate:
        k = _get_acceptable_psd_eigenvalue_shift(Q, k, delta)

    return Q + k * np.eye(n)


@jit
def ldlt(Q):
    """Computes the L D L^T decomposition of Q."""
    n, _ = Q.shape

    def outer_body(i, outer_carry):
        def inner_body(j, inner_carry):
            L, D_diag = inner_carry

            # Note: the terms with k >=i are 0.
            terms = vmap(lambda k: L[i, k] * L[j, k] * D_diag[k])(np.arange(n))

            L = L.at[i, j].set((1.0 / D_diag[j]) * (Q[i, j] - np.sum(terms)))

            return L, D_diag

        # Update L.
        L, D_diag = lax.fori_loop(
            lower=0, upper=i, body_fun=inner_body, init_val=outer_carry
        )

        # Update D_diag.
        # Note: the terms with k >=i are 0.
        terms = vmap(lambda k: L[i, k] * L[i, k] * D_diag[k])(np.arange(n))
        D_diag = D_diag.at[i].set(Q[i, i] - np.sum(terms))

        return L, D_diag

    return lax.fori_loop(
        lower=0,
        upper=n,
        body_fun=outer_body,
        init_val=(np.eye(n), np.zeros(n)),
    )


@jit
def is_positive_definite(Q, *, delta=0.0):
    """Checks whether the matrix Q is positive-definite.
    Does a L D L^T decomposition and checks that the diagonal entries of D are positive.
    See these for reference:
    1. https://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition
    2. https://services.math.duke.edu/~jdr/2021f-218/materials/week11.pdf

    Args:
      Q: [n, n] symmetric matrix.
      delta: minimum eigenvalue of the projection.

    Returns:
      [n, n] symmetric matrix projection of the input.
    """
    _, D_diag = ldlt(Q)
    return np.all(D_diag > delta)


@jit
def solve_lower_unitriangular(L, b):
    """Solves Lx=b for x, where L is lower uni-triangular."""

    n, _ = L.shape

    def f(carry, elem):
        partial_new_x = carry
        i = elem

        new_x_elem = b[i] - np.dot(L[i, :], partial_new_x)

        new_output = new_x_elem
        new_carry = vmap(
            lambda k: np.where(np.equal(k, i), new_x_elem, partial_new_x[k])
        )(np.arange(n))

        return new_carry, new_output

    return lax.scan(f, np.zeros_like(b), np.arange(n), n)[1]


@jit
def solve_upper_unitriangular(U, b):
    """Solves Ux=b for x, where U is upper uni-triangular."""

    n, _ = U.shape

    def f(carry, elem):
        partial_new_x = carry
        i = elem

        new_x_elem = b[i] - np.dot(U[i, :], partial_new_x)

        new_output = new_x_elem
        new_carry = vmap(
            lambda k: np.where(np.equal(k, i), new_x_elem, partial_new_x[k])
        )(np.arange(n))

        return new_carry, new_output

    return lax.scan(f, np.zeros_like(b), np.arange(n), n, reverse=True)[1]


@jit
def solve_ldlt(A, b):
    n, _ = A.shape
    L, D_diag = ldlt(A)
    z = solve_lower_unitriangular(L, b)
    y = np.diag(1.0 / D_diag) @ z
    return solve_upper_unitriangular(L.T, y)
