from jax import numpy as np

import jax

from functools import partial

from .linalg_helpers import is_positive_definite


@jax.jit
def solve(
    *,
    f,
    c,
    g,
    ws_x,
    ws_s,
    ws_y,
    ws_z,
    max_iterations=100,
    max_kkt_violation=1e-6,
    tau=0.995,
    min_delta=1e-9,
    delta_update_factor=10,
    gamma=1e-6,
    armijo_factor=1e-4,
    line_search_factor=0.5,
):
    """
    Solves an optimization problem of the form:
        min_x f(x) s.t. (c(x) = 0 and g(x) + s = 0 and s >= 0)

    min_delta determines the minimum regularization on the objective Hessian.
    gamma regularizes the constraints to ensure the Newton-KKT system is non-singular.
    tau is a parameter used in the fraction-to-the-boundary rule.
    line_search_factor determines how much to backtrack at each line search iteration.
    """

    assert ws_s.min() > 0.0, "ws_s must contain only positive entries."
    assert ws_z.min() > 0.0, "ws_z must contain only positive entries."

    def split_vars(xsyz):
        x_dim = ws_x.shape[0]
        s_dim = ws_s.shape[0]
        z_dim = ws_z.shape[0]

        assert s_dim == z_dim, "Incompatible shapes of s and z."

        x = xsyz[:x_dim]
        s = xsyz[x_dim : x_dim + s_dim]
        y = xsyz[x_dim + s_dim : -z_dim]
        z = xsyz[-z_dim:]
        return x, s, y, z

    def combine_vars(x, s, y, z):
        return np.concatenate([x, s, y, z])

    def barrier_augmented_lagrangian(xsyz, mu):
        x, s, y, z = split_vars(xsyz)
        return (
            f(x)
            - np.dot(y, c(x))
            - np.dot(z, g(x) + s)
            - gamma * np.dot(y, y)
            - mu * np.log(s).sum()
        )

    def adaptive_mu(s, z):
        # Uses the LOQO rule mentioned in Nocedal & Wright.
        m = s.shape[0]
        dot = np.dot(s, z)
        zeta = (s * z).min() * m / dot
        sigma = 0.1 * np.minimum(0.5 * (1.0 - zeta) / zeta, 2) ** 3
        return sigma * dot / m

    def get_delta(M):
        dim = M.shape[0]
        def body(delta):
            return delta_update_factor * delta
        def should_continue(delta):
            return np.logical_not(is_positive_definite(M + delta * np.eye(dim)))
        return jax.lax.while_loop(body, should_continue, min_delta)

    def get_rho(x, s, dx):
        # D(merit_function; dx) = D(f; dx) - rho * || (c(x), g(x) + s) ||
        # rho > (D(f; dx) + k) / || (c(x), g(x) + s) || iff D(merit_function; dx) < -k.
        f_slope = jax.grad(f)(x).dot(dx)
        d = np.linalg.norm(np.concatenate([c(x), g(x) + s]))
        k = np.maximum(d, 2.0 * np.abs(f_slope))
        return np.minimum((f_slope + k) / d, 1e9)

    def merit_function(x, s, rho):
        return f(x) + rho * np.linalg.norm(np.concatenate([c(x), g(x) + s]))

    def merit_function_slope(x, s, dx, rho):
        return jax.grad(f)(x).dot(dx) - rho * np.linalg.norm(np.concatenate([c(x), g(x) + s]))

    def optimization_loop(inputs):
        x = inputs["x"]
        s = inputs["s"]
        y = inputs["y"]
        z = inputs["z"]
        iteration = inputs["iteration"]

        mu = adaptive_mu(s, z)

        xsyz = combine_vars(x, s, y, z)
        al = partial(barrier_augmented_lagrangian, mu=mu)
        lhs = jax.hessian(al)(xsyz)
        rhs = -jax.grad(al)(xsyz)

        x_dim = x.shape[0]
        remaining_dim = lhs.shape[0] - x_dim

        delta = get_delta(lhs[:x_dim, :x_dim])
        lhs = lhs + jax.scipy.linalg.block_diag(
            delta * np.eye(x_dim),
            np.zeros([remaining_dim, remaining_dim])
        )

        dxsyz = np.linalg.solve(lhs, rhs)

        dx, ds, dy, dz = split_vars(dxsyz)

        # s + alpha_s_max * ds >= (1 - tau) * s
        mod_ds = np.minimum(ds, np.full_like(ds, -1e-12))
        alpha_s_max = np.minimum((-tau * s / mod_ds).min(), 1.0)

        # z + alpha_z_max * dz >= (1 - tau) * z
        mod_dz = np.minimum(dz, np.full_like(dz, -1e-12))
        alpha_z_max = np.minimum((-tau * z / mod_dz).min(), 1.0)

        rho = get_rho(x=x, s=s, dx=dx)
        m = partial(merit_function, s=s, rho=rho)
        m_slope = merit_function_slope(x=x, s=s, dx=dx, rho=rho)

        def ls_body(alpha):
            return alpha * line_search_factor

        def ls_continue(alpha):
            return m(x + alpha * dx) - m(x) <= armijo_factor * m_slope * alpha

        alpha = jax.lax.while_loop(ls_continue, ls_body, 1.0)

        # print(f"{iteration=}, {alpha=}, {m(x)=}, {f(x)=}, {c(x)=}, {m_slope=}")

        new_x = x + alpha * dx
        new_s = s + np.minimum(alpha_s_max, alpha) * ds
        new_y = y + alpha * dy
        new_z = z + np.minimum(alpha_z_max, alpha) * dz

        converged = np.abs(rhs).max() < max_kkt_violation
        should_continue = np.logical_and(
            np.logical_not(converged), iteration < max_iterations
        )

        return {
            "x": new_x,
            "s": new_s,
            "y": new_y,
            "z": new_z,
            "iteration": iteration + 1,
            "should_continue": should_continue,
            "converged": converged,
        }

    def continuation_criteria(inputs):
        return inputs["should_continue"]

    inputs = {
        "x": ws_x,
        "s": ws_s,
        "y": ws_y,
        "z": ws_z,
        "iteration": 0,
        "should_continue": True,
        "converged": False,
    }

    outputs = jax.lax.while_loop(
        continuation_criteria,
        optimization_loop,
        inputs,
    )

    return {
        "x": outputs["x"],
        "s": outputs["s"],
        "y": outputs["y"],
        "z": outputs["z"],
        "iteration": outputs["iteration"],
        "converged": outputs["converged"],
    }
