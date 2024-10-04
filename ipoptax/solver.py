# TODO(joao)

from jax import numpy as np

import jax

from functools import partial

import numpy as onp


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
    gamma=1e-6,
):
    """
    Solves an optimization problem of the form:
        min_x f(x) s.t. (c(x) = 0 and g(x) + s = 0 and s >= 0)

    gamma regularizes the constraints to ensure the Newton-KKT system is non-singular.
    tau is a parameter used in the fraction-to-the-boundary rule.
    """

    # TODO(joao):
    # 1. Select merit function parameter rho.
    # 2. Implement line search.

    assert ws_s.min() > 0.0, "ws_s must contain only positive entries."
    assert ws_z.min() > 0.0, "ws_z must contain only positive entries."

    def split_vars(xsyz):
        x_dim = ws_x.shape[0]
        s_dim = ws_s.shape[0]
        y_dim = ws_y.shape[0]
        z_dim = ws_z.shape[0]

        assert s_dim == z_dim, "Incompatible shapes of s and z."

        x = xsyz[:x_dim]
        s = xsyz[x_dim:x_dim + s_dim]
        y = xsyz[x_dim + s_dim:-z_dim]
        z = xsyz[-z_dim:]
        return x, s, y, z

    def combine_vars(x, s, y, z):
        return np.concatenate([x, s, y, z])

    def augmented_lagrangian(xsyz, mu):
        x, s, y, z = split_vars(xsyz)
        return f(x) - np.dot(y, c(x)) - np.dot(z, c(x) + s) - gamma * np.dot(y, y) - mu * np.log(s).sum()

    def adaptive_mu(x, s, y, z):
        # Uses the LOQO rule mentioned in Nocedal & Wright.
        m = s.shape[0]
        dot = np.dot(s, z)
        zeta = (s * z).min() * m / dot
        sigma = 0.1 * np.min(0.5 * (1.0 - zeta) / zeta, 2) ** 3
        return sigma * dot / m

    def optimization_loop(inputs):
        x = inputs["x"]
        s = inputs["s"]
        y = inputs["y"]
        z = inputs["z"]
        iteration = inputs["iteration"]

        mu = adaptive_mu(x, s, y, z)

        xsyz = combine_vars(x, s, y, z)
        al = partial(augmented_lagrangian, mu=mu)
        lhs = jax.hessian(al)(xsyz)
        rhs = -jax.grad(al)(xsyz)

        f = jax.scipy.linalg.cho_factor(lhs)
        dxsyz = jax.scipy.linalg.cho_solve(f, rhs)

        dx, ds, dy, dz = split_vars(dxsyz)

        # s + alpha_s_max * ds >= (1 - tau) * s
        alpha_s_max = np.min((-tau * s[ds < 0.0] / ds[ds < 0.0]).min(), 1.0)

        # z + alpha_z_max * dz >= (1 - tau) * z
        alpha_z_max = np.min((-tau * z[dz < 0.0] / dz[dz < 0.0]).min(), 1.0)

        # TODO(joao): add line search, etc. what's the easiest way of picking the merit rho?
        x = x + dx

        converged = rhs.abs().max() < max_kkt_violation
        should_continue = np.logical_and(np.logical_not(converged), iteration < max_iterations)

        return {
            "x": new_x,
            "s": new_s,
            "y": new_y,
            "z": new_z,
            "iteration": iteration + 1,
            "should_continue": should_continue,
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
    }
