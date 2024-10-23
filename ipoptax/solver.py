from jax import numpy as np

import jax

from functools import partial

from .linalg_helpers import project_psd_cone


@partial(jax.jit, static_argnames=("f", "c", "g", "print_logs"))
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
    tau_min=0.995,
    mu_min=1e-12,
    min_delta=1e-9,
    gamma=1e-6,
    armijo_factor=1e-4,
    line_search_factor=0.5,
    line_search_min_step_size=1e-6,
    print_logs=True,
):
    """
    Solves an optimization problem of the form:
        min_x f(x) s.t. (c(x) = 0 and g(x) + s = 0 and s >= 0)

    min_delta determines the minimum regularization on the objective Hessian.
    gamma regularizes the constraints to ensure the Newton-KKT system is non-singular.
    tau_min is a parameter used in the fraction-to-the-boundary rule.
    line_search_factor determines how much to backtrack at each line search iteration.
    """

    # assert ws_s.min() > 0.0, "ws_s must contain only positive entries."
    # assert ws_z.min() > 0.0, "ws_z must contain only positive entries."

    def split_vars(xsyz):
        x_dim = ws_x.shape[0]
        s_dim = ws_s.shape[0]
        z_dim = ws_z.shape[0]

        # assert s_dim == z_dim, "Incompatible shapes of s and z."

        x = xsyz[:x_dim]
        s = xsyz[x_dim : x_dim + s_dim]
        y = xsyz[x_dim + s_dim : -z_dim]
        z = xsyz[-z_dim:]
        return x, s, y, z

    def combine_vars(*, x, s, y, z):
        return np.concatenate([x, s, y, z])

    def barrier_augmented_lagrangian(xsyz, *, mu):
        x, s, y, z = split_vars(xsyz)
        return (
            f(x)
            + np.dot(y, c(x))
            + np.dot(z, g(x) + s)
            - mu * np.log(s).sum()
        )

    def adaptive_mu(*, s, z, iteration):
        # Uses the LOQO rule mentioned in Nocedal & Wright.
        m = s.shape[0]
        dot = np.dot(s, z)
        zeta = (s * z).min() * m / dot
        sigma = 0.1 * np.minimum(0.5 * (1.0 - zeta) / zeta, 2) ** 3
        return sigma * dot / m

    def get_rho(*, x, s, dx):
        # D(merit_function; dx) = D(f; dx) - rho * || (c(x), g(x) + s) ||
        # rho > (D(f; dx) + k) / || (c(x), g(x) + s) || iff D(merit_function; dx) < -k.
        f_slope = jax.grad(f)(x).dot(dx)
        d = np.linalg.norm(np.concatenate([c(x), g(x) + s]))
        k = np.maximum(d, 2.0 * np.abs(f_slope))
        return np.minimum((f_slope + k) / d, 1e9)

    def merit_function(*, x, s, rho):
        return f(x) + rho * np.linalg.norm(
            np.concatenate([c(x), g(x) + s])
        )

    def merit_function_slope(*, x, s, dx, rho):
        return jax.grad(f)(x).dot(dx) - rho * np.linalg.norm(
            np.concatenate([c(x), g(x) + s])
        )

    def compute_search_direction_autodiff(*, x, s, y, z, mu):
        x_dim = x.shape[0]
        s_dim = s.shape[0]
        y_dim = y.shape[0]
        z_dim = z.shape[0]
        xsyz = combine_vars(x=x, s=s, y=y, z=z)
        al = partial(barrier_augmented_lagrangian, mu=mu)
        lhs = jax.hessian(al)(xsyz)
        rhs = -jax.grad(al)(xsyz)
        lhs += jax.scipy.linalg.block_diag(
            np.zeros([x_dim, x_dim]),
            np.zeros([s_dim, s_dim]),
            -gamma * np.eye(y_dim),
            -gamma * np.eye(z_dim),
        )

        lhs = lhs.at[:x_dim, :x_dim].set(project_psd_cone(lhs[:x_dim, :x_dim], delta=min_delta))
        dxsyz = np.linalg.solve(lhs, rhs)
        error = np.linalg.norm(lhs @ dxsyz - rhs)
        return dxsyz, rhs, error

    def compute_search_direction_stable_direct_method(*, x, s, y, z, mu):
        x_dim = x.shape[0]
        s_dim = s.shape[0]
        y_dim = y.shape[0]
        z_dim = z.shape[0]

        def al_x(xx):
            return barrier_augmented_lagrangian(
                np.concatenate([xx, s, y, z]), mu=mu
            )

        D1L = jax.grad(al_x)(x)
        D2L = project_psd_cone(jax.hessian(al_x)(x), delta=min_delta)
        C = jax.jacfwd(c)(x)
        G = jax.jacfwd(g)(x)
        lhs = np.block(
            [
                [D2L, np.zeros([x_dim, s_dim]), C.T, G.T],
                [
                    np.zeros([s_dim, x_dim]),
                    np.diag(z),
                    np.zeros([s_dim, y_dim]),
                    np.diag(s),
                ],
                [
                    C,
                    np.zeros([y_dim, s_dim]),
                    -gamma * np.eye(y_dim),
                    np.zeros([y_dim, z_dim]),
                ],
                [
                    G,
                    np.eye(z_dim),
                    np.zeros([z_dim, y_dim]),
                    -gamma * np.eye(z_dim),
                ],
            ]
        )

        rhs = -np.concatenate(
            [D1L, s * z - mu * np.ones_like(s), c(x), g(x) + s]
        )

        dxsyz = np.linalg.solve(lhs, rhs)
        error = np.linalg.norm(lhs @ dxsyz - rhs)
        return dxsyz, rhs, error


    def optimization_loop(inputs):
        x = inputs["x"]
        s = inputs["s"]
        y = inputs["y"]
        z = inputs["z"]
        iteration = inputs["iteration"]

        mu = np.maximum(adaptive_mu(s=s, z=z, iteration=iteration), mu_min)

        x_dim = x.shape[0]
        s_dim = s.shape[0]

        dxsyz, rhs, lin_sys_error = compute_search_direction_stable_direct_method(
            x=x, s=s, y=y, z=z, mu=mu
        )

        dx, ds, dy, dz = split_vars(dxsyz)

        tau = np.maximum(tau_min, np.where(mu > 0.0, 1.0 - mu, 0.0))

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
            # TODO(joao): the line search has to be done in terms of both x and s, at least.
            #             until the line search code is fixed, it's turned off.
            return False
            return np.logical_and(
                m(x + alpha * dx) - m(x) >= armijo_factor * m_slope * alpha,
                alpha > line_search_min_step_size,
            )

        # TODO(joao): this seems non-standard; investigate why it's needed.
        alpha = jax.lax.while_loop(ls_continue, ls_body, 1.0)
        # alpha = jax.lax.while_loop(ls_continue, ls_body, alpha_s_max)

        if print_logs:
            jax.debug.print(
                "{:^+10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}",
                iteration,
                alpha,
                m(x=x),
                f(x),
                np.linalg.norm(c(x)),
                np.linalg.norm(g(x) + s),
                m_slope,
                alpha_s_max,
                alpha_z_max,
                np.linalg.norm(dx),
                np.linalg.norm(ds),
                np.linalg.norm(dy),
                np.linalg.norm(dz),
                mu,
                lin_sys_error,
            )

        dual_alpha = alpha_z_max

        new_x = x + alpha * dx
        new_s = s + np.minimum(alpha, alpha_s_max) * ds
        new_y = y + dual_alpha * dy
        new_z = z + dual_alpha * dz

        # Note that the derivatives with respect to s should be excluded,
        # because of the log-barrier terms.
        converged = (
            np.maximum(
                np.abs(rhs[:x_dim]).max(), np.abs(rhs[x_dim + s_dim :]).max()
            )
            < max_kkt_violation
        )
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

    if print_logs:
        jax.debug.print(
            "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}",
            "iteration",
            "alpha",
            "merit",
            "f",
            "|c|",
            "|g+s|",
            "m_slope",
            "alpha_s_m",
            "alpha_z_m",
            "|dx|",
            "|ds|",
            "|dy|",
            "|dz|",
            "mu",
            "linsys_res",
        )

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
