from absl.testing import absltest, parameterized
from ipoptax.solver import solve

import numpy as np

import jax


class Test(parameterized.TestCase):
    def setUp(self):
        super(Test, self).setUp()

    def testSimpleNLP(self):
        @jax.jit
        def f(x):
            return x[1] * (5.0 + x[0])

        @jax.jit
        def c(_):
            return jax.numpy.array([])

        @jax.jit
        def g(x):
            return jax.numpy.array(
                [5.0 - x[0] * x[1], x[0] * x[0] + x[1] * x[1] - 20.0]
            )

        ws_x = np.zeros(2)
        ws_s = np.ones(2)
        ws_y = np.zeros(0)
        ws_z = np.ones(2)

        outputs = solve(
            f=f,
            c=c,
            g=g,
            ws_x=ws_x,
            ws_s=ws_s,
            ws_y=ws_y,
            ws_z=ws_z,
            max_iterations=20,
            max_kkt_violation=1e-12,
        )
        converged = outputs["converged"]
        iterations = outputs["iteration"]
        jax.debug.print("converged={}, iterations={}", converged, iterations)

        approx_solution = np.array([-1.15747396, -4.31975162])

        jax.debug.print("The optimal value is {}.", f(approx_solution))

        self.assertTrue(converged.item())

        self.assertTrue(
            (np.linalg.norm(outputs["x"] - approx_solution < 1e-6)).item()
        )


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    np.set_printoptions(threshold=1000000)
    np.set_printoptions(linewidth=1000000)

    absltest.main()
