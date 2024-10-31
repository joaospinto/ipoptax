from absl.testing import absltest, parameterized
from ipoptax.solver import solve

import numpy as np

import jax


class Test(parameterized.TestCase):
    def setUp(self):
        super(Test, self).setUp()

    def testSimpleQP(self):
        # Simple test from the OSQP repo.
        P = np.array([[4.0, 1.0], [1.0, 2.0]])
        q = np.ones(2)

        @jax.jit
        def f(x):
            return 0.5 * x.T @ P @ x + q.T @ x

        @jax.jit
        def c(x):
            return np.array([[1.0, 1.0]]) @ x - np.array([1.0])

        @jax.jit
        def g(x):
            return np.array(
                [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
            ) @ x - np.array([0.7, 0.0, 0.7, 0.0])

        n = P.shape[0]

        ws_x = np.zeros(n)
        ws_s = np.ones(4)
        ws_y = np.zeros(1)
        ws_z = np.ones(4)

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

        print(f"The optimal value is {f(np.array([0.3, 0.7]))}.")

        self.assertTrue(converged)

        self.assertTrue(
            np.linalg.norm(outputs["x"] - np.array([0.3, 0.7]) < 1e-6)
        )


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    np.set_printoptions(threshold=1000000)
    np.set_printoptions(linewidth=1000000)

    absltest.main()
