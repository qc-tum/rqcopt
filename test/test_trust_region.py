import numpy as np
import unittest
import rqcopt as oc


class TestTrustRegion(unittest.TestCase):

    def test_quadratic_equation(self):
        """
        Test quadratic equation solving.
        """
        rng = np.random.default_rng()
        for p in [0., rng.standard_normal()]:
            q = rng.standard_normal()
            while p**2 - q < 0:
                q = rng.standard_normal()
            f = lambda x: x**2 + 2*p*x + q
            x1, x2 = oc.solve_quadratic_equation(p, q)
            self.assertAlmostEqual(f(x1), 0)
            self.assertAlmostEqual(f(x2), 0)


if __name__ == "__main__":
    unittest.main()
