import numpy as np
import unittest
import rqcopt as oc


class TestUtilityFunctions(unittest.TestCase):

    def test_polar_decomposition(self):
        """
        Test polar decomposition.
        """
        rng = np.random.default_rng()
        a = 0.5 * oc.crandn((5, 5), rng)
        u, p = oc.polar_decomp(a)
        # `u` must be unitary
        self.assertTrue(np.allclose(u @ u.conj().T, np.identity(u.shape[0])))
        # `p` must be positive semidefinite
        self.assertTrue(np.allclose(p, p.conj().T))
        self.assertTrue(np.all(np.linalg.eigvalsh(p) >= 0))
        # require u @ p == a
        self.assertTrue(np.allclose(u @ p, a))

    def test_antisymmetric_encoding(self):
        """
        Test mapping between a real and antisymmetric matrix.
        """
        rng = np.random.default_rng()
        # random real matrix
        r = rng.standard_normal((7, 7))
        w = oc.real_to_antisymm(r)
        self.assertTrue(np.allclose(w, -w.conj().T))
        self.assertTrue(np.allclose(r, oc.antisymm_to_real(w)))
        self.assertAlmostEqual(np.linalg.norm(r, "fro"), np.linalg.norm(w, "fro"))
        # another random real matrix
        s = rng.standard_normal((7, 7))
        z = oc.real_to_antisymm(s)
        # mapping preserves inner products
        self.assertAlmostEqual(np.trace(s.T @ r), np.trace(z.conj().T @ w))


if __name__ == "__main__":
    unittest.main()
