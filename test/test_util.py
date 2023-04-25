import numpy as np
import unittest
import rqcopt as oc
from scipy.stats import unitary_group


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

    def test_skew_encoding(self):
        """
        Test mapping between a real vector and skew-symmetric matrix.
        """
        rng = np.random.default_rng()
        n = 7
        # random real vector
        r = rng.standard_normal(n * (n - 1) // 2)
        w = oc.real_to_skew(r, n)
        self.assertTrue(np.allclose(w, -w.T))
        self.assertTrue(np.allclose(r, oc.skew_to_real(w)))
        self.assertAlmostEqual(np.linalg.norm(r), np.linalg.norm(w, "fro"))
        # another random real vector
        s = rng.standard_normal(n * (n - 1) // 2)
        z = oc.real_to_skew(s, n)
        # mapping preserves inner products
        self.assertAlmostEqual(np.dot(s, r), np.trace(z.T @ w))

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

    def test_blockenc_isometry(self):
        """
        Test block-encoding isometry.
        """
        rng = np.random.default_rng()
        # random unitary matrix 4x4 (2 qubits)
        L = 8
        # default matrix has alternating ancillary qubits
        anc = [i for i in range(1, L, 2)]
        p = oc.blockenc_isometry(L, anc)
        p_auto = oc.blockenc_isometry(L)
        self.assertTrue(np.allclose(p, p_auto))
        # random initial state, projected ancillaries on 0
        pp = p.dot(p.conj().T)
        psi = oc.crandn(2**L, rng)
        final_psi = pp.dot(psi)
        # if I project on |1> states over the ancillary I get 0 probability
        for i in anc:
            pi = np.identity(1)
            for j in range(i):
                pi = np.kron(pi, np.identity(2))
            pi = np.kron(pi, [[0,0],[0,1]])
            for  j in range(L-i-1):
                pi = np.kron(pi, np.identity(2))
            pi_state = pi @ final_psi
            self.assertTrue(np.allclose(np.einsum('i,i', pi_state.conj(), pi_state)**2, 0))
        



if __name__ == "__main__":
    unittest.main()
