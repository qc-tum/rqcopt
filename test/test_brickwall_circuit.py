import numpy as np
from scipy.stats import unitary_group, ortho_group
import unittest
import rqcopt as oc


def eval_numerical_gradient(f, x, h=1e-5):
    """
    Approximate the numeric gradient of a function via
    the difference quotient (f(x + h) - f(x - h)) / (2 h).
    """
    grad = np.zeros_like(x)

    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        i = it.multi_index
        xi_ref = x[i]
        x[i] = xi_ref + h
        fpos = f(x)         # evaluate f(x + h)
        x[i] = xi_ref - h
        fneg = f(x)         # evaluate f(x - h)
        x[i] = xi_ref       # restore
        # compute the partial derivative via centered difference quotient
        grad[i] = (fpos - fneg) / (2 * h)
        it.iternext() # step to next dimension

    return grad


def eval_numerical_gradient_complex(f, x, h=1e-5):
    """
    Approximate the numeric gradient (Wirtinger convention)
    of a function via difference quotients.
    """
    grad = np.zeros_like(x, dtype=complex)

    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        i = it.multi_index
        xi_ref = x[i]
        x[i] = xi_ref + h
        fpos = f(x)         # evaluate f(x + h)
        x[i] = xi_ref - h
        fneg = f(x)         # evaluate f(x - h)
        x[i] = xi_ref + 1j*h
        f_up = f(x)         # evaluate f(x + i*h)
        x[i] = xi_ref - 1j*h
        f_dn = f(x)         # evaluate f(x - i*h)
        x[i] = xi_ref       # restore
        # compute the partial derivative via centered difference quotient
        grad[i] = ((fpos - fneg) - 1j*(f_up - f_dn)) / (4 * h)
        it.iternext() # step to next dimension

    return grad


def shift_dims(psi, L, s):
    """
    Cyclically shift dimensions of `psi`, interpreted as 2 x ... x 2 array.
    """
    return np.reshape(np.transpose(np.reshape(psi, L*[2]), np.roll(range(L), s)), -1)


class TestBrickwallCircuit(unittest.TestCase):

    def test_parallel_gates(self):
        """
        Test parallel gate construction.
        """
        rng = np.random.default_rng()
        # system size
        L = 10
        # random unitary
        V = unitary_group.rvs(4, random_state=rng)
        Weven = oc.parallel_gates(V, L)
        Wodd  = oc.parallel_gates(V, L, np.roll(range(L), -1))
        # ensure that shift works correctly
        psi = oc.crandn(2**L, rng)
        self.assertAlmostEqual(
            np.linalg.norm(Weven @ psi -
            shift_dims(Wodd @ shift_dims(psi, L, 1), L, -1)), 0., delta=1e-12)

    def test_simple_gradient(self):
        """
        Test gradient computation for tr[U† V].
        """
        rng = np.random.default_rng()
        # random unitary
        V = unitary_group.rvs(4, random_state=rng)
        # complex random matrix
        U = 0.5 * oc.crandn((4, 4), rng)
        dV = oc.parallel_gates_grad(V, 2, U)
        # should recover 'U'
        self.assertTrue(np.allclose(dV, U))

    def test_parallel_gates_gradient(self):
        """
        Test gradient computation for parallel gates.
        """
        rng = np.random.default_rng()
        # system size
        L = 8
        # random unitary
        V = unitary_group.rvs(4, random_state=rng)
        # surrounding matrix
        U = 0.5 * oc.crandn(2 * (2**L,), rng)
        perm = rng.permutation(L)
        dV = oc.parallel_gates_grad(V, L, U, perm)
        # numerical gradient via finite difference approximation
        f = lambda v: np.trace(U.conj().T @ oc.parallel_gates(v, L, perm)).real
        dV_num = 2 * eval_numerical_gradient_complex(f, V, h=1e-6).conj()
        self.assertTrue(np.allclose(dV_num, dV))

    def test_parallel_gates_directed_gradient(self):
        """
        Test directed gradient computation for parallel gates.
        """
        rng = np.random.default_rng()
        # system size
        L = 8
        # random unitary
        V = unitary_group.rvs(4, random_state=rng)
        # direction
        Z = 0.5 * oc.crandn((4, 4), rng)
        perm = rng.permutation(L)
        dW = oc.parallel_gates_directed_grad(V, L, Z, perm)
        # numerical gradient via finite difference approximation
        f = lambda t: oc.parallel_gates(V + t*Z, L, perm)
        h = 1e-6
        dW_num = (f(h) - f(-h)) / (2*h)
        self.assertTrue(np.allclose(dW_num, dW))

    def test_parallel_gates_hessian(self):
        """
        Test Hessian computation for parallel gates.
        """
        rng = np.random.default_rng()
        # system size
        L = 8
        # random unitary
        V = unitary_group.rvs(4, random_state=rng)
        # surrounding matrix
        U = 0.5 * oc.crandn(2 * (2**L,), rng)
        perm = rng.permutation(L)
        # direction
        rZ = 0.5 * oc.crandn((4, 4), rng)
        for Z in [rZ, oc.project_unitary_tangent(V, rZ)]:
            for uproj in [False, True]:
                dV = oc.parallel_gates_hess(V, L, Z, U, perm, unitary_proj=uproj)
                # numerical Hessian via finite difference approximation
                gf = lambda t: oc.parallel_gates_grad(V + t*Z, L, U, perm)
                if uproj:
                    f = lambda t: oc.project_unitary_tangent(V + t*Z, gf(t))
                else:
                    f = gf
                h = 1e-6
                dV_num = (f(h) - f(-h)) / (2*h)
                self.assertTrue(np.allclose(dV_num, dV))

    def test_brickwall_unitary_gradient(self):
        """
        Test gradient computation for a brickwall of parallel gates.
        """
        rng = np.random.default_rng()
        # system size
        L = 8
        # number of layers
        n = 5
        # random unitaries
        Vlist = np.stack([unitary_group.rvs(4, random_state=rng) for _ in range(n)])
        # random permutations
        perms = [rng.permutation(L) for _ in range(n)]
        # surrounding matrix
        U = 0.5 * oc.crandn(2 * (2**L,), rng)
        dVlist = oc.brickwall_unitary_grad(Vlist, L, U, perms)
        # numerical gradient via finite difference approximation
        f = lambda ulist: np.trace(U.conj().T @ oc.brickwall_unitary(ulist, L, perms)).real
        dVlist_num = 2 * eval_numerical_gradient_complex(f, Vlist.copy(), h=1e-6).conj()
        self.assertTrue(np.allclose(dVlist_num, dVlist))
        # gradient of Frobenius distance to U
        dVlist = 2 * oc.brickwall_unitary_grad(Vlist, L, oc.brickwall_unitary(Vlist, L, perms) - U, perms)
        f = lambda ulist: np.linalg.norm(oc.brickwall_unitary(ulist, L, perms) - U, "fro")**2
        dVlist_num = 2 * eval_numerical_gradient_complex(f, Vlist.copy(), h=1e-6).conj()
        self.assertTrue(np.allclose(dVlist_num, dVlist))

    def test_brickwall_unitary_directed_gradient(self):
        """
        Test directed gradient computation for a brickwall of parallel gates.
        """
        rng = np.random.default_rng()
        # system size
        L = 8
        # number of layers
        n = 5
        # random unitaries
        Vlist = [unitary_group.rvs(4, random_state=rng) for _ in range(n)]
        # random permutations
        perms = [rng.permutation(L) for _ in range(n)]
        for k in range(n):
            # direction
            Z = 0.5 * oc.crandn((4, 4), rng)
            dW = oc.brickwall_unitary_directed_grad(Vlist, L, Z, k, perms)
            # numerical gradient via finite difference approximation
            f = lambda t: oc.brickwall_unitary(Vlist[:k] + [Vlist[k] + t*Z] + Vlist[k+1:], L, perms)
            h = 1e-6
            dW_num = (f(h) - f(-h)) / (2*h)
            self.assertTrue(np.allclose(dW_num, dW))

    def test_brickwall_unitary_hessian(self):
        """
        Test Hessian computation for a brickwall of parallel gates.
        """
        rng = np.random.default_rng()
        # system size
        L = 8
        # number of layers
        n = 5
        # random unitaries
        Vlist = [unitary_group.rvs(4, random_state=rng) for _ in range(n)]
        # random permutations
        perms = [rng.permutation(L) for _ in range(n)]
        # surrounding matrix
        U = 0.5 * oc.crandn(2 * (2**L,), rng)
        for k in range(n):
            # direction
            rZ = 0.5 * oc.crandn((4, 4), rng)
            for Z in [rZ, oc.project_unitary_tangent(Vlist[k], rZ)]:
                for uproj in [False, True]:
                    dVlist = oc.brickwall_unitary_hess(Vlist, L, Z, k, U, perms, unitary_proj=uproj)
                    # numerical Hessian via finite difference approximation
                    gf = lambda t: oc.brickwall_unitary_grad(Vlist[:k] + [Vlist[k] + t*Z] + Vlist[k+1:], L, U, perms)
                    if uproj:
                        f = lambda t: np.stack([oc.project_unitary_tangent(Vlist[j] + t*Z if j == k else Vlist[j], grad)
                                                for j, grad in enumerate(gf(t))])
                    else:
                        f = gf
                    h = 1e-6
                    dVlist_num = (f(h) - f(-h)) / (2*h)
                    # compare
                    self.assertTrue(np.allclose(dVlist_num, dVlist))

    def test_brickwall_unitary_hessian_matrix(self):
        """
        Test Hessian matrix computation for a brickwall of parallel gates.
        """
        rng = np.random.default_rng()
        # system size
        L = 8
        # number of layers
        n = 5
        # random unitaries
        Vlist = [unitary_group.rvs(4, random_state=rng) for _ in range(n)]
        # random permutations
        perms = [rng.permutation(L) for _ in range(n)]
        # surrounding matrix
        U = 0.25 * oc.crandn(2 * (2**L,), rng)
        H = oc.brickwall_unitary_hessian_matrix(Vlist, L, U, perms)
        # must be symmetric
        self.assertTrue(np.allclose(H, H.T))

    def test_brickwall_ortho_hessian_matrix(self):
        """
        Test Hessian matrix computation for a brickwall of parallel gates,
        for the case of real-valued matrices.
        """
        rng = np.random.default_rng()
        # system size
        L = 8
        # number of layers
        n = 5
        # random unitaries
        Vlist = [ortho_group.rvs(4, random_state=rng) for _ in range(n)]
        # random permutations
        perms = [rng.permutation(L) for _ in range(n)]
        # surrounding matrix
        U = 0.25 * rng.standard_normal(2 * (2**L,))
        H = oc.brickwall_ortho_hessian_matrix(Vlist, L, U, perms)
        # must be symmetric
        self.assertTrue(np.allclose(H, H.T))

    def test_squared_brickwall_unitary_gradient(self):
        """
        Test gradient computation for a squared brickwall circuit of parallel gates.
        """
        rng = np.random.default_rng()
        # system size
        L = 8
        # number of layers
        n = 5
        # random unitaries
        Vlist = np.stack([unitary_group.rvs(4, random_state=rng) for _ in range(n)])
        # random permutations
        perms = [rng.permutation(L) for _ in range(n)]
        # enclosing Hermitian matrices
        A = 0.1 * oc.crandn(2 * (2**L,), rng)
        B = 0.1 * oc.crandn(2 * (2**L,), rng)
        A = 0.5 * (A + A.conj().T)
        B = 0.5 * (B + B.conj().T)
        dVlist = oc.squared_brickwall_unitary_grad(Vlist, L, A, B, perms)
        # numerical gradient via finite difference approximation
        def f(vlist):
            W = oc.brickwall_unitary(vlist, L, perms)
            return np.trace(A @ W.conj().T @ B @ W).real
        dVlist_num = 2 * eval_numerical_gradient_complex(f, Vlist.copy(), h=1e-6).conj()
        self.assertTrue(np.allclose(dVlist_num, dVlist))

    def test_squared_brickwall_unitary_hessian(self):
        """
        Test Hessian computation for a squared brickwall circuit of parallel gates.
        """
        rng = np.random.default_rng()
        # system size
        L = 8
        # number of layers
        n = 5
        # random unitaries
        Vlist = [unitary_group.rvs(4, random_state=rng) for _ in range(n)]
        # random permutations
        perms = [rng.permutation(L) for _ in range(n)]
        # enclosing Hermitian matrices
        A = 0.1 * oc.crandn(2 * (2**L,), rng)
        B = 0.1 * oc.crandn(2 * (2**L,), rng)
        A = 0.5 * (A + A.conj().T)
        B = 0.5 * (B + B.conj().T)
        for k in range(n):
            # direction
            rZ = 0.5 * oc.crandn((4, 4), rng)
            for Z in [rZ, oc.project_unitary_tangent(Vlist[k], rZ)]:
                for uproj in [False, True]:
                    dVlist = oc.squared_brickwall_unitary_hess(Vlist, L, Z, k, A, B, perms, unitary_proj=uproj)
                    # numerical Hessian via finite difference approximation
                    gf = lambda t: oc.squared_brickwall_unitary_grad(Vlist[:k] + [Vlist[k] + t*Z] + Vlist[k+1:], L, A, B, perms)
                    if uproj:
                        f = lambda t: np.stack([oc.project_unitary_tangent(Vlist[j] + t*Z if j == k else Vlist[j], grad)
                                                for j, grad in enumerate(gf(t))])
                    else:
                        f = gf
                    h = 1e-6
                    dVlist_num = (f(h) - f(-h)) / (2*h)
                    # compare
                    self.assertTrue(np.allclose(dVlist_num, dVlist))

    def test_squared_brickwall_unitary_hessian_matrix(self):
        """
        Test Hessian matrix computation for a squared brickwall circuit of parallel gates.
        """
        rng = np.random.default_rng()
        # system size
        L = 8
        # number of layers
        n = 5
        # random unitaries
        Vlist = [unitary_group.rvs(4, random_state=rng) for _ in range(n)]
        # random permutations
        perms = [rng.permutation(L) for _ in range(n)]
        # enclosing Hermitian matrices
        A = 0.1 * oc.crandn(2 * (2**L,), rng)
        B = 0.1 * oc.crandn(2 * (2**L,), rng)
        A = 0.5 * (A + A.conj().T)
        B = 0.5 * (B + B.conj().T)
        H = oc.squared_brickwall_unitary_hessian_matrix(Vlist, L, A, B, perms)
        # must be symmetric
        self.assertTrue(np.allclose(H, H.T))

    def test_squared_brickwall_ortho_hessian_matrix(self):
        """
        Test Hessian matrix computation for a squared brickwall circuit of parallel gates.
        """
        rng = np.random.default_rng()
        # system size
        L = 8
        # number of layers
        n = 5
        # random unitaries
        Vlist = [ortho_group.rvs(4, random_state=rng) for _ in range(n)]
        # random permutations
        perms = [rng.permutation(L) for _ in range(n)]
        # enclosing Hermitian matrices
        A = 0.1 * rng.standard_normal(2 * (2**L,))
        B = 0.1 * rng.standard_normal(2 * (2**L,))
        A = 0.5 * (A + A.T)
        B = 0.5 * (B + B.T)
        H = oc.squared_brickwall_ortho_hessian_matrix(Vlist, L, A, B, perms)
        # must be symmetric
        self.assertTrue(np.allclose(H, H.T))


if __name__ == "__main__":
    unittest.main()
