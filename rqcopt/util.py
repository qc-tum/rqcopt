import numpy as np


def polar_decomp(a):
    """
    Perform a polar decomposition of a matrix: ``a = u p``,
    with `u` unitary and `p` positive semidefinite.
    """
    u, s, vh = np.linalg.svd(a)
    return u @ vh, (vh.conj().T * s) @ vh


def symm(w):
    """
    Symmetrize a matrix by projecting it onto the symmetric subspace.
    """
    return 0.5 * (w + w.conj().T)


def antisymm(w):
    """
    Antisymmetrize a matrix by projecting it onto the antisymmetric (skew-symmetric) subspace.
    """
    return 0.5 * (w - w.conj().T)


def real_to_antisymm(r):
    """
    Map a real-valued square matrix to an anti-symmetric matrix of the same dimension.
    """
    return 0.5*(r - r.T) + 0.5j*(r + r.T)


def antisymm_to_real(w):
    """
    Map an anti-symmetric matrix to a real-valued square matrix of the same dimension.
    """
    return w.real + w.imag


def real_to_skew(r, n: int):
    """
    Map a real vector to a skew-symmetric matrix containing the vector entries in its upper-triangular part.
    """
    if len(r) != n * (n - 1) // 2:
        raise ValueError("length of input vector does not match matrix dimension")
    w = np.zeros((n, n))
    # sqrt(2) factor to preserve inner products
    w[np.triu_indices(n, k=1)] = r / np.sqrt(2)
    w -= w.T
    return w


def skew_to_real(w):
    """
    Map a real skew-symmetric matrix to a real vector containing the upper-triangular entries.
    """
    # sqrt(2) factor to preserve inner products
    return np.sqrt(2) * w[np.triu_indices(len(w), k=1)]


def project_unitary_tangent(u, z):
    """
    Project `z` onto the tangent plane at the unitary matrix `u`.
    """
    # formula remains valid for `u` an isometry (element of the Stiefel manifold)
    return z - u @ symm(u.conj().T @ z)


def blockenc_isometry(L: int, idx_anc=None):
    """
    Construct a block-encoding isometry of the form
    (for unspecified ancillary indices):

      --------
      --|0>
      --------
      --|0>
        .
        .

    or according to the list of ancillary qubits in 'idx_anc'.
    """
    if idx_anc is None:
        b = np.array([[ 1., 0.],
                      [ 0., 0.],
                      [ 0., 1.],
                      [ 0., 0.]])
        p = np.identity(1)
        for i in range(L // 2):
            p = np.kron(p, b)
    else:
        id2 = np.identity(2)
        ket0 = np.array([[1.], [0.]])
        p = np.identity(1)
        for i in range(L):
             if i in idx_anc:
                 p = np.kron(p, ket0)
             else:
                 p = np.kron(p, id2)
    return p


def crandn(size=None, rng: np.random.Generator=None):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    if rng is None: rng = np.random.default_rng()
    # 1/sqrt(2) is a normalization factor
    return (rng.standard_normal(size) + 1j*rng.standard_normal(size)) / np.sqrt(2)
