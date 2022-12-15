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


def project_unitary_tangent(u, z):
    """
    Project `z` onto the tangent plane at the unitary matrix `u`.
    """
    # formula remains valid for `u` an isometry (element of the Stiefel manifold)
    return z - u @ symm(u.conj().T @ z)


def crandn(size, rng: np.random.Generator=None):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    if rng is None: rng = np.random.default_rng()
    # 1/sqrt(2) is a normalization factor
    return (rng.normal(size=size) + 1j*rng.normal(size=size)) / np.sqrt(2)
