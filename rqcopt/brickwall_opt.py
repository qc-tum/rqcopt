import numpy as np
from .brickwall_circuit import (
    brickwall_unitary,
    brickwall_unitary_gradient_vector,
    brickwall_unitary_hessian_matrix,
    brickwall_ortho_gradient_vector,
    brickwall_ortho_hessian_matrix,
    squared_brickwall_unitary_gradient_vector,
    squared_brickwall_unitary_hessian_matrix,
    squared_brickwall_ortho_gradient_vector,
    squared_brickwall_ortho_hessian_matrix)
from .trust_region import riemannian_trust_region_optimize
from .util import polar_decomp, real_to_antisymm, real_to_skew


def brickwall_quadratic_model(Vlist, L: int, U, perms, hlist, rng: np.random.Generator=None):
    """
    Compute target function along a random direction in tangent space,
    and the corresponding quadratic approximation.
    """
    n = len(Vlist)
    # target function
    f = lambda vlist: -np.trace(U.conj().T @ brickwall_unitary(vlist, L, perms)).real
    f0 = f(Vlist)
    # gradient
    grad = -brickwall_unitary_gradient_vector(Vlist, L, U, perms)
    # Hessian matrix; eigenvalues can be of either sign
    H = -brickwall_unitary_hessian_matrix(Vlist, L, U, perms)
    # random direction
    if rng is None: rng = np.random.default_rng()
    eta = rng.standard_normal(n * 16)
    eta /= np.linalg.norm(eta)
    # model function (Taylor approximation)
    q = lambda h: f0 + h*np.dot(grad, eta) + 0.5 * h**2 * np.dot(eta, H @ eta)
    # target function in direction 'eta'
    eta_mat = np.reshape(eta, (n, 4, 4))
    eta_mat = [Vlist[j] @ real_to_antisymm(eta_mat[j]) for j in range(n)]
    feta = lambda h: f([polar_decomp(Vlist[j] + h*eta_mat[j])[0] for j in range(n)])
    # return eta and the target function and quadratic model evaluated at 'hlist'
    return eta, np.array([feta(h) for h in hlist]), np.array([q(h) for h in hlist])


def optimize_brickwall_circuit(L: int, U, Vlist_start, perms, **kwargs):
    """
    Optimize the quantum gates in a brickwall layout to approximate
    the unitary matrix `U` using a trust-region method.
    """
    # target function
    f = lambda vlist: -np.trace(U.conj().T @ brickwall_unitary(vlist, L, perms)).real
    gradfunc = lambda vlist: -brickwall_unitary_gradient_vector(vlist, L, U, perms)
    hessfunc = lambda vlist: -brickwall_unitary_hessian_matrix(vlist, L, U, perms)
    # quantify error by spectral norm
    errfunc = lambda vlist: np.linalg.norm(brickwall_unitary(vlist, L, perms) - U, ord=2)
    kwargs["gfunc"] = errfunc
    # perform optimization
    Vlist, f_iter, err_iter = riemannian_trust_region_optimize(
        f, retract_unitary_list, gradfunc, hessfunc, np.stack(Vlist_start), **kwargs)
    return Vlist, f_iter, err_iter


def optimize_brickwall_circuit_blockenc(L: int, H, P, Vlist_start, perms, real: bool, **kwargs):
    """
    Optimize the quantum gates in a brickwall layout to approximate
    the matrix `H` based on block-encoding with projector P, using a trust-region method.
    """
    PP  = P @ P.conj().T
    PHP = P @ H @ P.conj().T
    # target function
    f = lambda vlist: 0.5 * np.linalg.norm(P.conj().T @ brickwall_unitary(vlist, L, perms) @ P - H, "fro")**2
    if real:
        gradfunc = lambda vlist: 0.5 * squared_brickwall_ortho_gradient_vector(vlist, L, PP, PP, perms) - brickwall_ortho_gradient_vector(vlist, L, PHP, perms)
        hessfunc = lambda vlist: 0.5 * squared_brickwall_ortho_hessian_matrix (vlist, L, PP, PP, perms) - brickwall_ortho_hessian_matrix (vlist, L, PHP, perms)
    else:
        gradfunc = lambda vlist: 0.5 * squared_brickwall_unitary_gradient_vector(vlist, L, PP, PP, perms) - brickwall_unitary_gradient_vector(vlist, L, PHP, perms)
        hessfunc = lambda vlist: 0.5 * squared_brickwall_unitary_hessian_matrix (vlist, L, PP, PP, perms) - brickwall_unitary_hessian_matrix(vlist, L, PHP, perms)
    # quantify error by spectral norm
    errfunc = lambda vlist: np.linalg.norm(P.conj().T @ brickwall_unitary(vlist, L, perms) @ P - H, ord=2)
    kwargs["gfunc"] = errfunc
    if real:
        retractfunc = lambda vlist, eta: retract_ortho_list(vlist, eta)
    else:
        retractfunc = lambda vlist, eta: retract_unitary_list(vlist, eta)
    # perform optimization
    Vlist, f_iter, err_iter = riemannian_trust_region_optimize(
        f, retractfunc, gradfunc, hessfunc, np.stack(Vlist_start), **kwargs)
    return Vlist, f_iter, err_iter


def retract_unitary_list(vlist, eta):
    """
    Retraction for unitary matrices, with tangent direction represented as anti-symmetric matrices.
    """
    n = len(vlist)
    eta = np.reshape(eta, (n, 4, 4))
    dvlist = [vlist[j] @ real_to_antisymm(eta[j]) for j in range(n)]
    return np.stack([polar_decomp(vlist[j] + dvlist[j])[0] for j in range(n)])


def retract_ortho_list(vlist, eta):
    """
    Retraction for orthogonal matrices, with tangent direction represented as skew-symmetric matrices.
    """
    n = len(vlist)
    eta = np.reshape(eta, (n, 6))
    dvlist = [vlist[j] @ real_to_skew(eta[j], 4) for j in range(n)]
    return np.stack([polar_decomp(vlist[j] + dvlist[j])[0] for j in range(n)])
