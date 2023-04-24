import numpy as np
from .util import project_unitary_tangent, antisymm, real_to_antisymm, antisymm_to_real, real_to_skew, skew_to_real, blockenc_isometry


def parallel_gates(V, L, perm=None):
    """
    Two-qubit gate V applied to lattice site pairs: V ⊗ ... ⊗ V,
    optionally with subsequent permutation of quantum wires.
    """
    assert L % 2 == 0
    W = np.identity(1)
    for i in range(L // 2):
        W = np.kron(W, V)
    if perm is not None:
        W = permute_operation(W, perm)
    return W


def parallel_gates_grad(V, L, U, perm=None):
    """
    Compute the gradient of Re tr[U† (V ⊗ ... ⊗ V)] with respect to V.
    """
    assert V.shape == (4, 4)
    assert U.shape == (2**L, 2**L)
    assert L % 2 == 0
    if perm is not None:
        inv_perm = np.argsort(perm)
        U = permute_operation(U, inv_perm)
    G = np.zeros_like(V)
    for i in range(0, L, 2):
        Ua = parallel_gates(V, i)
        Ub = parallel_gates(V, L-i-2)
        T = np.reshape(U, 2 * (2**i, 4, 2**(L-i-2)))
        T = np.tensordot(Ua.conj(), T, axes=((0, 1), (0, 3)))
        T = np.tensordot(Ub.conj(), T, axes=((0, 1), (1, 3)))
        assert T.ndim == 2
        G += T
    return G


def parallel_gates_directed_grad(V, L, Z, perm=None):
    """
    Compute the gradient of V ⊗ ... ⊗ V in direction `Z`.
    """
    assert L % 2 == 0
    G = 0
    for i in range(L // 2):
        W = np.identity(1)
        for j in range(L // 2):
            if i == j:
                W = np.kron(W, Z)
            else:
                W = np.kron(W, V)
        G += W
    if perm is not None:
        G = permute_operation(G, perm)
    return G


def parallel_gates_hess(V, L, Z, U, perm=None, unitary_proj=False):
    """
    Compute the Hessian of V -> Re tr[U† (V ⊗ ... ⊗ V)] in direction Z.
    """
    assert V.shape == (4, 4)
    assert Z.shape == (4, 4)
    assert U.shape == (2**L, 2**L)
    assert L % 2 == 0 and L > 0
    if perm is not None:
        inv_perm = np.argsort(perm)
        U = permute_operation(U, inv_perm)
    G = np.zeros_like(V)
    for i in range(0, L, 2):
        for j in range(0, i, 2):
            # j < i
            Va = parallel_gates(V, j)
            Vb = parallel_gates(V, i-j-2)
            Vc = parallel_gates(V, L-i-2)
            T = np.reshape(U, 2 * (2**j, 4, 2**(i-j-2), 4, 2**(L-i-2)))
            G += np.einsum(T, range(10), Va.conj(), (0, 5), Z.conj(), (1, 6), Vb.conj(), (2, 7), Vc.conj(), (4, 9), (3, 8))
        for j in range(i + 2, L, 2):
            # i < j
            Va = parallel_gates(V, i)
            Vb = parallel_gates(V, j-i-2)
            Vc = parallel_gates(V, L-j-2)
            T = np.reshape(U, 2 * (2**i, 4, 2**(j-i-2), 4, 2**(L-j-2)))
            G += np.einsum(T, range(10), Va.conj(), (0, 5), Vb.conj(), (2, 7), Z.conj(), (3, 8), Vc.conj(), (4, 9), (1, 6))
    if unitary_proj:
        G = project_unitary_tangent(V, G)
        # additional terms resulting from the projection of the gradient
        # onto the Stiefel manifold (unitary matrices)
        grad = parallel_gates_grad(V, L, U, perm=None)  # U is already permuted
        G -= 0.5 * (Z @ grad.conj().T @ V + V @ grad.conj().T @ Z)
        if not np.allclose(Z, project_unitary_tangent(V, Z)):
            G -= 0.5 * (Z @ V.conj().T + V @ Z.conj().T) @ grad
    return G


def brickwall_unitary(Vlist, L, perms):
    """
    Construct the unitary matrix representation of a brickwall-type
    quantum circuit with periodic boundary conditions.
    """
    W = np.identity(2**L)
    for V, perm in zip(Vlist, perms):
        W = parallel_gates(V, L, perm) @ W
    return W


def brickwall_unitary_grad(Vlist, L, U, perms):
    """
    Compute the gradient of Re tr[U† W] with respect to Vlist,
    where W is the brickwall circuit constructed from the gates in Vlist.
    """
    return np.stack([
        parallel_gates_grad(Vlist[j], L,
                            brickwall_unitary(Vlist[j+1:], L, perms[j+1:]).conj().T @
                            U @
                            brickwall_unitary(Vlist[:j], L, perms[:j]).conj().T,
                            perms[j])
            for j in range(len(Vlist))])


def brickwall_unitary_gradient_vector(Vlist, L, U, perms):
    """
    Represent the gradient of Re tr[U† W] as real vector,
    where W is the brickwall circuit constructed from the gates in Vlist.
    """
    grad = brickwall_unitary_grad(Vlist, L, U, perms)
    # project gradient onto unitary manifold, represent as anti-symmetric matrix
    # and then convert to a vector
    return np.stack([antisymm_to_real(
        antisymm(Vlist[j].conj().T @ grad[j]))
        for j in range(len(grad))]).reshape(-1)


def brickwall_ortho_gradient_vector(Vlist, L, U, perms):
    """
    Represent the gradient of tr[Uᵀ W] as real vector,
    where W is the brickwall circuit constructed from the gates in Vlist,
    for the case of real-valued matrices.
    """
    if any([np.iscomplexobj(V) for V in Vlist]) or np.iscomplexobj(U):
        raise ValueError("requiring real-valued input matrices")
    grad = brickwall_unitary_grad(Vlist, L, U, perms)
    # project gradient onto orthogonal matrix manifold, represent as skew-symmetric matrix
    # and then convert to a vector
    return np.stack([skew_to_real(
        antisymm(Vlist[j].T @ grad[j]))
        for j in range(len(grad))]).reshape(-1)


def brickwall_unitary_directed_grad(Vlist, L, Z, k, perms):
    """
    Compute the gradient of W in direction Z with respect to Vlist[k],
    where W is the brickwall circuit constructed from the gates in Vlist.
    """
    return (  brickwall_unitary(Vlist[k+1:], L, perms[k+1:])
            @ parallel_gates_directed_grad(Vlist[k], L, Z, perms[k])
            @ brickwall_unitary(Vlist[:k], L, perms[:k]))


def brickwall_unitary_hess(Vlist, L, Z, k, U, perms, unitary_proj=False):
    """
    Compute the Hessian of Re tr[U† W] in direction Z with respect to Vlist[k],
    where W is the brickwall circuit constructed from the gates in Vlist.
    """
    n = len(Vlist)
    dVlist = np.stack([np.zeros_like(V) for V in Vlist])
    for j in range(k):
        # j < k
        # directed gradient with respect to Vlist[k] in direction Z
        UdZk = (  brickwall_unitary(Vlist[j+1:k], L, perms[j+1:k]).conj().T
                @ parallel_gates_directed_grad(Vlist[k], L, Z, perms[k]).conj().T
                @ brickwall_unitary(Vlist[k+1:], L, perms[k+1:]).conj().T
                @ U
                @ brickwall_unitary(Vlist[:j], L, perms[:j]).conj().T)
        dVj = parallel_gates_grad(Vlist[j], L, UdZk, perms[j])
        if unitary_proj:
            dVlist[j] += project_unitary_tangent(Vlist[j], dVj)
        else:
            dVlist[j] += dVj

    # Hessian for layer k
    Ueff = (  brickwall_unitary(Vlist[k+1:], L, perms[k+1:]).conj().T
            @ U
            @ brickwall_unitary(Vlist[:k], L, perms[:k]).conj().T)
    dVlist[k] += parallel_gates_hess(Vlist[k], L, Z, Ueff, perms[k], unitary_proj=unitary_proj)

    for j in range(k + 1, n):
        # k < j
        # directed gradient with respect to Vlist[k] in direction Z
        UdZk = (  brickwall_unitary(Vlist[j+1:], L, perms[j+1:]).conj().T
                @ U
                @ brickwall_unitary(Vlist[:k], L, perms[:k]).conj().T
                @ parallel_gates_directed_grad(Vlist[k], L, Z, perms[k]).conj().T
                @ brickwall_unitary(Vlist[k+1:j], L, perms[k+1:j]).conj().T)
        dVj = parallel_gates_grad(Vlist[j], L, UdZk, perms[j])
        if unitary_proj:
            dVlist[j] += project_unitary_tangent(Vlist[j], dVj)
        else:
            dVlist[j] += dVj

    return dVlist


def brickwall_unitary_hessian_matrix(Vlist, L, U, perms):
    """
    Construct the Hessian matrix of Re tr[U† W] with respect to Vlist
    defining the layers of W,
    where W is the brickwall circuit constructed from the gates in Vlist.
    """
    n = len(Vlist)
    H = np.zeros((n, 16, n, 16))
    for j in range(n):
        for k in range(16):
            # unit vector
            Z = np.zeros(16)
            Z[k] = 1
            Z = real_to_antisymm(np.reshape(Z, (4, 4)))
            dVZj = brickwall_unitary_hess(Vlist, L, Vlist[j] @ Z, j, U, perms, unitary_proj=True)
            for i in range(n):
                H[i, :, j, k] = antisymm_to_real(antisymm(Vlist[i].conj().T @ dVZj[i])).reshape(-1)
    return H.reshape((n * 16, n * 16))


def brickwall_ortho_hessian_matrix(Vlist, L, U, perms):
    """
    Construct the Hessian matrix of tr[Uᵀ W] with respect to Vlist
    defining the layers of W,
    where W is the brickwall circuit constructed from the gates in Vlist,
    for the case of real-valued matrices.
    """
    if any([np.iscomplexobj(V) for V in Vlist]) or np.iscomplexobj(U):
        raise ValueError("requiring real-valued input matrices")
    n = len(Vlist)
    H = np.zeros((n, 6, n, 6))
    for j in range(n):
        for k in range(6):
            # unit vector
            Z = np.zeros(6)
            Z[k] = 1
            Z = real_to_skew(Z, 4)
            dVZj = brickwall_unitary_hess(Vlist, L, Vlist[j] @ Z, j, U, perms, unitary_proj=True)
            for i in range(n):
                H[i, :, j, k] = skew_to_real(antisymm(Vlist[i].T @ dVZj[i]))
    return H.reshape((n * 6, n * 6))


def squared_brickwall_unitary_grad(Vlist, L, A, B, perms):
    """
    Compute the gradient of tr[A W† B W] with respect to Vlist,
    where W is the brickwall circuit constructed from the gates in Vlist
    and A and B are Hermitian.
    """
    return 2 * brickwall_unitary_grad(Vlist, L, B @ brickwall_unitary(Vlist, L, perms) @ A, perms)


def squared_brickwall_unitary_gradient_vector(Vlist, L, A, B, perms):
    """
    Represent the gradient of tr[A W† B W] as real vector,
    where W is the brickwall circuit constructed from the gates in Vlist
    and A and B are Hermitian.
    """
    return 2 * brickwall_unitary_gradient_vector(Vlist, L, B @ brickwall_unitary(Vlist, L, perms) @ A, perms)


def squared_brickwall_ortho_gradient_vector(Vlist, L, A, B, perms):
    """
    Represent the gradient of tr[A Wᵀ B W] as real vector,
    where W is the brickwall circuit constructed from the gates in Vlist
    and A and B are Hermitian,
    for the case of real-valued matrices.
    """
    return 2 * brickwall_ortho_gradient_vector(Vlist, L, B @ brickwall_unitary(Vlist, L, perms) @ A, perms)


def squared_brickwall_unitary_hess(Vlist, L, Z, k, A, B, perms, unitary_proj=False):
    """
    Compute the Hessian of tr[A W† B W] in direction Z with respect to Vlist[k],
    where W is the brickwall circuit constructed from the gates in Vlist
    and A and B are Hermitian.
    """
    H1 = brickwall_unitary_hess(Vlist, L, Z, k, B @ brickwall_unitary(Vlist, L, perms) @ A, perms, unitary_proj)
    H2 = brickwall_unitary_grad(Vlist, L, B @ brickwall_unitary_directed_grad(Vlist, L, Z, k, perms) @ A, perms)
    if unitary_proj:
        H2 = np.stack([project_unitary_tangent(Vlist[j], dVj) for j, dVj in enumerate(H2)])
    return 2 * (H1 + H2)


def squared_brickwall_unitary_hessian_matrix(Vlist, L, A, B, perms):
    """
    Construct the Hessian matrix of tr[A W† B W] with respect to Vlist
    defining the layers of W,
    where W is the brickwall circuit constructed from the gates in Vlist.
    """
    n = len(Vlist)
    H = np.zeros((n, 16, n, 16))
    for j in range(n):
        for k in range(16):
            # unit vector
            Z = np.zeros(16)
            Z[k] = 1
            Z = real_to_antisymm(np.reshape(Z, (4, 4)))
            dVZj = squared_brickwall_unitary_hess(Vlist, L, Vlist[j] @ Z, j, A, B, perms, unitary_proj=True)
            for i in range(n):
                H[i, :, j, k] = antisymm_to_real(antisymm(Vlist[i].conj().T @ dVZj[i])).reshape(-1)
    return H.reshape((n * 16, n * 16))


def squared_brickwall_ortho_hessian_matrix(Vlist, L, A, B, perms):
    """
    Construct the Hessian matrix of tr[A Wᵀ B W] with respect to Vlist
    defining the layers of W,
    where W is the brickwall circuit constructed from the gates in Vlist,
    for the case of real-valued matrices.
    """
    if any([np.iscomplexobj(V) for V in Vlist]) or np.iscomplexobj(A) or np.iscomplexobj(B):
        raise ValueError("requiring real-valued input matrices")
    n = len(Vlist)
    H = np.zeros((n, 6, n, 6))
    for j in range(n):
        for k in range(6):
            # unit vector
            Z = np.zeros(6)
            Z[k] = 1
            Z = real_to_skew(Z, 4)
            dVZj = squared_brickwall_unitary_hess(Vlist, L, Vlist[j] @ Z, j, A, B, perms, unitary_proj=True)
            for i in range(n):
                H[i, :, j, k] = skew_to_real(antisymm(Vlist[i].T @ dVZj[i]))
    return H.reshape((n * 6, n * 6))


def permute_operation(U: np.ndarray, perm):
    """
    Find the representation of a matrix after permuting lattice sites.
    """
    nsites = len(perm)
    assert U.shape == (2**nsites, 2**nsites)
    perm = list(perm)
    U = np.reshape(U, (2*nsites) * (2,))
    U = np.transpose(U, perm + [nsites + p for p in perm])
    U = np.reshape(U, (2**nsites, 2**nsites))
    return U


def projection_probability(Vlist, L, perms, state=None):
    """
    Calculates the probability of measuring |0000...> on the ancillary qubits
    after the brickwall circuit.
    """
    # initial state |0...0> on the whole system
    if state is None:
        state = np.zeros(2**L)
        state[0] = 1
    assert(len(state)==2**L)
    state = np.einsum('ij,j', brickwall_unitary(Vlist, L, perms), state)
    p = blockenc_isometry(L//2)
    pp = p.dot(p.conj().T)
    return abs(np.einsum('i,ij,j', state.conj(), pp, state))**2