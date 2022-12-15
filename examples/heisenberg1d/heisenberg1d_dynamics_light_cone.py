import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import qib


def sparse_local_single_op(U, j, L):
    """
    Construct the overall sparse matrix representation
    of the gate `U` acting on site `j`.
    """
    assert U.shape == (2, 2)
    return sparse.kron(sparse.eye(2**j), sparse.kron(U, sparse.eye(2**(L-j-1))))


def main():

    # side length of lattice
    L = 13

    # Hamiltonian parameters
    J = ( 1,     1, -0.5)
    h = ( 0.75,  0,  0)

    # construct Hamiltonian
    latt = qib.lattice.IntegerLattice((L,), pbc=True)
    field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
    H = qib.HeisenbergHamiltonian(field, J, h).as_matrix()
    print("H.shape:", H.shape)

    # find ground state wavefunction
    # compute algebraically smallest eigenvalues and corresponding eigenvectors of H
    ϵ0, ψ0 = sparse.linalg.eigsh(H, which="SA")
    i = np.argmin(ϵ0)
    ϵ0 = ϵ0[i]
    ψ0 = ψ0[:, i]
    print("exact ground state energy of H:", ϵ0)

    # visualize spectrum
    if L <= 10:
        λ = np.linalg.eigvalsh(H.todense())
        plt.plot(λ, '.')
        plt.xlabel(r"$j$")
        plt.ylabel(r"$\lambda_j$")
        plt.show()
        print(f"abs(min(λ) - ϵ0): {abs(min(λ) - ϵ0)} (should be zero)")

    # Pauli-Z matrix
    Z = sparse.csr_matrix([[1, 0], [0, -1]], dtype=float)

    # 〈Z_j〉
    zavr = np.vdot(ψ0, sparse_local_single_op(Z, 0, L) @ ψ0)
    print("zavr:", zavr)

    # Z_j measurement operators
    meas_ops = [sparse_local_single_op(Z - zavr*np.identity(2), j, L) for j in range(L)]

    Δt = 0.01
    nsteps = 200
    # for `expm_multiply`
    H = sparse.csc_matrix(H)

    # store measurement average values
    corr = np.zeros((L, nsteps+1), dtype=complex)

    ψt = ψ0.copy()
    # apply Z_j to ψ0
    Zψ = sparse_local_single_op(Z - zavr*np.identity(2), 0, L) @ ψ0
    print(f"np.vdot(ψ0, Zψ): {np.vdot(ψ0, Zψ)} (should be zero)")

    for n in range(nsteps + 1):
        for j in range(L):
            corr[j, n] = np.vdot(ψt, meas_ops[j] @ Zψ)
        # wavefunctions at next time step
        ψt = sparse.linalg.expm_multiply(-1j*H*Δt, ψt)
        Zψ = sparse.linalg.expm_multiply(-1j*H*Δt, Zψ)

    # visualize dynamical correlation functions
    vel = 2.75
    plt.imshow(np.roll(corr, shift=(L-1)//2, axis=0).real.T,
               interpolation="nearest", aspect="auto",
               origin="lower", extent=(-L//2+0.5, L//2+0.5, 0, Δt*nsteps))
    plt.xlabel("j")
    plt.ylabel("t")
    plt.title(fr"$\langle \psi | Z_j(t) Z_0(0) | \psi \rangle$ for J={J}, h={h}; velocity: {vel}")
    plt.colorbar()
    plt.plot([ 1, 1 + 2*vel], [ 0, 2], "w")
    plt.plot([-1,-1 - 2*vel], [ 0, 2], "w")
    plt.savefig("heisenberg1d_dynamics_light_cone.pdf")
    plt.show()


if __name__ == "__main__":
    main()
