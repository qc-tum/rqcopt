import numpy as np
import scipy
import qib
import h5py
import rqcopt as oc
import matplotlib.pyplot as plt


def create_heisenberg_hamiltonian():
    """
    Create the Heisenberg Hamiltonian.
    """
    # number of physical qubits
    L = 3
    # Hamiltonian parameters
    J = (1, 1, -0.5)
    h = (0.75, 0, 0)

    # construct Hamiltonian
    latt = qib.lattice.IntegerLattice((L,), pbc=True)
    field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
    return qib.HeisenbergHamiltonian(field, J, h)


def heisenberg1d_blockenc_opt(nlayers: int, bootstrap: bool, real: bool, rng: np.random.Generator = None, anc = None, **kwargs):
    """
    Optimize the quantum gates in a brickwall layout to approximate
    the an Heisenberg Hamiltonian by block-encoding.
    """
    print(f"optimizing a circuit with {nlayers} layers...")

    H_op = create_heisenberg_hamiltonian()
    if anc is not None:
        L = H_op.nsites + len(anc)
    else:
        L = 2*H_op.nsites
    # spectral norm must be smaller than 1
    nH = np.linalg.norm(H_op.as_matrix().todense(), ord=2)
    scale = 1.25
    H_op.J /= scale*nH
    H_op.h /= scale*nH
    H = np.array(H_op.as_matrix().todense())
    if real:
        H = np.real(H)
    nH = np.linalg.norm(H, ord=2)
    assert(nH < 1)
    print(f"spectral norm of Hamiltonian: {nH} (must be smaller than 1)")

    print("Hamiltonian:")
    print(H)

    # unitaries used as starting point for optimization
    if bootstrap:
        # load optimized unitaries for nlayers - 2 from disk
        oum = "real" if real else "complex"
        with h5py.File(f"heisenberg1d_blockenc_opt_n{nlayers-2}_{oum}.hdf5", "r") as f:
            # parameters must agree
            assert f.attrs["L"] == L
            assert all([a == b for a, b in zip(f.attrs["J"], H_op.J)])
            assert all([a == b for a, b in zip(f.attrs["h"], H_op.h)])
            if anc is not None:
                assert all([a == b for a, b in zip(f.attrs["anc"], anc)])
            Vlist_start = f["Vlist"][:]
            assert Vlist_start.shape[0] == nlayers - 2
        # pad identity matrices
        id4 = np.identity(4).reshape((1, 4, 4))
        Vlist_start = np.concatenate((id4, Vlist_start, id4), axis=0)
        assert Vlist_start.shape[0] == nlayers
    else:
        if real:
            Vlist_start = [scipy.stats.ortho_group.rvs(4, random_state=rng) for _ in range(nlayers)]
        else:
            Vlist_start = [scipy.stats.unitary_group.rvs(4, random_state=rng) for _ in range(nlayers)]
    perms = [None if i % 2 == 0 else np.roll(range(L), 1) for i in range(-(nlayers // 2), (nlayers + 1) // 2)]
    assert len(perms) == nlayers
    # block-encoding isometry
    P = oc.blockenc_isometry(L, anc)

    # perform optimization
    Vlist, f_iter, err_iter = oc.optimize_brickwall_circuit_blockenc(L, H, P, Vlist_start, perms, real, **kwargs)
    prob = oc.projection_probability(Vlist, L, perms, anc)
    # visualize optimization progress
    print(f"err_iter before: {err_iter[0]}")
    print(f"err_iter after {len(err_iter)-1} iterations: {err_iter[-1]}")
    print(f"final probability of measuring 0 on the ancillary qubits: {prob}")
    plt.semilogy(range(len(err_iter)), err_iter)
    plt.xlabel("iteration")
    plt.ylabel("spectral norm error")
    plt.title(f"optimization progress for a quantum circuit with {len(Vlist)} layers (Heisenberg)")
    plt.show()
    # target function
    plt.semilogy(range(len(f_iter)), np.array(f_iter))
    plt.xlabel("iteration")
    plt.ylabel(r"$f(\mathrm{Vlist})$")
    plt.title(f"optimization target function for a quantum circuit with {len(Vlist)} layers (Heisenberg)")
    plt.show()

    # save results to disk
    f_iter = np.array(f_iter)
    err_iter = np.array(err_iter)
    oum = "real" if real else "complex"
    with h5py.File(f"heisenberg1d_blockenc_opt_n{nlayers}_{oum}.hdf5", "w") as f:
        f.create_dataset("Vlist", data=Vlist)
        f.create_dataset("f_iter", data=f_iter)
        f.create_dataset("err_iter", data=err_iter)
        # store parameters
        f.attrs["L"] = L
        f.attrs["J"] = H_op.J
        f.attrs["h"] = H_op.h
        f.attrs["prob"] = prob
        if anc is not None:
            f.attrs["anc"] = anc


def main():
    # settings
    rng = np.random.default_rng(seed=142)
    #anc = [0]
    anc = [i for i in range(1, 6, 2)]
    restart = True
    real = False
    
    # 3 layers
    heisenberg1d_blockenc_opt(3, False, real, rng, anc, niter=100)

    # 5 layers
    heisenberg1d_blockenc_opt(5, restart, real, rng, anc, niter=100)

    # 7 layers
    heisenberg1d_blockenc_opt(7, restart, real, rng, anc, niter=100)

    # 9 layers
    heisenberg1d_blockenc_opt(9, restart, real, rng, anc, niter=100)


if __name__ == "__main__":
    main()
