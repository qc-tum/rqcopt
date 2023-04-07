import numpy as np
import scipy
import qib
import h5py
import rqcopt as oc
import matplotlib.pyplot as plt


def ising1d_blockenc_opt(nlayers: int, bootstrap: bool, rng: np.random.Generator = None, **kwargs):
    """
    Optimize the quantum gates in a brickwall layout to approximate
    the an Ising Hamiltonian by block-encoding.
    """
    print(f"optimizing a circuit with {nlayers} layers...")

    # number of qubits (both physical and auxiliary)
    L = 6
    # Hamiltonian parameters
    scale = 0.25
    J = scale * 1
    g = scale * 0.75

    # construct Hamiltonian
    latt = qib.lattice.IntegerLattice((L // 2,), pbc=True)
    field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
    H = np.array(qib.IsingHamiltonian(field, J, 0., g).as_matrix().todense())
    # spectral norm must be smaller than 1
    nH = np.linalg.norm(H, ord=2)
    print(f"spectral norm of Hamiltonian: {nH} (must be smaller than 1)")

    print("Hamiltonian:")
    print(H)

    # unitaries used as starting point for optimization
    if bootstrap:
        # load optimized unitaries for nlayers - 2 from disk
        with h5py.File(f"ising1d_blockenc_opt_n{nlayers-2}.hdf5", "r") as f:
            # parameters must agree
            assert f.attrs["L"] == L
            assert f.attrs["J"] == J
            assert f.attrs["g"] == g
            Vlist_start = f["Vlist"][:]
            assert Vlist_start.shape[0] == nlayers - 2
        # pad identity matrices
        id4 = np.identity(4).reshape((1, 4, 4))
        Vlist_start = np.concatenate((id4, Vlist_start, id4), axis=0)
        assert Vlist_start.shape[0] == nlayers
    else:
        Vlist_start = [scipy.stats.unitary_group.rvs(4, random_state=rng) for _ in range(nlayers)]
    perms = [None if i % 2 == 0 else np.roll(range(L), -1) for i in range(-(nlayers // 2), (nlayers + 1) // 2)]
    assert len(perms) == nlayers
    # block-encoding isometry
    P = oc.blockenc_isometry(L // 2)

    # perform optimization
    Vlist, f_iter, err_iter = oc.optimize_brickwall_circuit_blockenc(L, H, P, Vlist_start, perms, **kwargs)

    # visualize optimization progress
    print(f"err_iter before: {err_iter[0]}")
    print(f"err_iter after {len(err_iter)-1} iterations: {err_iter[-1]}")
    plt.semilogy(range(len(err_iter)), err_iter)
    plt.xlabel("iteration")
    plt.ylabel("spectral norm error")
    plt.title(f"optimization progress for a quantum circuit with {len(Vlist)} layers")
    plt.show()
    # target function
    plt.semilogy(range(len(f_iter)), np.array(f_iter))
    plt.xlabel("iteration")
    plt.ylabel(r"$f(\mathrm{Vlist})$")
    plt.title(f"optimization target function for a quantum circuit with {len(Vlist)} layers")
    plt.show()

    # save results to disk
    f_iter = np.array(f_iter)
    err_iter = np.array(err_iter)
    with h5py.File(f"ising1d_blockenc_opt_n{nlayers}.hdf5", "w") as f:
        f.create_dataset("Vlist", data=Vlist)
        f.create_dataset("f_iter", data=f_iter)
        f.create_dataset("err_iter", data=err_iter)
        # store parameters
        f.attrs["L"] = L
        f.attrs["J"] = float(J)
        f.attrs["g"] = float(g)


def main():

    # 3 layers
    rng = np.random.default_rng(seed=142)
    ising1d_blockenc_opt(3, False, rng, niter=50)

    # 5 layers
    ising1d_blockenc_opt(5, True, niter=50)

    # 7 layers
    ising1d_blockenc_opt(7, True, niter=20)


if __name__ == "__main__":
    main()
