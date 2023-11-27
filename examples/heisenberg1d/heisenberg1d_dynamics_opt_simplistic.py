import numpy as np
import scipy
import qib
import h5py
import rqcopt as oc
import matplotlib.pyplot as plt


def heisenberg1d_dynamics_opt_simplistic(nlayers: int, **kwargs):
    """
    Optimize the quantum gates in a brickwall layout to approximate
    the time evolution governed by a Heisenberg-type Hamiltonian,
    using identity matrices as "simplistic" starting point.
    """
    # side length of lattice
    L = 6
    # Hamiltonian parameters
    J = ( 1,     1, -0.5)
    h = ( 0.75,  0,  0)

    # construct Hamiltonian
    latt = qib.lattice.IntegerLattice((L,), pbc=True)
    field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
    H = qib.HeisenbergHamiltonian(field, J, h).as_matrix()

    # time
    t = 0.25

    # reference global unitary
    expiH = scipy.linalg.expm(-1j*H.todense()*t)

    # unitaries used as starting point for optimization
    Vlist_start = [np.identity(4, dtype=complex) for _ in range(nlayers)]
    perms = [None if i % 2 == 0 else np.roll(range(L), 1) for i in range(len(Vlist_start))]
    # perform optimization
    Vlist, f_iter, err_iter = oc.optimize_brickwall_circuit(L, expiH, Vlist_start, perms, **kwargs)

    # visualize optimization progress
    print(f"err_iter before: {err_iter[0]}")
    print(f"err_iter after {len(err_iter)-1} iterations: {err_iter[-1]}")
    plt.semilogy(range(len(err_iter)), err_iter)
    plt.xlabel("iteration")
    plt.ylabel("spectral norm error")
    plt.title(f"optimization progress for a quantum circuit with {len(Vlist)} layers")
    plt.show()
    # rescaled and shifted target function
    plt.semilogy(range(len(f_iter)), 1 + np.array(f_iter) / 2**L)
    plt.xlabel("iteration")
    plt.ylabel(r"$1 + f(\mathrm{Vlist})/2^L$")
    plt.title(f"optimization target function for a quantum circuit with {len(Vlist)} layers")
    plt.show()

    # save results to disk
    f_iter = np.array(f_iter)
    err_iter = np.array(err_iter)
    with h5py.File(f"heisenberg1d_dynamics_opt_simplistic_n{nlayers}.hdf5", "w") as f:
        f.create_dataset("Vlist", data=Vlist)
        f.create_dataset("f_iter", data=f_iter)
        f.create_dataset("err_iter", data=err_iter)
        # store parameters
        f.attrs["L"] = L
        f.attrs["J"] = J
        f.attrs["h"] = h
        f.attrs["t"] = float(t)


def main():

    # Note: running this code might take several hours

    # 3 layers
    heisenberg1d_dynamics_opt_simplistic(3, niter=15)

    # 5 layers
    heisenberg1d_dynamics_opt_simplistic(5, niter=40)

    # 7 layers
    heisenberg1d_dynamics_opt_simplistic(7, niter=40)

    # 9 layers
    heisenberg1d_dynamics_opt_simplistic(9, niter=60)

    # 11 layers
    heisenberg1d_dynamics_opt_simplistic(11, niter=80)

    # 13 layers
    heisenberg1d_dynamics_opt_simplistic(13, niter=60)

    # 15 layers
    heisenberg1d_dynamics_opt_simplistic(15, niter=150)

    # 17 layers
    heisenberg1d_dynamics_opt_simplistic(17, niter=150)

    # 19 layers
    heisenberg1d_dynamics_opt_simplistic(19, niter=150)


if __name__ == "__main__":
    main()
