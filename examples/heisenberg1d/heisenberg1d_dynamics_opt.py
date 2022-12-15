import numpy as np
import scipy
import qib
import h5py
import rqcopt as oc
import matplotlib.pyplot as plt


def construct_heisenberg_local_term(J, h):
    """
    Construct local interaction term of a Heisenberg-type Hamiltonian on a one-dimensional
    lattice for interaction parameters `J` and external field parameters `h`.
    """
    # Pauli matrices
    X = np.array([[ 0.,  1.], [ 1.,  0.]])
    Y = np.array([[ 0., -1j], [ 1j,  0.]])
    Z = np.array([[ 1.,  0.], [ 0., -1.]])
    I = np.identity(2)
    return (  J[0]*np.kron(X, X)
            + J[1]*np.kron(Y, Y)
            + J[2]*np.kron(Z, Z)
            + h[0]*0.5*(np.kron(X, I) + np.kron(I, X))
            + h[1]*0.5*(np.kron(Y, I) + np.kron(I, Y))
            + h[2]*0.5*(np.kron(Z, I) + np.kron(I, Z)))


def heisenberg1d_dynamics_opt(nlayers: int, bootstrap: bool, coeffs_start=[], **kwargs):
    """
    Optimize the quantum gates in a brickwall layout to approximate
    the time evolution governed by a Heisenberg-type Hamiltonian.
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
    if bootstrap:
        # load optimized unitaries for nlayers - 2 from disk
        with h5py.File(f"heisenberg1d_dynamics_opt_n{nlayers-2}.hdf5", "r") as f:
            # parameters must agree
            assert f.attrs["L"] == L
            assert np.array_equal(f.attrs["J"], J)
            assert np.array_equal(f.attrs["h"], h)
            assert f.attrs["t"] == t
            Vlist_start = f["Vlist"][:]
            assert Vlist_start.shape[0] == nlayers - 2
        # pad identity matrices
        id4 = np.identity(4).reshape((1, 4, 4))
        Vlist_start = np.concatenate((id4, Vlist_start, id4), axis=0)
        assert Vlist_start.shape[0] == nlayers
        perms = [None if i % 2 == 1 else np.roll(range(L), -1) for i in range(len(Vlist_start))]
    else:
        # local Hamiltonian term
        hloc = construct_heisenberg_local_term(J, h)
        assert len(coeffs_start) == nlayers
        Vlist_start = [scipy.linalg.expm(-1j*c*t*hloc) for c in coeffs_start]
        perms = [None if i % 2 == 0 else np.roll(range(L), -1) for i in range(len(Vlist_start))]
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
    with h5py.File(f"heisenberg1d_dynamics_opt_n{nlayers}.hdf5", "w") as f:
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
    # use a single Strang splitting step as starting point for optimization
    strang = oc.SplittingMethod.suzuki(2, 1)
    heisenberg1d_dynamics_opt(3, False, strang.coeffs, niter=15)

    # 5 layers
    heisenberg1d_dynamics_opt(5, True, niter=40)

    # 7 layers
    heisenberg1d_dynamics_opt(7, True, niter=40)

    # 9 layers
    # use order-4 RKN method by Robert I. McLachlan as starting point for optimization
    heisenberg1d_dynamics_opt(9, False, oc.SplittingMethod.mclachlan4(4).coeffs, niter=60)

    # 11 layers
    heisenberg1d_dynamics_opt(11, True, niter=80)

    # 13 layers
    # use order-4 PRK method by Blanes and Moan as starting point for optimization
    heisenberg1d_dynamics_opt(13, False, oc.SplittingMethod.blanes_moan().coeffs, niter=60)

    # 15 layers
    heisenberg1d_dynamics_opt(15, True, niter=150, tcg_abstol=1e-12, tcg_reltol=1e-10)

    # 17 layers
    # use two steps of order-4 RKN method by Robert I. McLachlan as starting point for optimization
    mcla = oc.SplittingMethod.mclachlan4(4)
    _, coeffs_start_n17 = oc.merge_layers(2*mcla.indices, 2*mcla.coeffs)
    # divide by 2 since we are taking two steps
    coeffs_start_n17 = [0.5*c for c in coeffs_start_n17]
    print("coeffs_start_n17:", coeffs_start_n17)
    heisenberg1d_dynamics_opt(17, False, coeffs_start_n17, niter=150)

    # 19 layers
    heisenberg1d_dynamics_opt(19, True, niter=250, tcg_abstol=1e-12, tcg_reltol=1e-10)


if __name__ == "__main__":
    main()
