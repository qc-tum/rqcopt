import numpy as np
import scipy
import qib
import h5py
import rqcopt as oc
import matplotlib.pyplot as plt


def construct_kinetic_term(J: float):
    """
    Construct the kinetic hopping term of the Fermi-Hubbard Hamiltonian
    on a one-dimensional lattice, based on Jordan-Wigner encoding.
    """
    return -J * np.array([
        [0., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 0.]])


def construct_interaction_term(u: float):
    """
    Construct the local interaction term of the Fermi-Hubbard Hamiltonian
    on a one-dimensional lattice, based on Jordan-Wigner encoding.
    """
    return u * np.diag([0., 0., 0., 1.])


def fermi_hubbard1d_dynamics_opt(nlayers: int, bootstrap: bool, method_start: oc.SplittingMethod=None, **kwargs):
    """
    Optimize the quantum gates in a brickwall layout to approximate
    the time evolution governed by the Fermi-Hubbard Hamiltonian.
    """
    # side length of lattice (spinful sites)
    L = 4
    # Hamiltonian parameters
    J = 1
    u = 4

    # construct Hamiltonian
    # using open boundary conditions and manually adding modified wrap-around hopping term
    latt = qib.lattice.IntegerLattice((L,), pbc=False)
    field = qib.field.Field(qib.field.ParticleType.FERMION, qib.lattice.LayeredLattice(latt, 2))
    H = qib.FermiHubbardHamiltonian(field, float(J), float(u), spin=True).as_matrix().todense()
    # construct hopping term between first and second site and then shift circularly
    adj_single = np.zeros((L, L), dtype=int)
    adj_single[0, 1] = 1
    adj_single[1, 0] = 1
    hop_single = qib.operator.FieldOperatorTerm([qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_CREATE),
                                                 qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_ANNIHIL)],
                                                 -J * np.kron(np.identity(2), adj_single))
    T_wrap = np.asarray(qib.FieldOperator([hop_single]).as_matrix().todense())
    # circular shift
    T_wrap = oc.permute_operation(T_wrap, list(np.roll(range(L), -1)) + list(np.roll(range(L, 2*L), -1)))
    H += T_wrap

    # time
    t = 0.25

    # reference global unitary
    expiH = scipy.linalg.expm(-1j*H*t)

    # site permutations for correct mapping of two-qubit gates
    hop_even_sites = list(range(2*L))
    hop_odd_sites  = list(np.roll(range(L), 1)) + list(np.roll(range(L, 2*L), 1))
    intpot_sites   = [j + L*s for j in range(L) for s in range(2)]
    print("hop_even_sites:", hop_even_sites)
    print("hop_odd_sites: ", hop_odd_sites)
    print("intpot_sites:  ", intpot_sites)
    perm_set = [None,           # even hopping
                hop_odd_sites,  # odd hopping
                intpot_sites]   # interaction term

    # unitaries used as starting point for optimization
    if bootstrap:
        # load optimized unitaries for nlayers - 4 from disk
        with h5py.File(f"fermi_hubbard1d_dynamics_opt_n{nlayers-4}.hdf5", "r") as f:
            # parameters must agree
            assert f.attrs["L"] == L
            assert f.attrs["J"] == J
            assert f.attrs["u"] == u
            assert f.attrs["t"] == t
            Vlist_start = f["Vlist"][:]
            assert Vlist_start.shape[0] == nlayers - 4
        # pad identity matrices
        id4 = np.identity(4).reshape((1, 4, 4))
        Vlist_start = np.concatenate((Vlist_start, id4, id4, id4, id4), axis=0)
        assert Vlist_start.shape[0] == nlayers
        assert (nlayers - 1) % 4 == 0
        indices = [0] + ((nlayers - 1) // 4)*[1, 2, 1, 0]
        perms = [perm_set[i] for i in indices]
    else:
        assert method_start.nterms == 3
        assert len(method_start.coeffs) == nlayers
        # local Hamiltonian terms
        tkin = construct_kinetic_term(J)
        vpot = construct_interaction_term(u)
        Vlist_start = []
        perms = []
        for i, c in zip(method_start.indices, method_start.coeffs):
            Vlist_start.append(scipy.linalg.expm(-1j*c*t*(tkin if i <= 1 else vpot)))
            perms.append(perm_set[i])
    # perform optimization
    Vlist, f_iter, err_iter = oc.optimize_brickwall_circuit(2*L, expiH, Vlist_start, perms, **kwargs)

    # visualize optimization progress
    print(f"err_iter before: {err_iter[0]}")
    print(f"err_iter after {len(err_iter)-1} iterations: {err_iter[-1]}")
    plt.semilogy(range(len(err_iter)), err_iter)
    plt.xlabel("iteration")
    plt.ylabel("spectral norm error")
    plt.title(f"optimization progress for a quantum circuit with {len(Vlist)} layers")
    plt.show()
    # rescaled and shifted target function
    plt.semilogy(range(len(f_iter)), 1 + np.array(f_iter) / 2**(2*L))
    plt.xlabel("iteration")
    plt.ylabel(r"$1 + f(\mathrm{Vlist})/2^{2L}$")
    plt.title(f"optimization target function for a quantum circuit with {len(Vlist)} layers")
    plt.show()

    # save results to disk
    f_iter = np.array(f_iter)
    err_iter = np.array(err_iter)
    with h5py.File(f"fermi_hubbard1d_dynamics_opt_n{nlayers}.hdf5", "w") as f:
        f.create_dataset("Vlist", data=Vlist)
        f.create_dataset("f_iter", data=f_iter)
        f.create_dataset("err_iter", data=err_iter)
        # store parameters
        f.attrs["L"] = L
        f.attrs["J"] = float(J)
        f.attrs["u"] = float(u)
        f.attrs["t"] = float(t)


def main():

    # Note: running this code might take several hours

    # 5 layers
    # use a single Strang splitting step as starting point for optimization
    strang = oc.SplittingMethod.suzuki(3, 1)
    print("strang.coeffs:", strang.coeffs)
    print("strang.indices:", strang.indices)
    fermi_hubbard1d_dynamics_opt(5, False, strang, niter=10)

    # 9 layers
    # use two Strang splitting steps as starting point for optimization
    strang = oc.SplittingMethod.suzuki(3, 1)
    indices_start_n9, coeffs_start_n9 = oc.merge_layers(2*strang.indices, 2*strang.coeffs)
    # divide by 2 since we are taking two steps
    coeffs_start_n9 = [0.5*c for c in coeffs_start_n9]
    print("coeffs_start_n9:", coeffs_start_n9)
    print("indices_start_n9:", indices_start_n9)
    strang2 = oc.SplittingMethod(3, indices_start_n9, coeffs_start_n9, 2)
    fermi_hubbard1d_dynamics_opt(9, False, strang2, niter=16)

    # 13 layers
    # use three Strang splitting steps as starting point for optimization
    strang = oc.SplittingMethod.suzuki(3, 1)
    indices_start_n13, coeffs_start_n13 = oc.merge_layers(3*strang.indices, 3*strang.coeffs)
    # divide by 3 since we are taking three steps
    coeffs_start_n13 = [c/3 for c in coeffs_start_n13]
    print("coeffs_start_n13:", coeffs_start_n13)
    print("indices_start_n13:", indices_start_n13)
    strang3 = oc.SplittingMethod(3, indices_start_n13, coeffs_start_n13, 2)
    fermi_hubbard1d_dynamics_opt(13, False, strang3, niter=30)

    # 21 layers
    # use the order-4 Suzuki method as starting point for optimization
    suz4 = oc.SplittingMethod.suzuki(3, 2)
    print("suz4.coeffs:", suz4.coeffs)
    print("suz4.indices:", suz4.indices)
    # note: not fully converged yet, can reach even higher precision with more iterations
    fermi_hubbard1d_dynamics_opt(21, False, suz4, niter=80)


if __name__ == "__main__":
    main()
