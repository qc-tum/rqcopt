import numpy as np
import scipy
import qib
import h5py
import rqcopt as oc
import matplotlib.pyplot as plt


def construct_ising_local_term(J, g, ndim):
    """
    Construct local interaction term of Ising Hamiltonian on a 'ndim'-dimensional
    lattice for interaction parameter `J` and external field parameter `g`.
    """
    # Pauli-X and Z matrices
    X = np.array([[0.,  1.], [1.,  0.]])
    Z = np.array([[1.,  0.], [0., -1.]])
    I = np.identity(2)
    return J*np.kron(Z, Z) + g*(0.5/ndim)*(np.kron(X, I) + np.kron(I, X))


def ising_ladder_dynamics_opt(nlayers: int, method_start: oc.SplittingMethod=None, **kwargs):
    """
    Optimize the quantum gates in a brickwall layout to approximate
    the time evolution governed by an Ising Hamiltonian.
    """
    # side lengths of lattice
    Lx = 4
    Ly = 2
    # total number of lattice sites
    L = Lx * Ly

    # Hamiltonian parameters
    J = 1
    g = 3

    # construct Hamiltonian
    latt = qib.lattice.IntegerLattice((Lx, Ly), pbc=True)
    field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
    H = qib.IsingHamiltonian(field, J, 0., g).as_matrix()
    print("H.shape:", H.shape)

    # time
    t = 0.25
    print("t:", t)

    # reference global unitary
    expiH = scipy.linalg.expm(-1j*H.todense()*t)

    assert method_start.nterms == 3
    assert len(method_start.coeffs) == nlayers
    # local Hamiltonian terms
    hloc_horz = construct_ising_local_term(J, g, 2)
    hloc_vert = construct_ising_local_term(J, 2*g, 2) # factor 2 due to special case of two lattice sites in current direction
    hloc = [hloc_horz, hloc_horz, hloc_vert]
    # permutations specifying gate layout
    horz_even_sites = [0, 2, 1, 3, 4, 6, 5, 7]
    horz_odd_sites  = [2, 4, 3, 5, 6, 0, 7, 1]
    perm_set = [horz_even_sites, # horizontal even
                horz_odd_sites,  # horizontal odd
                None]            # vertical

    # unitaries used as starting point for optimization
    Vlist_start = []
    perms = []
    for i, c in zip(method_start.indices, method_start.coeffs):
        Vlist_start.append(scipy.linalg.expm(-1j*c*t*hloc[i]))
        perms.append(perm_set[i])
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
    with h5py.File(f"ising_ladder_dynamics_opt_n{nlayers}.hdf5", "w") as f:
        f.create_dataset("Vlist", data=Vlist)
        f.create_dataset("indices", data=method_start.indices)
        f.create_dataset("f_iter", data=f_iter)
        f.create_dataset("err_iter", data=err_iter)
        # store parameters
        f.attrs["Lx"] = Lx
        f.attrs["Ly"] = Ly
        f.attrs["J"]  = float(J)
        f.attrs["g"]  = float(g)
        f.attrs["t"]  = float(t)


def main():

    # Note: running this code might take several hours

    # 5 layers
    # use a single Strang splitting step as starting point for optimization
    strang = oc.SplittingMethod.suzuki(3, 1)
    print("strang.coeffs:", strang.coeffs)
    print("strang.indices:", strang.indices)
    ising_ladder_dynamics_opt(5, strang, niter=10)

    # 9 layers
    # use two Strang splitting steps as starting point for optimization
    strang = oc.SplittingMethod.suzuki(3, 1)
    indices_start_n9, coeffs_start_n9 = oc.merge_layers(2*strang.indices, 2*strang.coeffs)
    # divide by 2 since we are taking two steps
    coeffs_start_n9 = [0.5*c for c in coeffs_start_n9]
    print("coeffs_start_n9:", coeffs_start_n9)
    print("indices_start_n9:", indices_start_n9)
    method_start_n9 = oc.SplittingMethod(3, indices_start_n9, coeffs_start_n9, 2)
    ising_ladder_dynamics_opt(9, method_start_n9, niter=16)

    # 13 layers
    # use three Strang splitting steps as starting point for optimization
    strang = oc.SplittingMethod.suzuki(3, 1)
    indices_start_n13, coeffs_start_n13 = oc.merge_layers(3*strang.indices, 3*strang.coeffs)
    # divide by 3 since we are taking three steps
    coeffs_start_n13 = [c/3 for c in coeffs_start_n13]
    print("coeffs_start_n13:", coeffs_start_n13)
    print("indices_start_n13:", indices_start_n13)
    method_start_n13 = oc.SplittingMethod(3, indices_start_n13, coeffs_start_n13, 2)
    ising_ladder_dynamics_opt(13, method_start_n13, niter=30)

    # 17 layers
    # use four Strang splitting steps as starting point for optimization
    strang = oc.SplittingMethod.suzuki(3, 1)
    indices_start_n17, coeffs_start_n17 = oc.merge_layers(4*strang.indices, 4*strang.coeffs)
    # divide by 4 since we are taking four steps
    coeffs_start_n17 = [c/4 for c in coeffs_start_n17]
    print("coeffs_start_n17:", coeffs_start_n17)
    print("indices_start_n17:", indices_start_n17)
    method_start_n17 = oc.SplittingMethod(3, indices_start_n17, coeffs_start_n17, 2)
    ising_ladder_dynamics_opt(17, method_start_n17, niter=30)


if __name__ == "__main__":
    main()
