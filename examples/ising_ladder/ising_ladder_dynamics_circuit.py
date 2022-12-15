import numpy as np
import scipy
import matplotlib.pyplot as plt
import h5py
import qib
import rqcopt as oc


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


def trotterized_time_evolution(L: int, hloc, perm_set, method: oc.SplittingMethod, dt: float, nsteps: int):
    """
    Compute the numeric ODE flow operator of the quantum time evolution
    based on the provided splitting method.
    """
    Vlist = []
    perms = []
    for i, c in zip(method.indices, method.coeffs):
        Vlist.append(scipy.linalg.expm(-1j*c*dt*hloc[i]))
        perms.append(perm_set[i])
    V = oc.brickwall_unitary(Vlist, L, perms)
    return np.linalg.matrix_power(V, nsteps)


def main():

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
    # visualize spectrum
    λ = np.linalg.eigvalsh(H.todense())
    plt.plot(λ, '.')
    plt.xlabel(r"$j$")
    plt.ylabel(r"$\lambda_j$")
    plt.title(f"Ising Hamiltonian for J = {J}, g = {g} on a lattice with {Lx} x {Ly} sites")
    plt.show()

    # reference global unitary
    t = 0.25
    expiH = scipy.linalg.expm(-1j*H.todense()*t)

    # local Hamiltonian terms
    hloc_horz = construct_ising_local_term(J, g, 2)
    hloc_vert = construct_ising_local_term(J, 2*g, 2) # factor 2 due to special case of two lattice sites in current direction
    hloc = [hloc_horz, hloc_horz, hloc_vert]
    # permutations specifying gate layout
    horz_even_sites = [0, 2, 1, 3, 4, 6, 5, 7]
    horz_odd_sites  = [2, 4, 3, 5, 6, 0, 7, 1]
    perm_set = [list(np.argsort(horz_even_sites)),  # horizontal even
                list(np.argsort(horz_odd_sites)),   # horizontal odd
                None]                               # vertical

    # real-time evolution via Strang splitting for different time steps
    nsteps_stra = np.array([2**i for i in range(6)])
    err_stra = np.zeros(len(nsteps_stra))
    stra = oc.SplittingMethod.suzuki(3, 1)
    print("stra.coeffs:", stra.coeffs)
    for i, nsteps in enumerate(nsteps_stra):
        print("nsteps:", nsteps)
        dt = t / nsteps
        print("dt:", dt)
        W = trotterized_time_evolution(L, hloc, perm_set, stra, dt, nsteps)
        err_stra[i] = np.linalg.norm(W - expiH, ord=2)
        print(f"err_stra[{i}]: {err_stra[i]}")
    # convergence plot
    dt_list = t / nsteps_stra
    plt.loglog(dt_list, err_stra, '.-', label="Strang")
    plt.loglog(dt_list, 6*np.array(dt_list)**2, '--', label="Δt^2")
    plt.xlabel("Δt")
    plt.ylabel("error")
    plt.legend()
    plt.title(f"real-time evolution up to t = {t} using Strang splitting")
    plt.show()

    # real-time evolution via Suzuki's order-4 splitting for different time steps
    nsteps_suz4 = np.array(sorted([2**i for i in range(3)]))
    err_suz4 = np.zeros(len(nsteps_suz4))
    suz4 = oc.SplittingMethod.suzuki(3, 2)
    for i, nsteps in enumerate(nsteps_suz4):
        print("nsteps:", nsteps)
        dt = t / nsteps
        print("dt:", dt)
        W = trotterized_time_evolution(L, hloc, perm_set, suz4, dt, nsteps)
        err_suz4[i] = np.linalg.norm(W - expiH, ord=2)
        print(f"err_suz4[{i}]: {err_suz4[i]}")
    # convergence plot
    dt_list = t / nsteps_suz4
    plt.loglog(dt_list, err_suz4, '.-', label="Suzuki ")
    plt.loglog(dt_list, 5*np.array(dt_list)**4, '--', label="~Δt^4")
    plt.xlabel("Δt")
    plt.ylabel("error")
    plt.legend()
    plt.title(f"real-time evolution up to t = {t} using Suzuki's order-4 splitting")
    plt.show()

    # real-time evolution using Yoshida's minimal order-4 method for different time steps
    nsteps_yosh = np.array([2**i for i in range(4)])
    err_yosh = np.zeros(len(nsteps_yosh))
    yosh = oc.SplittingMethod.yoshida4(3)
    for i, nsteps in enumerate(nsteps_yosh):
        print("nsteps:", nsteps)
        dt = t / nsteps
        print("dt:", dt)
        W = trotterized_time_evolution(L, hloc, perm_set, yosh, dt, nsteps)
        err_yosh[i] = np.linalg.norm(W - expiH, ord=2)
        print(f"err_yosh[{i}]: {err_yosh[i]}")
    # convergence plot
    dt_list = t / nsteps_yosh
    plt.loglog(dt_list, err_yosh, '.-', label="Yoshida order 4")
    plt.loglog(dt_list, 50*np.array(dt_list)**4, '--', label="Δt^4")
    plt.xlabel("Δt")
    plt.ylabel("error")
    plt.legend()
    plt.title(f"real-time evolution up to t = {t} using Yoshida's method of order-4")
    plt.show()

    # real-time evolution using the order-6 method AY 15-6 by Auzinger et al. for different time steps
    nsteps_auzi = np.array([2**i for i in range(3)])
    err_auzi = np.zeros(len(nsteps_auzi))
    # recommended method is m = 5; we use the m = 4 method anyway since it requires two substeps less
    auzi = oc.SplittingMethod.auzinger15_6()
    for i, nsteps in enumerate(nsteps_auzi):
        print("nsteps:", nsteps)
        dt = t / nsteps
        print("dt:", dt)
        W = trotterized_time_evolution(L, hloc, perm_set, auzi, dt, nsteps)
        err_auzi[i] = np.linalg.norm(W - expiH, ord=2)
        print(f"err_auzi[{i}]: {err_auzi[i]}")
    # convergence plot
    dt_list = t / nsteps_auzi
    plt.loglog(dt_list, err_auzi, '.-', label="Auzinger AY 15-6")
    plt.loglog(dt_list, 100*np.array(dt_list)**6, '--', label="~Δt^6")
    plt.xlabel("Δt")
    plt.ylabel("error")
    plt.legend()
    plt.title(f"real-time evolution up to t = {t} using the AY 15-6 method of order 6")
    plt.show()

    # optimized circuits
    err_iter_opt = {}
    for nlayers in [5, 9, 13, 17]:
        with h5py.File(f"ising_ladder_dynamics_opt_n{nlayers}.hdf5", "r") as f:
            # parameters must agree
            assert f.attrs["Lx"] == Lx
            assert f.attrs["Ly"] == Ly
            assert f.attrs["J"]  == J
            assert f.attrs["g"]  == g
            assert f.attrs["t"]  == t
            err_iter_opt[nlayers] = f["err_iter"][:]

    # compare in terms of number of layers
    plt.loglog([5, 9, 13, 17],
                [err_iter_opt[n][-1] for n in [5, 9, 13, 17]], 'o-', linewidth=2, label="opt. circuit")
    plt.loglog((stra.num_layers-1)*nsteps_stra + 1, err_stra, '.-', label="Strang")
    plt.loglog((suz4.num_layers-1)*nsteps_suz4 + 1, err_suz4, 'v-', label="Suzuki order 4")
    plt.loglog((yosh.num_layers-1)*nsteps_yosh + 1, err_yosh, '^-', label="Yoshida order 4")
    plt.loglog((auzi.num_layers-1)*nsteps_auzi + 1, err_auzi, 'D-', label="Auzinger AY 15-6")
    xt = [5, 9, 13, 17, 25, 41, 65, 97]
    plt.xticks(xt, [rf"$\mathdefault{{{l}}}$" for l in xt])
    plt.xlabel("number of layers")
    plt.ylabel("error")
    plt.legend(loc="upper right")
    plt.title(rf"$\mathrm{{approximating }}\ e^{{-i H^{{\mathrm{{Ising}}}} t}} \ \mathrm{{on}} \ {Lx}\times{Ly} \ \mathrm{{lattice}}, J = {J}, g = {g}, t = {t}$")
    plt.savefig("ising_ladder_dynamics_circuit_error.pdf")
    plt.show()


if __name__ == "__main__":
    main()
