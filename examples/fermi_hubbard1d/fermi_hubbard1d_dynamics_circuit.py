import numpy as np
import scipy
import matplotlib.pyplot as plt
import h5py
import qib
import rqcopt as oc


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
    # visualize spectrum
    λ = np.linalg.eigvalsh(H)
    plt.plot(λ, '.')
    plt.xlabel(r"$j$")
    plt.ylabel(r"$\lambda_j$")
    plt.title(f"modified Fermi-Hubbard Hamiltonian for J = {J}, u = {u} on a 1D lattice with {L} sites")
    plt.show()

    # reference global unitary
    t = 0.25
    expiH = scipy.linalg.expm(-1j*H*t)

    # local Hamiltonian terms
    tkin = construct_kinetic_term(J)
    vpot = construct_interaction_term(u)
    hloc = [tkin, tkin, vpot]
    hop_even_sites = list(range(2*L))
    hop_odd_sites  = list(np.roll(range(L), 1)) + list(np.roll(range(L, 2*L), 1))
    intpot_sites   = [j + L*s for j in range(L) for s in range(2)]
    print("hop_even_sites:", hop_even_sites)
    print("hop_odd_sites: ", hop_odd_sites)
    print("intpot_sites:  ", intpot_sites)
    perm_set = [None,           # even hopping
                hop_odd_sites,  # odd hopping
                intpot_sites]   # interaction term

    # real-time evolution via Strang splitting for different time steps
    nsteps_stra = np.array([2**i for i in range(4)])
    err_stra = np.zeros(len(nsteps_stra))
    stra = oc.SplittingMethod.suzuki(3, 1)
    print("stra.coeffs:", stra.coeffs)
    for i, nsteps in enumerate(nsteps_stra):
        print("nsteps:", nsteps)
        dt = t / nsteps
        print("dt:", dt)
        W = trotterized_time_evolution(2*L, hloc, perm_set, stra, dt, nsteps)
        err_stra[i] = np.linalg.norm(W - expiH, ord=2)
        print(f"err_stra[{i}]: {err_stra[i]}")
    # convergence plot
    dt_list = t / nsteps_stra
    plt.loglog(dt_list, err_stra, '.-', label="Strang")
    plt.loglog(dt_list, 2*np.array(dt_list)**2, '--', label="~Δt^2")
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
        W = trotterized_time_evolution(2*L, hloc, perm_set, suz4, dt, nsteps)
        err_suz4[i] = np.linalg.norm(W - expiH, ord=2)
        print(f"err_suz4[{i}]: {err_suz4[i]}")
    # convergence plot
    dt_list = t / nsteps_suz4
    plt.loglog(dt_list, err_suz4, '.-', label="Suzuki ")
    plt.loglog(dt_list, 0.5*np.array(dt_list)**4, '--', label="~Δt^4")
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
        W = trotterized_time_evolution(2*L, hloc, perm_set, yosh, dt, nsteps)
        err_yosh[i] = np.linalg.norm(W - expiH, ord=2)
        print(f"err_yosh[{i}]: {err_yosh[i]}")
    # convergence plot
    dt_list = t / nsteps_yosh
    plt.loglog(dt_list, err_yosh, '.-', label="Yoshida order 4")
    plt.loglog(dt_list, 10*np.array(dt_list)**4, '--', label="~Δt^4")
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
        W = trotterized_time_evolution(2*L, hloc, perm_set, auzi, dt, nsteps)
        err_auzi[i] = np.linalg.norm(W - expiH, ord=2)
        print(f"err_auzi[{i}]: {err_auzi[i]}")
    # convergence plot
    dt_list = t / nsteps_auzi
    plt.loglog(dt_list, err_auzi, '.-', label="Auzinger AY 15-6")
    plt.loglog(dt_list, 6*np.array(dt_list)**6, '--', label="~Δt^6")
    plt.xlabel("Δt")
    plt.ylabel("error")
    plt.legend()
    plt.title(f"real-time evolution up to t = {t} using the AY 15-6 method of order 6")
    plt.show()

    # optimized circuits
    err_iter_opt = {}
    for nlayers in [5, 9, 13, 21]:
        with h5py.File(f"fermi_hubbard1d_dynamics_opt_n{nlayers}.hdf5", "r") as f:
            # parameters must agree
            assert f.attrs["L"] == L
            assert f.attrs["J"] == J
            assert f.attrs["u"] == u
            assert f.attrs["t"] == t
            err_iter_opt[nlayers] = f["err_iter"][:]

    # compare in terms of number of layers
    plt.loglog([5, 9, 13, 21],
               [err_iter_opt[n][-1] for n in [5, 9, 13, 21]], 'o-', linewidth=2, label="opt. circuit")
    plt.loglog((stra.num_layers-1)*nsteps_stra + 1, err_stra, '.-', label="Strang")
    plt.loglog((suz4.num_layers-1)*nsteps_suz4 + 1, err_suz4, 'v-', label="Suzuki order 4")
    plt.loglog((yosh.num_layers-1)*nsteps_yosh + 1, err_yosh, '^-', label="Yoshida order 4")
    plt.loglog((auzi.num_layers-1)*nsteps_auzi + 1, err_auzi, 'D-', label="Auzinger AY 15-6")
    xt = [3, 5, 7, 9, 13, 25, 49]
    plt.xticks(xt, [rf"$\mathdefault{{{l}}}$" for l in xt])
    plt.xlabel("number of layers")
    plt.ylabel("error")
    plt.legend(loc="upper right")
    plt.title(rf"$\mathrm{{approximating }}\ e^{{-i H^{{\mathrm{{FH}}}} t}} \ \mathrm{{on}} \ \mathrm{{1D}} \ \mathrm{{lattice}} \ \mathrm{{with}} \ {L} \ \mathrm{{sites}}, J = {J}, u = {u}, t = {t}$")
    plt.savefig("fermi_hubbard1d_dynamics_circuit_error.pdf")
    plt.show()


if __name__ == "__main__":
    main()
