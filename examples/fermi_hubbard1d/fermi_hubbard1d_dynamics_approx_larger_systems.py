import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import h5py
import qib
import rqcopt as oc


def compute_circuit_errors(J, u, Llist, t, nlayers):
    """
    Compute circuit approximation errors for various system sizes.
    """
    expiH = {}
    for L in Llist:
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
        # reference time evolution operator
        expiH[L] = scipy.linalg.expm(-1j*H*t)

    perm_set = {}
    for L in Llist:
        # site permutations for correct mapping of two-qubit gates
        hop_even_sites = None # equivalent to list(range(2*L))
        hop_odd_sites  = list(np.roll(range(L), 1)) + list(np.roll(range(L, 2*L), 1))
        intpot_sites   = [j + L*s for j in range(L) for s in range(2)]
        perm_set[L] = [hop_even_sites, # even hopping
                       hop_odd_sites,  # odd hopping
                       intpot_sites]   # interaction term

    # load optimized unitaries from disk
    Vlist = len(nlayers)*[None]
    indices = len(nlayers)*[None]
    err_opt = len(nlayers)*[None]
    for j, n in enumerate(nlayers):
        with h5py.File(f"fermi_hubbard1d_dynamics_opt_n{n}.hdf5", "r") as f:
            # parameters must agree
            assert f.attrs["L"] == Llist[0]
            assert f.attrs["J"] == J
            assert f.attrs["u"] == u
            assert f.attrs["t"] == t
            Vlist[j] = f["Vlist"][:]
            assert Vlist[j].shape[0] == n
            err_opt[j] = f["err_iter"][-1]
        assert (n - 1) % 4 == 0
        indices[j] = [0] + ((n - 1) // 4)*[1, 2, 1, 0]

    # approximation error of optimized circuits for larger system sizes
    circ_err = np.zeros((len(Llist), len(nlayers)))
    for i, L in enumerate(Llist):
        for j in range(len(nlayers)):
            perms = [perm_set[L][k] for k in indices[j]]
            circ_err[i, j] = np.linalg.norm(oc.brickwall_unitary(Vlist[j], 2*L, perms) - expiH[L], ord=2)

    print("error computation consistency check:", np.linalg.norm(err_opt - circ_err[0], np.inf))

    return circ_err


def main(recompute=True):

    # Hamiltonian parameters
    J = 1
    u = 4

    # various system sizes
    Llist = [4, 6]

    # time
    t = 0.25

    # number of circuit layers
    nlayers = [5, 9, 13, 21]

    if recompute:
        circ_err = compute_circuit_errors(J, u, Llist, t, nlayers)
        # save errors to disk
        with h5py.File("fermi_hubbard1d_dynamics_approx_larger_systems.hdf5", "w") as f:
            f.create_dataset("circ_err", data=circ_err)
            # store parameters
            f.attrs["J"] = float(J)
            f.attrs["u"] = float(u)
            f.attrs["t"] = float(t)
            f.attrs["Llist"] = Llist
            f.attrs["nlayers"] = nlayers
    else:
        # load errors from disk
        with h5py.File("fermi_hubbard1d_dynamics_approx_larger_systems.hdf5", "r") as f:
            # parameters must agree
            assert f.attrs["J"] == J
            assert f.attrs["u"] == u
            assert f.attrs["t"] == t
            assert np.array_equal(f.attrs["Llist"], Llist)
            assert np.array_equal(f.attrs["nlayers"], nlayers)
            circ_err = f["circ_err"][:]
    print(circ_err)

    # define plot colors
    clr_base = mc.to_rgb(plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
    clrs = len(Llist)*[None]
    for i in range(len(Llist)):
        s = i / (len(Llist) - 1)
        clrs[i] = ((1 - s)*clr_base[0], (1 - s)*clr_base[1], (1 - s)*clr_base[2])

    for i, L in enumerate(Llist):
        plt.loglog(nlayers, circ_err[i], '.-', color=clrs[i], label=f"L = {L}")
    xt = [5, 6, 9, 13, 17]
    plt.xticks(xt, [rf"$\mathdefault{{{l}}}$" if l % 2 == 1 else "" for l in xt])
    plt.xlabel("number of layers")
    plt.ylabel("error")
    plt.legend(loc="upper right")
    plt.title(rf"$\mathrm{{approximating }}\ e^{{-i H^{{\mathrm{{FH}}}} t}} \ \mathrm{{for}} \ J = {J}, u = {u}, t = {t}$")
    plt.savefig("fermi_hubbard1d_dynamics_approx_larger_systems.pdf")
    plt.show()


if __name__ == "__main__":
    main()
