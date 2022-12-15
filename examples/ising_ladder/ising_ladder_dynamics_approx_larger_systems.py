import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import h5py
import qib
import rqcopt as oc


def compute_circuit_errors(J, g, Lxlist, t, nlayers):
    """
    Compute circuit approximation errors for various system sizes.
    """
    expiH = {}
    for Lx in Lxlist:
        # construct Hamiltonian
        latt = qib.lattice.IntegerLattice((Lx, 2), pbc=True)
        field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
        H = qib.IsingHamiltonian(field, J, 0., g).as_matrix()
        # reference time evolution operator
        expiH[Lx] = scipy.linalg.expm(-1j*H.todense()*t)

    perm_set = {}
    for Lx in Lxlist:
        # permutations specifying gate layout
        horz_even_sites = list(range(0, 2*Lx, 2)) + list(range(1, 2*Lx, 2))
        horz_odd_sites  = list(np.roll(range(0, 2*Lx, 2), -1)) + list(np.roll(range(1, 2*Lx, 2), -1))
        perm_set[Lx] = [list(np.argsort(horz_even_sites)),  # horizontal even
                        list(np.argsort(horz_odd_sites)),   # horizontal odd
                        None]                               # vertical

    # load optimized unitaries from disk
    Vlist = len(nlayers)*[None]
    indices = len(nlayers)*[None]
    err_opt = len(nlayers)*[None]
    for j, n in enumerate(nlayers):
        with h5py.File(f"ising_ladder_dynamics_opt_n{n}.hdf5", "r") as f:
            # parameters must agree
            assert f.attrs["Lx"] == Lxlist[0]
            assert f.attrs["Ly"] == 2
            assert f.attrs["J"] == J
            assert f.attrs["g"] == g
            assert f.attrs["t"] == t
            Vlist[j] = f["Vlist"][:]
            indices[j] = f["indices"][:]
            assert Vlist[j].shape[0] == n
            err_opt[j] = f["err_iter"][-1]

    # approximation error of optimized circuits for larger system sizes
    circ_err = np.zeros((len(Lxlist), len(nlayers)))
    for i, Lx in enumerate(Lxlist):
        for j, n in enumerate(nlayers):
            perms = [perm_set[Lx][k] for k in indices[j]]
            circ_err[i, j] = np.linalg.norm(oc.brickwall_unitary(Vlist[j], 2*Lx, perms) - expiH[Lx], ord=2)

    print("error computation consistency check:", np.linalg.norm(err_opt - circ_err[0], np.inf))

    return circ_err


def main(recompute=True):

    # Hamiltonian parameters
    J = 1
    g = 3

    # various system sizes
    Lxlist = [4, 6]
    Ly = 2

    # time
    t = 0.25

    # number of circuit layers
    nlayers = [5, 9, 13, 17]

    if recompute:
        circ_err = compute_circuit_errors(J, g, Lxlist, t, nlayers)
        # save errors to disk
        with h5py.File("ising_ladder_dynamics_approx_larger_systems.hdf5", "w") as f:
            f.create_dataset("circ_err", data=circ_err)
            # store parameters
            f.attrs["J"] = float(J)
            f.attrs["g"] = float(g)
            f.attrs["t"] = float(t)
            f.attrs["Lxlist"] = Lxlist
            f.attrs["Ly"] = Ly
            f.attrs["nlayers"] = nlayers
    else:
        # load errors from disk
        with h5py.File("ising_ladder_dynamics_approx_larger_systems.hdf5", "r") as f:
            # parameters must agree
            assert f.attrs["J"] == J
            assert f.attrs["g"] == g
            assert f.attrs["t"] == t
            assert np.array_equal(f.attrs["Lxlist"], Lxlist)
            assert f.attrs["Ly"] == Ly
            assert np.array_equal(f.attrs["nlayers"], nlayers)
            circ_err = f["circ_err"][:]
    print(circ_err)

    # define plot colors
    clr_base = mc.to_rgb(plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
    clrs = len(Lxlist)*[None]
    for i in range(len(Lxlist)):
        s = i / (len(Lxlist) - 1)
        clrs[i] = ((1 - s)*clr_base[0], (1 - s)*clr_base[1], (1 - s)*clr_base[2])

    for i, Lx in enumerate(Lxlist):
        plt.loglog(nlayers, circ_err[i], '.-', color=clrs[i], label=f"Lx = {Lx}")
    xt = [5, 6, 9, 13, 17]
    plt.xticks(xt, [rf"$\mathdefault{{{l}}}$" if l % 2 == 1 else "" for l in xt])
    plt.xlabel("number of layers")
    plt.ylabel("error")
    plt.legend(loc="upper right")
    plt.title(rf"$\mathrm{{approximating }}\ e^{{-i H^{{\mathrm{{Ising}}}} t}} \ \mathrm{{for}} \ L_y = {Ly}, J = {J}, g = {g}, t = {t}$")
    plt.savefig("ising_ladder_dynamics_approx_larger_systems.pdf")
    plt.show()


if __name__ == "__main__":
    main()
