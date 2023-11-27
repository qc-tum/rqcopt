import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import h5py
import qib
import rqcopt as oc


def compute_circuit_errors(J, g, Llist, t, nlayers):
    """
    Compute circuit approximation errors for various system sizes.
    """
    expiH = {}
    for L in Llist:
        # construct Hamiltonian
        latt = qib.lattice.IntegerLattice((L,), pbc=True)
        field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
        H = qib.IsingHamiltonian(field, J, 0., g).as_matrix()
        # reference time evolution operator
        expiH[L] = scipy.linalg.expm(-1j*H.todense()*t)

    # load optimized unitaries from disk
    Vlist = len(nlayers)*[None]
    err_opt = len(nlayers)*[None]
    for j, n in enumerate(nlayers):
        with h5py.File(f"ising1d_dynamics_opt_n{n}.hdf5", "r") as f:
            # parameters must agree
            assert f.attrs["L"] == Llist[0]
            assert f.attrs["J"] == J
            assert f.attrs["g"] == g
            assert f.attrs["t"] == t
            Vlist[j] = f["Vlist"][:]
            assert Vlist[j].shape[0] == n
            err_opt[j] = f["err_iter"][-1]

    # approximation error of optimized circuits for larger system sizes
    circ_err = np.zeros((len(Llist), len(nlayers)))
    for i, L in enumerate(Llist):
        for j, n in enumerate(nlayers):
            perms = [None if i % 2 == 0 else np.roll(range(L), 1) for i in range(n)]
            circ_err[i, j] = np.linalg.norm(oc.brickwall_unitary(Vlist[j], L, perms) - expiH[L], ord=2)

    print("error computation consistency check:", np.linalg.norm(err_opt - circ_err[0], np.inf))

    return circ_err


def main(recompute=True):

    # Hamiltonian parameters
    J = 1
    g = 0.75

    # various system sizes
    Llist = [6, 8, 10, 12]

    # time
    t = 1

    # number of circuit layers
    nlayers = [3, 5, 7, 9]

    if recompute:
        circ_err = compute_circuit_errors(J, g, Llist, t, nlayers)
        # save errors to disk
        with h5py.File("ising1d_dynamics_approx_larger_systems.hdf5", "w") as f:
            f.create_dataset("circ_err", data=circ_err)
            # store parameters
            f.attrs["J"] = float(J)
            f.attrs["g"] = float(g)
            f.attrs["t"] = float(t)
            f.attrs["Llist"] = Llist
            f.attrs["nlayers"] = nlayers
    else:
        # load errors from disk
        with h5py.File("ising1d_dynamics_approx_larger_systems.hdf5", "r") as f:
            # parameters must agree
            assert f.attrs["J"] == J
            assert f.attrs["g"] == g
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
    xt = range(3, 10)
    plt.xticks(xt, [rf"$\mathdefault{{{l}}}$" for l in xt])
    plt.xlabel("number of layers")
    plt.ylabel("error")
    plt.legend(loc="upper right")
    plt.title(rf"$\mathrm{{approximating }}\ e^{{-i H^{{\mathrm{{Ising}}}} t}} \ \mathrm{{for}} \ J = {J}, g = {g}, t = {t}$")
    plt.savefig("ising1d_dynamics_approx_larger_systems.pdf")
    plt.show()


if __name__ == "__main__":
    main()
