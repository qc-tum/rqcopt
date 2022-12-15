import numpy as np
import scipy
import matplotlib.pyplot as plt
import h5py
import qib
import rqcopt as oc


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


def trotterized_time_evolution(L: int, hloc, coeffs, dt: float, nsteps):
    """
    Compute the numeric ODE flow operator of the quantum time evolution
    based on even-odd splitting into commuting local terms,
    with the splitting method specified by coefficients `coeffs`.
    """
    Ulist = [scipy.linalg.expm(-1j*c*dt*hloc) for c in coeffs]
    perms = [None if i % 2 == 0 else np.roll(range(L), -1) for i in range(len(coeffs))]
    V = oc.brickwall_unitary(Ulist, L, perms)
    return np.linalg.matrix_power(V, nsteps)


def main():

    # side length of lattice
    L = 6

    # Hamiltonian parameters
    J = ( 1,    1, -0.5)
    h = ( 0.75, 0,  0)

    # construct Hamiltonian
    latt = qib.lattice.IntegerLattice((L,), pbc=True)
    field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
    H = qib.HeisenbergHamiltonian(field, J, h).as_matrix()
    print("H.shape:", H.shape)
    # visualize spectrum
    λ = np.linalg.eigvalsh(H.todense())
    plt.plot(λ, '.')
    plt.xlabel(r"$j$")
    plt.ylabel(r"$\lambda_j$")
    plt.title(f"Heisenberg Hamiltonian for J = {J},\nh = {h} on a 1D lattice with {L} sites")
    plt.show()

    # reference global unitary
    t = 0.25
    expiH = scipy.linalg.expm(-1j*H.todense()*t)

    # local Hamiltonian term
    hloc = construct_heisenberg_local_term(J, h)

    # real-time evolution via Strang splitting for different time steps
    nsteps_stra = np.array(sorted([2**i for i in range(6)] + [6]))
    err_stra = np.zeros(len(nsteps_stra))
    coeffs_stra = oc.SplittingMethod.suzuki(2, 1).coeffs
    print("coeffs_stra:", coeffs_stra)
    for i, nsteps in enumerate(nsteps_stra):
        print("nsteps:", nsteps)
        dt = t / nsteps
        print("dt:", dt)
        W = trotterized_time_evolution(L, hloc, coeffs_stra, dt, nsteps)
        err_stra[i] = np.linalg.norm(W - expiH, ord=2)
        print(f"err_stra[{i}]: {err_stra[i]}")
    # convergence plot
    dt_list = t / nsteps_stra
    plt.loglog(dt_list, err_stra, '.-', label="Strang")
    plt.loglog(dt_list, 5*np.array(dt_list)**2, '--', label="Δt^2")
    plt.xlabel("Δt")
    plt.ylabel("error")
    plt.legend()
    plt.title(f"real-time evolution up to t = {t} using Strang splitting")
    plt.show()

    # real-time evolution via Suzuki's order-4 splitting for different time steps
    nsteps_suz4 = np.array(sorted([2**i for i in range(3)]))
    err_suz4 = np.zeros(len(nsteps_suz4))
    coeffs_suz4 = oc.SplittingMethod.suzuki(2, 2).coeffs
    print("coeffs_suz4:", coeffs_suz4)
    for i, nsteps in enumerate(nsteps_suz4):
        print("nsteps:", nsteps)
        dt = t / nsteps
        print("dt:", dt)
        W = trotterized_time_evolution(L, hloc, coeffs_suz4, dt, nsteps)
        err_suz4[i] = np.linalg.norm(W - expiH, ord=2)
        print(f"err_suz4[{i}]: {err_suz4[i]}")
    # convergence plot
    dt_list = t / nsteps_suz4
    plt.loglog(dt_list, err_suz4, '.-', label="Suzuki ")
    plt.loglog(dt_list, np.array(dt_list)**4, '--', label="~Δt^4")
    plt.xlabel("Δt")
    plt.ylabel("error")
    plt.legend()
    plt.title(f"real-time evolution up to t = {t} using Suzuki's order-4 splitting")
    plt.show()

    # real-time evolution using Yoshida's minimal order-4 method for different time steps
    nsteps_yosh = np.array([2**i for i in range(4)])
    err_yosh = np.zeros(len(nsteps_yosh))
    coeffs_yosh = oc.SplittingMethod.yoshida4().coeffs
    for i, nsteps in enumerate(nsteps_yosh):
        print("nsteps:", nsteps)
        dt = t / nsteps
        print("dt:", dt)
        W = trotterized_time_evolution(L, hloc, coeffs_yosh, dt, nsteps)
        err_yosh[i] = np.linalg.norm(W - expiH, ord=2)
        print(f"err_yosh[{i}]: {err_yosh[i]}")
    # convergence plot
    dt_list = t / nsteps_yosh
    plt.loglog(dt_list, err_yosh, '.-', label="Yoshida order 4")
    plt.loglog(dt_list, 20*np.array(dt_list)**4, '--', label="Δt^4")
    plt.xlabel("Δt")
    plt.ylabel("error")
    plt.legend()
    plt.title(f"real-time evolution up to t = {t} using Yoshida's method of order-4")
    plt.show()

    # real-time evolution using an order-4 RKN method by Robert I. McLachlan for different time steps
    nsteps_mcla = np.array([2**i for i in range(3)])
    err_mcla = np.zeros(len(nsteps_mcla))
    # recommended method is m = 5; we use the m = 4 method anyway since it requires two substeps less
    coeffs_mcla = oc.SplittingMethod.mclachlan4(4).coeffs
    for i, nsteps in enumerate(nsteps_mcla):
        print("nsteps:", nsteps)
        dt = t / nsteps
        print("dt:", dt)
        W = trotterized_time_evolution(L, hloc, coeffs_mcla, dt, nsteps)
        err_mcla[i] = np.linalg.norm(W - expiH, ord=2)
        print(f"err_mcla[{i}]: {err_mcla[i]}")
    # convergence plot
    dt_list = t / nsteps_mcla
    plt.loglog(dt_list, err_mcla, '.-', label="McLachlan RKN-4")
    plt.loglog(dt_list, 2*np.array(dt_list)**4, '--', label="~Δt^4")
    plt.xlabel("Δt")
    plt.ylabel("error")
    plt.legend()
    plt.title(f"real-time evolution up to t = {t} using McLachlan's RKN-4 method of order 4")
    plt.show()

    # real-time evolution using order-4 PRK method by Blanes and Moan for different time steps
    nsteps_blan = np.array([2**i for i in range(3)])
    err_blan = np.zeros(len(nsteps_blan))
    coeffs_blan = oc.SplittingMethod.blanes_moan().coeffs
    for i, nsteps in enumerate(nsteps_blan):
        print("nsteps:", nsteps)
        dt = t / nsteps
        print("dt:", dt)
        W = trotterized_time_evolution(L, hloc, coeffs_blan, dt, nsteps)
        err_blan[i] = np.linalg.norm(W - expiH, ord=2)
        print(f"err_blan[{i}]: {err_blan[i]}")
    # convergence plot
    dt_list = t / nsteps_blan
    plt.loglog(dt_list, err_blan, '.-', label="Blanes Moan PRK-4")
    plt.loglog(dt_list, 0.1*np.array(dt_list)**4, '--', label="~Δt^4")
    plt.xlabel("Δt")
    plt.ylabel("error")
    plt.legend()
    plt.title(f"real-time evolution up to t = {t} using PRK-4 method by Blanes and Moan")
    plt.show()

    # optimized circuits
    err_iter_opt = {}
    for nlayers in range(3, 19, 2):
        with h5py.File(f"heisenberg1d_dynamics_opt_n{nlayers}.hdf5", "r") as f:
            # parameters must agree
            assert f.attrs["L"] == L
            assert np.array_equal(f.attrs["J"], J)
            assert np.array_equal(f.attrs["h"], h)
            assert f.attrs["t"] == t
            err_iter_opt[nlayers] = f["err_iter"][:]

    # compare in terms of number of layers
    plt.loglog(range(3, 19, 2),
               [err_iter_opt[n][-1] for n in range(3, 19, 2)], 'o-', linewidth=2, label="opt. circuit")
    plt.loglog((len(coeffs_stra)-1)*nsteps_stra + 1, err_stra, '.-', label="Strang")
    plt.loglog((len(coeffs_suz4)-1)*nsteps_suz4 + 1, err_suz4, 'v-', label="Suzuki order 4")
    plt.loglog((len(coeffs_yosh)-1)*nsteps_yosh + 1, err_yosh, '^-', label="Yoshida order 4")
    plt.loglog((len(coeffs_mcla)-1)*nsteps_mcla + 1, err_mcla, '+-', label="McLachlan RKN-4")
    plt.loglog((len(coeffs_blan)-1)*nsteps_blan + 1, err_blan, '*-', label="Blanes Moan PRK-4")
    xt = [3, 5, 7, 9, 13, 25, 49]
    plt.xticks(xt, [rf"$\mathdefault{{{l}}}$" for l in xt])
    plt.xlabel("number of layers")
    plt.ylabel("error")
    plt.legend(loc="upper right")
    plt.title(rf"$\mathrm{{approx }}\ e^{{-i H^{{\mathrm{{Heis}}}} t}} \ \mathrm{{for}} \ L = {L}, J = {J}, h_x = {h[0]}, t = {t}$")
    plt.savefig("heisenberg1d_dynamics_circuit_error.pdf")
    plt.show()


if __name__ == "__main__":
    main()
