import numpy as np
import h5py
import matplotlib.pyplot as plt


def main():

    # side length of lattice
    L = 6

    # Hamiltonian parameters
    J = 1
    g = 0.75

    # time step
    t = 1

    for nlayers in [3, 5, 7, 9]:

        # optimization iteration progress
        with h5py.File(f"ising1d_dynamics_opt_n{nlayers}.hdf5", "r") as f:
            # parameters must agree
            assert f.attrs["L"] == L
            assert f.attrs["J"] == J
            assert f.attrs["g"] == g
            assert f.attrs["t"] == t
            f_iter = f["f_iter"][:]
            err_iter = f["err_iter"][:]

        fig, ax1 = plt.subplots()

        # visualize optimization progress
        ax1.semilogy(range(len(err_iter)), err_iter)
        ax1.set_xlabel("iteration")
        ax1.set_ylabel("spectral norm error", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.set_title(f"{nlayers} layers, Ising model with L = {L}, J = {J}, g = {g}, t = {t}")
        # rescaled and shifted target function
        ax2 = ax1.twinx()
        ax2.semilogy(range(1, len(f_iter) + 1), 1 + np.array(f_iter) / 2**L, color="tab:orange")
        ax2.set_ylabel(r"$1 + f(G)/2^L$", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        plt.savefig(f"ising1d_dynamics_opt_visualization_n{nlayers}.pdf", bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    main()
