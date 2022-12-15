Riemannian quantum circuit optimization examples for quantum lattice models
---------------------------------------------------------------------------

The ``<model>_dynamics_opt.py`` script runs the optimization - note that this can take a while (several hours). The results of these runs are stored in the HDF5 files ``<model>_dynamics_opt_n?.hdf5``, with ``n`` denoting the number of circuit layers. You can directly access the optimized circuit gates from the provided files (``"Vlist"`` datasets) without re-running the optimization.

``<model>_dynamics_circuit.py`` shows a benchmark comparison with existing splitting methods from the literature.

``<model>_dynamics_approx_larger_systems.py`` generates the approximation error plots to demonstrate independence of system size.


The examples use the `qib <https://github.com/qc-tum/qib>`_ Python package to generate matrix representations of the model Hamiltonians. You have to clone and install this package from https://github.com/qc-tum/qib first. Note that the optimization functions of ``rqcopt`` work for any target unitary matrix which is translation invariant.
