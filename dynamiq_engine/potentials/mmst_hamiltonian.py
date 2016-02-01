import dynamiq_engine as dynq
from dynamiq_engine.potentials import PotentialEnergySurface
import numpy as np

class MMSTHamiltonian(PotentialEnergySurface):
    """Meyer-Miller-Stock-Thoss mapped electron Hamiltonian.

    Parameters
    ----------
    H_matrix : dynamiq_engine.NonadiabaticMatrix
        The input Hamiltonian matrix. 
    """

    def __init__(self, H_matrix, electronic_first=True):
        self.H_matrix = H_matrix
        self.n_electronic_states = H_matrix.n_electronic_states
        self.electronic_first = electronic_first

        runnables = self.H_matrix.runnable_entries.values()
        try:
            runnable_0 = runnables[0]
        except IndexError:
            # can this work?
            self.n_spatial = 0
            self.n_atoms = 0
        else:
            self.n_spatial = runnables[0].n_spatial
            self.n_atoms = runnables[0].n_atoms

        err_str = " not the same in all nonadiabatic matrix entries."
        for runnable in runnables:
            assert runnable.n_spatial == self.n_spatial, "n_spatial" + err_str
            assert runnable.n_atoms == self.n_atoms, "n_atoms" + err_str


    def _elect_cache(self, snap):
        """Returns a cache of the electronic parts of the Hamiltonian.

        This cache is such that `elect[(i,i)]` is $0.5(x_i^2 + p_i^2 - 1)$
        and `elect[(i,j)]` (j>i) is $x_i x_j + p_i p_j$, with $x$ and $p$
        being the MMST electronic variables.

        This is how the electronic variables appear and the MMST
        Hamiltonian, and these values are reused for the derivatives of the
        potential. Hence, a reason to cache them.
        """
        elect = {}
        for i in range(self.n_electronic_states):
            x_i = snap.electronic_coordinates[i]
            p_i = snap.electronic_momenta[i]
            elect[(i,i)] = 0.5*(x_i*x_i + p_i*p_i - 1.0)
            for j in range(i+1, self.n_electronic_states):
                x_j = snap.electronic_coordinates[j]
                p_j = snap.electronic_momenta[j]
                elect[(i,j)] = x_i*x_j + p_i*p_j
        return elect
 
    def V(self, snapshot):
        """For the MMST Hamiltonian, this is V_{eff}, the effective potential.

        We continue to define the kinetic energy as the nuclear kinetic
        energy. V_{eff} is everything else.
        """
        elect = self._elect_cache(snapshot)
        V_ij = self.H_matrix.numeric_matrix(snapshot)
       
        V = sum([elect[key] * V_ij[key] for key in self.H_matrix.keys()])
        return V

    
    def set_electronic_dHdq(self, electronic_dHdq, snapshot):
        V_ij = self.H_matrix.numeric_matrix(snapshot)

        for i in range(self.n_electronic_states):
            electronic_dHdq[i] = sum(
                [snapshot.electronic_coordinates[j] * V_ij[(i,j)]
                 for j in range(self.n_electronic_states)]
            )

    def set_dHdq(self, dHdq, snapshot):
        elect = self._elect_cache(snapshot)
        dHdq.fill(0.0)
        self._part_dHdq = np.zeros_like(dHdq)
        runnable_keys = [(i,j) 
                         for (i,j) in self.H_matrix.runnable_entries.keys() 
                         if i<=j] # upper triangular version

        for key in runnable_keys:
            self.H_matrix.runnable_entries[key].set_dHdq(self._part_dHdq,
                                                         snapshot)
            np.add(self._part_dHdq * elect[key], dHdq, dHdq)


    def set_electronic_dHdp(self, electronic_dHdp, snapshot):
        V_ij = self.H_matrix.numeric_matrix(snapshot)

        for i in range(self.n_electronic_states):
            electronic_dHdp[i] = sum(
                [snapshot.electronic_momenta[j] * V_ij[(i,j)]
                 for j in range(self.n_electronic_states)]
            )

    def electronic_dHdp(self, snapshot):
        e_dHdp = np.zeros(self.n_electronic_states)
        self.set_electronic_dHdp(e_dHdp, snapshot)
        return e_dHdp

    def electronic_dHdq(self, snapshot):
        e_dHdq = np.zeros(self.n_electronic_states)
        self.set_electronic_dHdq(e_dHdq, snapshot)
        return e_dHdq

    # dHdp (for nuclear only) is still the same

    # following are to be done later
    def L(self, snap):
        pass

    def d2Hdq2(self, snap):
        pass

    def d2Hdp2(self, snap):
        pass

    def d2Hdqdp(self, snap):
        pass

    def d2Hdpdq(self, snap):
        pass

