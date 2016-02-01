from integrator import Integrator
import numpy as np

import math

class CandyRozmus4(Integrator):
    """Fourth-order integrator by Candy and Rozmus.

    References
    ----------

    """
    def __init__(self, dt, potential, n_frames=1):
        super(CandyRozmus4, self).__init__(dt)
        self._a_k = [
            0.5*(1.0 - 1.0/math.sqrt(3.0))*self.dt,
            1.0/math.sqrt(3.0)*self.dt,
            -1.0/math.sqrt(3.0)*self.dt,
            0.5*(1.0 + 1.0/math.sqrt(3.0))*self.dt,
        ]
        self._b_k = [
            0.0,
            0.5*(0.5 + 1.0/math.sqrt(3.0))*self.dt,
            0.5 * self.dt,
            0.5*(0.5 - 1.0/math.sqrt(3.0))*self.dt
        ]
        n_spatial = potential.n_spatial
        n_atoms = potential.n_atoms
        self.local_dHdq = np.zeros(n_spatial * n_atoms)
        self.local_dHdp = np.zeros(n_spatial * n_atoms)

    def momentum_update(self, potential, snap, k):
        potential.set_dHdq(self.local_dHdq, snap)
        self.local_dHdq *= self._b_k[k]
        np.subtract(snap.momenta, self.local_dHdq, snap.momenta)

    def position_update(self, potential, snap, k):
        potential.set_dHdp(self.local_dHdp, snap)
        self.local_dHdp *=  self._a_k[k]
        np.add(snap.coordinates, self.local_dHdp, snap.coordinates)

    def action_step(self, potential, old_snap, new_snap, k):
        pass

    def step(self, potential, old_snap, new_snap):
        new_snap.copy_from(old_snap)
        for k in range(4):
            self.momentum_update(potential, new_snap, k)
            # TODO: monodromy and action
            self.position_update(potential, new_snap, k)
        # wrap PBCs if necessary

class CandyRozmus4MMST(CandyRozmus4):
    def __init__(self, dt, potential, n_frames=1):
        super(CandyRozmus4MMST, self).__init__(dt, potential, n_frames)
        self.local_electronic_dHdp = np.zeros(potential.n_electronic_states)
        self.local_electronic_dHdq = np.zeros(potential.n_electronic_states)
    
    def position_update(self, potential, snap, k):
        potential.set_dHdp(self.local_dHdp, snap)
        self.local_dHdp *=  self._a_k[k]
        potential.set_electronic_dHdp(self.local_electronic_dHdp, snap)
        self.local_electronic_dHdp *= self._a_k[k]
        np.add(snap.coordinates, self.local_dHdp, snap.coordinates)
        np.add(snap.electronic_coordinates, self.local_electronic_dHdp, 
               snap.electronic_coordinates)

    def momentum_update(self, potential, snap, k):
        potential.set_dHdq(self.local_dHdq, snap)
        self.local_dHdq *= self._b_k[k]
        potential.set_electronic_dHdq(self.local_electronic_dHdq, snap)
        self.local_electronic_dHdq *= self._b_k[k]
        np.subtract(snap.momenta, self.local_dHdq, snap.momenta)
        np.subtract(snap.electronic_momenta, self.local_electronic_dHdq,
                    snap.electronic_momenta)

# to be done later
class CandyRozmus4Monodromy(CandyRozmus4):
    """Fourth-order integrator by Candy and Rozmus, including monodromy
    propagation.

    References
    ----------
    """
    pass

class GeneralizedCandyRozmus4Monodromy(CandyRozmus4Monodromy):
    """Candy-Rozmus integrator including monodromy treatment when
    Hamiltonian is not separable.
    """
    pass
