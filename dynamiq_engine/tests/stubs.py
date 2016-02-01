import dynamiq_engine as dynq
import numpy as np

class PotentialStub(dynq.potentials.PotentialEnergySurface):
    def __init__(self, n_atoms=1, n_spatial=1):
        self.n_atoms = n_atoms
        self.n_spatial = n_spatial

    def H(self, snapshot):
        return 0.0

    def dHdq(self, snapshot):
        return np.array([[0.0]*self.n_spatial]*self.n_atoms)

    def dHdp(self, snapshot):
        return np.array([[0.0]*self.n_spatial]*self.n_atoms)
