# This module includes all the classes necessary to make the dynq engine
# compatible with OPS, etc.
import openpathsampling as paths
import numpy as np
import dynamiq_engine.features as features

from openpathsampling.netcdfplus import lazy_loading_attributes

@lazy_loading_attributes('_reversed')
class Snapshot(paths.AbstractSnapshot):
    __features__ = [paths.features.coordinates, features.momenta]
    def __init__(self, coordinates=None, momenta=None, monodromy=None, 
                 is_reversed=False, topology=None, reversed_copy=None):
        """
        Creates a dynq.Snapshot

        By default, the monodromy matrices and actions associated with each
        snapshot are not saved. 
        """
        self.coordinates = coordinates
        self.momenta = momenta
        self.topology = topology
        self.is_reversed = is_reversed
        # set monodromy matrices and action value
        if monodromy is not None:
            self._Mqq = monodromy[0]
            self._Mqp = monodromy[1]
            self._Mpq = monodromy[2]
            self._Mpp = monodromy[3]

    @property
    def velocities(self):
        return self.momenta * self.topology.inverse_masses

    @property
    def xyz(self):
        return self.coordinates

    def copy(self):
        new_snap = Snapshot(
            coordinates=self.coordinates.copy(), 
            momenta=self.momenta.copy(), 
            is_reversed=self.is_reversed,
            topology=self.topology
        )
        return new_snap

    def detach_monodromy(self):
        """Removes links to monodromy matrices.

        Useful if the already-allocated monodromy matrices will be reused
        for a later trajectory.
        """
        self._Mqq = None
        self._Mqp = None
        self._Mpq = None
        self._Mpp = None

    def copy_from(self, other):
        np.copyto(self.coordinates, other.coordinates)
        np.copyto(self.momenta, other.momenta)
        # TODO: monodromy
        self.topology = other.topology
        self.is_reversed = other.is_reversed

@lazy_loading_attributes('_reversed')
class MMSTSnapshot(Snapshot):
    __features__ = [
        paths.features.coordinates,
        features.momenta,
        features.electronic_coordinates,
        features.electronic_momenta
    ]
    def __init__(self, coordinates=None, momenta=None, monodromy=None,
                 electronic_coordinates=None, electronic_momenta=None,
                 is_reversed=False, topology=None, reversed_copy=None):
        super(MMSTSnapshot, self).__init__(
            coordinates=coordinates, momenta=momenta, monodromy=monodromy,
            is_reversed=is_reversed, topology=topology,
            reversed_copy=reversed_copy
        )
        self.electronic_coordinates = electronic_coordinates
        self.electronic_momenta = electronic_momenta


class Topology(paths.Topology):
    def __init__(self, masses, potential):
        n_atoms = potential.n_atoms
        n_spatial = potential.n_spatial
        super(Topology, self).__init__(n_atoms, n_spatial)
        self.masses = masses
        self._inverse_masses = np.array([1.0/m for m in masses])
        self.potential = potential

    @property
    def inverse_masses(self):
        return self._inverse_masses

    def subset(self, list_of_atoms):
        return self # pragma: no cover

class DynamiqEngine(paths.DynamicsEngine):
    default_options = {
        'integ' : None,
        'n_frames_max' : None,
        'nsteps_per_frame' : 1
    }

    base_snapshot_type = Snapshot

    def __init__(self, potential, integrator, template):
        self.potential = potential
        self.integrator = integrator

    @property
    def current_snapshot(self):
        pass

    @current_snapshot.setter
    def current_snapshot(self, snap):
        pass

    def generate_next_frame(self):
        self.integrator.step(self, self.nsteps_per_frame)
        return self.current_snapshot

    def generate_n_frames(self, n):
        pass
