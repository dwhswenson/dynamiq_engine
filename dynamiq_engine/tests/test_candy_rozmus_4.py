import dynamiq_engine as dynq
import numpy as np
from tools import *
import copy

from dynamiq_engine.integrators.candy_rozmus_4 import *
import dynamiq_engine.potentials as pes

from example_systems import ho_2_1

class testCandyRozmus4(object):
    def setup(self):
        self.potential = ho_2_1.potential
        self.ho = ho_2_1.potential
        self.topology = ho_2_1.topology
        self.integ = CandyRozmus4(0.01, self.potential)
        self.snap0 = dynq.Snapshot(
            coordinates=np.array([1.0]),
            momenta=np.array([0.0]),
            topology=self.topology
        )

    def exact_ho(self, initial_snap, time):
        m = initial_snap.topology.masses[0]
        omega = np.sqrt(self.ho.k / m)
        x0 = self.ho.x0
        p0 = initial_snap.momenta[0]
        q0 = initial_snap.coordinates[0]
        return exact_ho(time, omega, m, p0, q0, x0)

    @raises(RuntimeError)
    def test_prepare_bad_feature_error(self):
        import openpathsampling.engines.features as paths_f
        self.integ.prepare([paths_f.coordinates, paths_f.velocities])
    
    def test_cr4_step(self):
        from dynamiq_engine.features import momenta as f_momenta
        from openpathsampling.engines.features import coordinates as f_coordinates
        self.integ.prepare([f_coordinates, f_momenta])
        new_snap = dynq.Snapshot(coordinates=np.array([0.0]),
                                 momenta=np.array([0.0]),
                                 topology=self.topology)
        # snap0 is at he minimum; it shouldn't move
        self.integ.step(self.potential, self.snap0, new_snap)
        assert_array_almost_equal(self.snap0.coordinates, new_snap.coordinates)
        assert_array_almost_equal(self.snap0.momenta, new_snap.momenta)
        for i in range(10):
            self.integ.step(self.potential, new_snap, new_snap)
        assert_array_almost_equal(self.snap0.coordinates, new_snap.coordinates)
        assert_array_almost_equal(self.snap0.momenta, new_snap.momenta)

        snap1 = dynq.Snapshot(coordinates=np.array([0.0]),
                              momenta=np.array([0.0]),
                              topology=self.topology)
        self.integ.step(self.potential, snap1, new_snap)
        exact_0x01 = self.exact_ho(snap1, 0.01)
        assert_array_almost_equal(new_snap.coordinates, exact_0x01['q'])
        assert_array_almost_equal(new_snap.momenta, exact_0x01['p'])

        for i in range(9):
            self.integ.step(self.potential, new_snap, new_snap)
        exact_0x10 = self.exact_ho(snap1, 0.10)
        assert_array_almost_equal(new_snap.coordinates, exact_0x10['q'])
        assert_array_almost_equal(new_snap.momenta, exact_0x10['p'])

    def test_action(self):
        import openpathsampling.engines.features as paths_f
        import dynamiq_engine.features as dynq_f
        self.integ.prepare([paths_f.coordinates, dynq_f.momenta,
                            dynq_f.action])
        new_snap = dynq.Snapshot(coordinates=np.array([0.0]),
                                 momenta=np.array([1.0]),
                                 topology=self.topology)
        exact = self.exact_ho(new_snap, 0.1)
        self.integ.reset(new_snap)
        for i in range(10):
            self.integ.step(self.potential, new_snap, new_snap)
        # TODO: test action is correct
        assert_almost_equal(new_snap.action, exact['S'])

    def test_monodromy_dt(self):
        raise SkipTest
        import openpathsampling.engines.features as paths_f
        import dynamiq_engine.features as dynq_f
        self.integ.helpers = [dynq.integrators.StandardMonodromy()]
        self.integ.prepare([paths_f.coordinates, dynq_f.momenta,
                            dynq_f.monodromy])
        new_snap = dynq.Snapshot(coordinates=np.array([0.0]),
                                 momenta=np.array([1.0]),
                                 topology=self.topology)
        self.integ.reset(new_snap)

        pass

    def test_cast_snapshot(self):
        new_snap = dynq.Snapshot(coordinates=np.array([0.0]),
                                 momenta=np.array([1.0]),
                                 topology=self.topology)
        self.integ.helpers = [dynq.integrators.StandardMonodromy()]
        self.integ.prepare([paths_f.coordinates, dynq_f.momenta,
                            dynq_f.monodromy])
        cast_snap = self.integ.cast_snapshot(new_snap)
        raise SkipTest


class testCandyRozmus4MMST(object):
    def test_step_uncoupled(self):
        from math import sqrt
        # test uncoupled
        V0 = 2.0
        V1 = 3.0
        uncoupled_matrix = dynq.NonadiabaticMatrix([[V0, 0.0], [0.0, V1]])
        # DEBUG
        #uncoupled_matrix = dynq.NonadiabaticMatrix([[0.0, 0.0], [0.0, 3.0]])
        uncoupled = dynq.potentials.MMSTHamiltonian(uncoupled_matrix)
        uncoupled_topology = dynq.Topology(masses=[], potential=uncoupled)
        uncoupled_snap = dynq.MMSTSnapshot(
            coordinates=np.array([]), momenta=np.array([]),
            electronic_coordinates=np.array([1.0, 1.0]),
            electronic_momenta=np.array([1.0, 1.0]),
            topology=uncoupled_topology
        )
        uncoupled_integ = CandyRozmus4MMST(0.01, uncoupled)
        import dynamiq_engine.features as dynq_f
        import openpathsampling.engines.features as paths_f
        uncoupled_integ.prepare([paths_f.coordinates, dynq_f.momenta,
                                 dynq_f.electronic_coordinates,
                                 dynq_f.electronic_momenta,
                                 dynq_f.action])

        explicit_T = lambda pes, snap : (
            np.dot(snap.momenta, pes.dHdp(snap))
            + np.dot(snap.electronic_momenta, pes.electronic_dHdp(snap))
            #- pes.H(snap) + pes.V(snap)
        )
        uncoupled_integ.reset(uncoupled_snap)
        for i in range(10):
            uncoupled_integ.step(uncoupled, uncoupled_snap, uncoupled_snap)
            t=0.01*(i+1)
            ho1 = exact_ho(time=t, omega=2.0, m=1.0/2.0, q0=1.0, p0=1.0)
            ho2 = exact_ho(time=t, omega=3.0, m=1.0/3.0, q0=1.0, p0=1.0)
            T = uncoupled.T(uncoupled_snap)
            V = uncoupled.V(uncoupled_snap)
            assert_almost_equal(ho1['L'] + ho2['L'] + 0.5*(V0+V1), T-V)
            # TODO: note that there's some question about the correctness of
            # the action here. However, we DO have the correct Lagrangian,
            # and we get the correct action for other HO systems. So it is
            # possible that the problem is a fundamental issue with this
            # integration approach for the action.
            # For future work to test this, the starting point should be
            #print ho1['S']+ho2['S']+0.5*(V0+V1)*t, uncoupled_snap.action

        exact_1 = exact_ho(time=0.1, omega=2.0, m=1.0/2.0, q0=1.0, p0=1.0)
        exact_2 = exact_ho(time=0.1, omega=3.0, m=1.0/3.0, q0=1.0, p0=1.0)
        predicted_coordinates = [exact_1['q'][0], exact_2['q'][0]]
        predicted_momenta = [exact_1['p'][0], exact_2['p'][0]]
        assert_array_almost_equal(uncoupled_snap.electronic_coordinates,
                                  np.array(predicted_coordinates))
        assert_array_almost_equal(uncoupled_snap.electronic_momenta,
                                  np.array(predicted_momenta))
        
        assert_almost_equal(explicit_T(uncoupled, uncoupled_snap),
                            uncoupled.T(uncoupled_snap))


    def test_step_rabi(self):
        # test Rabi
        rabi_matrix = dynq.NonadiabaticMatrix([[2.0, 1.0], [1.0, 3.0]])
        rabi = dynq.potentials.MMSTHamiltonian(rabi_matrix)
        rabi_topology = dynq.Topology(masses=[], potential=rabi)
        rabi_snapshot = dynq.MMSTSnapshot(
            coordinates=np.array([]), momenta=np.array([]),
            electronic_coordinates=np.array([1.0, 0.0]),
            electronic_momenta=np.array([0.0, 1.0]),
            topology=rabi_topology
        )
        rabi_integ = CandyRozmus4MMST(0.01, rabi)
        for i in range(10):
            rabi_integ.step(rabi, rabi_snapshot, rabi_snapshot)

        # NOTE: unverified -- these results are checked against the output
        # that they first gave, not against any analytical results
        assert_array_almost_equal(rabi_snapshot.electronic_coordinates,
                                  np.array([1.07189698, 0.26951517]))
        assert_array_almost_equal(rabi_snapshot.electronic_momenta,
                                  np.array([-0.22220342, 0.85382907]))
        assert_array_almost_equal(rabi_snapshot.coordinates, np.array([]))
        assert_array_almost_equal(rabi_snapshot.momenta, np.array([]))


    def test_step_tully(self):
        # test Tully
        from example_systems import tully as tully_example
        tully = tully_example.potential
        tully_snapshot = copy.deepcopy(tully_example.snapshots[0])

        tully_integ = CandyRozmus4MMST(1.0, tully) # not default dt
        import dynamiq_engine.features as dynq_f
        import openpathsampling.engines.features as paths_f
        tully_integ.prepare([paths_f.coordinates, dynq_f.momenta,
                             dynq_f.electronic_coordinates,
                             dynq_f.electronic_momenta,
                             dynq_f.action])

        tully_integ.step(tully, tully_snapshot, tully_snapshot)

        # NOTE: unverified -- these results are checked against the output
        # that they first gave, not against any analytical results
        assert_not_equal(tully_snapshot.action, 0.0) # TODO: get tested val
        assert_array_almost_equal(tully_snapshot.coordinates,
                                  np.array([0.10959396]))
        assert_array_almost_equal(tully_snapshot.momenta,
                                  np.array([18.9922916]))
        assert_array_almost_equal(tully_snapshot.electronic_coordinates,
                                  np.array([0.70714581, 0.6075074]))
        assert_array_almost_equal(tully_snapshot.electronic_momenta,
                                  np.array([0.15844469, 0.07522877]))
