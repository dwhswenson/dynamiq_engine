import dynamiq_engine as dynq
import numpy as np
from tools import *
import stubs
from dynamiq_engine.dynamiq_engine_core import *

class testTopology(object):
    def setup(self):
        pes = stubs.PotentialStub()
        masses = np.array([2.0])
        self.topology = Topology(masses, pes)

    def test_inverse_masses(self):
        assert_array_almost_equal(self.topology.inverse_masses, 
                                  np.array([0.5]))


class testSnapshot(object):
    def setup(self):
        self.topology = Topology(
            masses=np.array([2.0]),
            potential=stubs.PotentialStub()
        )
        self.snap = Snapshot(
            coordinates=np.array([1.0]),
            momenta=np.array([1.0]),
            topology=self.topology
        )

    def test_velocities(self):
        assert_array_almost_equal(self.snap.velocities, np.array([0.5]))

    def test_xyz(self):
        assert_array_almost_equal(self.snap.coordinates, self.snap.xyz)


class testDynamiqEngine(object):
    def setup(self):
        from example_systems import ho_2_1
        self.topology = ho_2_1.topology
        self.potential = ho_2_1.potential
        self.integrator = ho_2_1.integrator
        self.snap = Snapshot(
            coordinates=np.array([0.0]),
            momenta=np.array([0.0]),
            topology=self.topology
        )
        self.engine = DynamiqEngine(potential=self.potential, 
                                    integrator=self.integrator,
                                    template=self.snap)

    def test_current_snapshot(self):
        snap = Snapshot(
            coordinates=np.array([1.0]),
            momenta=np.array([1.0]),
            topology=self.topology
        )
        self.engine.current_snapshot = snap
        loaded = self.engine.current_snapshot

        assert_array_almost_equal(snap.coordinates, loaded.coordinates)
        assert_array_almost_equal(snap.momenta, loaded.momenta)

    def test_generate_next_frame(self):
        snap = Snapshot(
            coordinates=np.array([1.0]),
            momenta=np.array([1.0]),
            topology=self.topology
        )
        self.engine.current_snapshot = snap
        self.engine.start()
        newsnap = self.engine.generate_next_frame()
        ho = exact_ho(time=self.integrator.dt, omega=2.0, m=0.5, x0=1.0,
                      p0=1.0, q0=1.0)
        assert_almost_equal(ho['q'], newsnap.coordinates)
        assert_almost_equal(ho['p'], newsnap.momenta)
        assert_true(newsnap is not snap)
        assert_almost_equal(newsnap.coordinates, 
                            self.engine.current_snapshot.coordinates)
        assert_almost_equal(newsnap.momenta,
                            self.engine.current_snapshot.momenta)

    def test_generate(self):
        import openpathsampling as paths
        ensemble = paths.LengthEnsemble(10)
        snap = Snapshot(
            coordinates=np.array([1.0]),
            momenta=np.array([1.0]),
            topology=self.topology
        )
        self.engine.current_snapshot = snap
        self.engine.start()
        traj = self.engine.generate(snap, running=[ensemble.can_append])
        assert_equal(len(traj), 10)
        # don't bother will all-to-all checks, but make sure the first snap
        # is not the same as any other and that the last snap is not the
        # same as any other
        snap0 = traj[0]
        snapN = traj[-1]
        assert_true(snap0 is not snapN)
        snap0_coord = snap0.coordinates.tolist()
        snapN_coord = snapN.coordinates.tolist()
        snap0_momenta = snap0.momenta.tolist()
        snapN_momenta = snapN.momenta.tolist()

        assert_not_equal(snap0_coord, snapN_coord)
        assert_not_equal(snap0_momenta, snapN_momenta)
        for snap in traj[1:-1]:
            assert_true(snap0 is not snap)
            assert_true(snapN is not snap)
            assert_not_equal(snap0_coord, snap.coordinates.tolist())
            assert_not_equal(snapN_coord, snap.coordinates.tolist())
            assert_not_equal(snap0_momenta, snap.momenta.tolist())
            assert_not_equal(snapN_momenta, snap.momenta.tolist())

