import dynamiq_engine as dynq
import numpy as np
from tools import *

from dynamiq_engine.integrators.monodromy import *

import dynamiq_engine.potentials as pes
from example_systems import anharmonic_morse, tully
import openpathsampling.engines.features as paths_f
import openpathsampling.engines as peng
import dynamiq_engine.features as dynq_f

MonodromySnapshot = peng.SnapshotFactory(
    name="MonodromySnapshot",
    features=[paths_f.coordinates, dynq_f.momenta, dynq_f.monodromy,
              paths_f.topology]
)

MonodromyMMSTSnapshot = peng.SnapshotFactory(
    name="MonodromySnapshot",
    features=[paths_f.coordinates, dynq_f.momenta, 
              dynq_f.electronic_coordinates, dynq_f.electronic_momenta,
              dynq_f.monodromy, paths_f.topology]
)

class testStandardMonodromy(object):
    def setup(self):
        self.potential = anharmonic_morse.potential
        self.topology = anharmonic_morse.topology
        self.monodromy = StandardMonodromy()
        self.integ = anharmonic_morse.integrator
        self.integ.helpers = [self.monodromy]
        self.integ.prepare([paths_f.coordinates, dynq_f.momenta,
                            dynq_f.monodromy])
        self.monodromy.prepare(self.integ)
        self.snap0 = MonodromySnapshot(
            coordinates=np.array([0.0]),
            momenta=np.array([1.0]),
            topology=self.topology
        )
        # Hpp = [[5.0]]
        # Hqq = [[0.432293130684491]]

        self.fixed_monodromy_1D = (np.array([[2.0]]), np.array([[3.0]]),
                                   np.array([[4.0]]), np.array([[5.0]]))


    def test_prepare(self):
        mono = StandardMonodromy()
        mono.prepare(self.integ)
        assert_equal(mono.second_derivatives.cross_terms, False)
        assert_equal(mono.second_derivatives, self.potential)
        for matrix in [mono._local_dMqq_dt, mono._local_dMqp_dt,
                       mono._local_dMpq_dt, mono._local_dMpp_dt,
                       mono._local_Hpp, mono._local_Hqq]:
            assert_equal(matrix, np.array([[0.0]]))
        for matrix in [mono._local_Hpq, mono._local_Hqp]:
            assert_equal(matrix, None)

    def test_reset(self):
        fresh_snap = MonodromySnapshot(
            coordinates=np.array([1.0]),
            momenta=np.array([1.0]),
            topology=self.topology
        )
        # check that the snapshot has the monodromy feature, although the
        # matrix itself is unset
        assert_equal(fresh_snap.Mqq, None)
        assert_equal(fresh_snap.Mqp, None)
        assert_equal(fresh_snap.Mpq, None)
        assert_equal(fresh_snap.Mpp, None)
        # check that reset sets it correctly
        self.monodromy.reset(fresh_snap)
        assert_equal(fresh_snap.Mqq, np.array([[1.0]]))
        assert_equal(fresh_snap.Mpp, np.array([[1.0]]))
        assert_equal(fresh_snap.Mqp, np.array([[0.0]]))
        assert_equal(fresh_snap.Mpq, np.array([[0.0]]))
        # fix it to something else, then undo it
        (fresh_snap.Mqq, fresh_snap.Mqp,
         fresh_snap.Mpq, fresh_snap.Mpp) = self.fixed_monodromy_1D
        self.monodromy.reset(fresh_snap)
        assert_equal(fresh_snap.Mqq, np.array([[1.0]]))
        assert_equal(fresh_snap.Mpp, np.array([[1.0]]))
        assert_equal(fresh_snap.Mqp, np.array([[0.0]]))
        assert_equal(fresh_snap.Mpq, np.array([[0.0]]))

    def test_dMqq_dt(self):
        # dMqq/dt = Hpq*Mqq + Hpp*Mpq
        self.integ.reset(self.snap0)
        dMqq_dt = self.monodromy.dMqq_dt(self.potential, self.snap0)
        # Hpq = Mpq = 0
        assert_equal(dMqq_dt.tolist(), [[0.0]])
        
        snap = self.snap0
        (snap.Mqq, snap.Mqp, snap.Mpq, snap.Mpp) = self.fixed_monodromy_1D
        dMqq_dt = self.monodromy.dMqq_dt(self.potential, snap)
        #dMqq/dt = Hpp*Mpq = 5.0*4.0 = 20.0
        assert_equal(dMqq_dt.tolist(), [[20.0]])

    def test_dMqp_dt(self):
        # dMqp/dt = Hpq * Mqp + Hpp * Mpp
        #         = Hpp * Mpp
        # first   = 5.0 * 1.0 = 5.0
        # fixed   = 5.0 * 5.0 = 25.0
        self.integ.reset(self.snap0)
        dMqp_dt = self.monodromy.dMqp_dt(self.potential, self.snap0)
        assert_equal(dMqp_dt.tolist(), [[5.0]])

        snap = self.snap0
        (snap.Mqq, snap.Mqp, snap.Mpq, snap.Mpp) = self.fixed_monodromy_1D
        dMqp_dt = self.monodromy.dMqp_dt(self.potential, snap)
        assert_equal(dMqp_dt.tolist(), [[25.0]])

    def test_dMpq_dt(self):
        # dMpq/dt = -Hqq * Mqq - Hqp * Mpq
        #         = -Hqq * Mqq
        # first   = -(0.432293130684491) * 1.0 = -0.432293130684491
        # fixed   = -(0.432293130684491) * 2.0 = -0.864586261368982
        self.integ.reset(self.snap0)
        dMpq_dt = self.monodromy.dMpq_dt(self.potential, self.snap0)
        assert_array_almost_equal(dMpq_dt, np.array([[-0.432293130684491]]))
        
        snap = self.snap0
        (snap.Mqq, snap.Mqp, snap.Mpq, snap.Mpp) = self.fixed_monodromy_1D
        dMpq_dt = self.monodromy.dMpq_dt(self.potential, snap)
        assert_array_almost_equal(dMpq_dt, np.array([[-0.864586261368982]]))

    def test_dMpp_dt(self):
        # dMpp/dt = -Hqq * Mqp - Hqp * Mpp
        #         = -Hqq * Mqp
        # first   = -(0.432293130684491) * 0.0 = 0.0
        # fixed   = -(0.432293130684491) * 3.0 = -1.29687939205347
        self.integ.reset(self.snap0)
        dMpp_dt = self.monodromy.dMpp_dt(self.potential, self.snap0)
        assert_array_almost_equal(dMpp_dt, np.array([[0.0]]))

        snap = self.snap0
        (snap.Mqq, snap.Mqp, snap.Mpq, snap.Mpp) = self.fixed_monodromy_1D
        dMpp_dt = self.monodromy.dMpp_dt(self.potential, snap)
        assert_array_almost_equal(dMpp_dt, np.array([[-1.29687939205347]]))


class testStandardMonodromyMMST(object):
    def setup(self):
        self.monodromy = StandardMonodromy()
        self.potential = tully.potential
        self.topology = tully.topology
        self.integ = tully.integrator
        self.integ.helpers = [self.monodromy]
        self.integ.prepare([paths_f.coordinates, dynq_f.momenta,
                            dynq_f.electronic_coordinates,
                            dynq_f.electronic_momenta,
                            dynq_f.monodromy])
        self.monodromy.prepare(self.integ)
        self.snap0 = MonodromyMMSTSnapshot(
            coordinates=np.array([0.1]),
            momenta=np.array([19.0]),
            electronic_coordinates=np.array([0.7, 0.6]),
            electronic_momenta=np.array([0.2, 0.1]),
            topology=self.topology,
        )
        # Hqq = [[0.01586485,  0.04950249,  0.10324073],
        #        [0.04950249, -0.01586485, -0.10051409],
        #        [0.10324073, -0.10051409, -0.04902564]]
        # Hqp = [[0, 0, 0], [0, 0, 0], [0.03020453, -0.01757739, 0]]
        # Hpq = [[0, 0, 0.03020453], [0, 0, -0.01757739], [0, 0, 0]]
        # Hpp = [[0.01586485, 0.04950249, 0],
        #        [0.04950249, -0.01586485, 0],
        #        [0, 0, 0.00050505]]
        self.fixed_monodromy_3D = (np.array([[1.0, 2.0, 3.0],
                                             [4.0, 5.0, 6.0],
                                             [7.0, 8.0, 9.0]]),
                                   np.array([[1.1, 2.1, 3.1],
                                             [4.1, 5.1, 6.1],
                                             [7.1, 8.1, 9.1]]),
                                   np.array([[1.2, 2.2, 3.2],
                                             [4.2, 5.2, 6.2],
                                             [7.2, 8.2, 9.2]]),
                                   np.array([[1.3, 2.3, 3.3],
                                             [4.3, 5.3, 6.3],
                                             [7.3, 8.3, 9.3]]))

    def test_prepare(self):
        self.monodromy.prepare(self.integ)
        assert_equal(self.monodromy.second_derivatives.cross_terms, True)
        assert_equal(self.monodromy.second_derivatives, self.potential)
        mono = self.monodromy  # to simplify names
        for matrix in [mono._local_dMqq_dt, mono._local_dMqp_dt,
                       mono._local_dMpq_dt, mono._local_dMpp_dt,
                       mono._local_Hpp, mono._local_Hqq,
                       mono._local_Hqp, mono._local_Hpq, mono._tmp]:
            assert_array_almost_equal(matrix, np.array([[0.0, 0.0, 0.0],
                                                        [0.0, 0.0, 0.0],
                                                        [0.0, 0.0, 0.0]]))

    def test_dMqq_dt(self):
        # dMqq/dt = Hpq . Mqq + Hpp . Mpq
        self.integ.reset(self.snap0)
        dMqq_dt = self.monodromy.dMqq_dt(self.potential, self.snap0)
        d2Hdpdq = self.potential.d2Hdpdq(self.snap0)
        # Mpq = 0; Mpp = 1 => dMqq/dt = Hpq 
        assert_array_almost_equal(dMqq_dt, d2Hdpdq)

        # dMqq/dt = np.dot(Hpq, Mqq) + np.dot(Hpp, Mpq)
        #         = np.array([[ 0.43837999,  0.53395186,  0.62952373],
        #                     [-0.13027111, -0.11421086, -0.09815061],
        #                     [ 0.00363636,  0.00414141,  0.00464646]])
        snap = self.snap0
        (snap.Mqq, snap.Mqp, snap.Mpq, snap.Mpp) = self.fixed_monodromy_3D
        dMqq_dt = self.monodromy.dMqq_dt(self.potential, snap)
        assert_array_almost_equal(
            dMqq_dt, 
            np.array([[ 0.43837999,  0.53395186,  0.62952373],
                      [-0.13027111, -0.11421086, -0.09815061],
                      [ 0.00363636,  0.00414141,  0.00464646]])
        )
    
    def test_dMqp_dt(self):
        # dMqp/dt = np.dot(Hpq, Mqp) + np.dot(Hpp, Mpp)
        # first   = Hpp
        # fixed   = np.array([[ 0.44793718,  0.54350904,  0.63908091],
        #                     [-0.12866509, -0.11260484, -0.09654459],
        #                     [ 0.00368687,  0.00419192,  0.00469697]])
        self.integ.reset(self.snap0)
        dMqp_dt = self.monodromy.dMqp_dt(self.potential, self.snap0)
        Hpp = self.potential.d2Hdp2(self.snap0)
        assert_array_almost_equal(dMqp_dt, Hpp)

        snap = self.snap0
        (snap.Mqq, snap.Mqp, snap.Mpq, snap.Mpp) = self.fixed_monodromy_3D
        dMqp_dt = self.monodromy.dMqp_dt(self.potential, snap)
        assert_array_almost_equal(
            dMqp_dt,
            np.array([[ 0.44793718,  0.54350904,  0.63908091],
                      [-0.12866509, -0.11260484, -0.09654459],
                      [ 0.00368687, 0.00419192,  0.00469697]])
        )

    def test_dMpq_dt(self):
        # dMpq/dt = -np.dot(Hqq, Mqq) - np.dot(Hqp, Mpq)
        # first   = -Hqq
        # fixed   = np.array([[-0.93655992, -1.10516799, -1.27377606],
        #                     [ 0.71755554,  0.78443199,  0.85130844],
        #                     [ 0.67957471,  0.71324657,  0.74691843]])
        self.integ.reset(self.snap0)
        dMpq_dt = self.monodromy.dMpq_dt(self.potential, self.snap0)
        Hqq = self.potential.d2Hdq2(self.snap0)
        assert_array_almost_equal(dMpq_dt, -Hqq)

        snap = self.snap0
        (snap.Mqq, snap.Mqp, snap.Mpq, snap.Mpp) = self.fixed_monodromy_3D
        dMpq_dt = self.monodromy.dMpq_dt(self.potential, snap)
        assert_array_almost_equal(
            dMpq_dt,
            np.array([[-0.93655992, -1.10516799, -1.27377606],
                      [ 0.71755554,  0.78443199,  0.85130844],
                      [ 0.67957471,  0.71324657,  0.74691843]])
        )
    
    def test_dMpp_dt(self):
        # dMpp/dt = -np.dot(Hqq, Mqp) - np.dot(Hqp, Mpp)
        # first   = -Hqp
        # fixed   = np.array([[-0.95342073, -1.1220288 , -1.29063687],
        #                     [ 0.72424318,  0.79111964,  0.85799609],
        #                     [ 0.6829419 ,  0.71661376,  0.75028562]])
        self.integ.reset(self.snap0)
        dMpp_dt = self.monodromy.dMpp_dt(self.potential, self.snap0)
        Hqp = self.potential.d2Hdqdp(self.snap0)
        assert_array_almost_equal(dMpp_dt, -Hqp)

        snap = self.snap0
        (snap.Mqq, snap.Mqp, snap.Mpq, snap.Mpp) = self.fixed_monodromy_3D
        dMpp_dt = self.monodromy.dMpp_dt(self.potential, snap)
        assert_array_almost_equal(
            dMpp_dt,
            np.array([[-0.95342073, -1.1220288 , -1.29063687],
                      [ 0.72424318,  0.79111964,  0.85799609],
                      [ 0.6829419 ,  0.71661376,  0.75028562]])
        )
