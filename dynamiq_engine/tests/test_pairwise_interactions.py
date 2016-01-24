import dynamiq_engine as dynq
from tools import *

from dynamiq_engine.potentials.pairwise_interactions import *

class testConstantInteraction(object):
    def setup(self):
        self.cst = ConstantInteraction(value=3.5)

    def test_f(self):
        tests = {
            0.0 : 3.5,
            1.0 : 3.5,
            0.5 : 3.5
        }
        check_function(self.cst.f, tests)
        check_function(self.cst, tests)

    def test_dfdx(self):
        tests = { 0.0 : 0.0, 1.0 : 0.0, 0.5 : 0.0 }
        check_function(self.cst.dfdx, tests)

    def test_d2fdx2(self):
        tests = { 0.0 : 0.0, 1.0 : 0.0, 0.5 : 0.0 }
        check_function(self.cst.d2fdx2, tests)

class testHarmonicOscillatorInteraction(object):
    def setup(self):
        self.ho = HarmonicOscillatorInteraction(k=2.0, x0=1.0)

    def test_f(self):
        tests = {
            0.0 : 1.0, # = 0.5*2.0*(0.0-1.0)^2
            1.0 : 0.0,
            0.5 : 0.25, # = 0.5*2.0*(0.5-1.0)^2
            2.0 : 1.0
        }
        check_function(self.ho.f, tests)
        check_function(self.ho, tests)

    def test_dfdx(self):
        tests = {
            0.0 : -2.0,
            1.0 : 0.0,
            0.5 : -1.0,
            2.0 : 2.0
        }
        check_function(self.ho.dfdx, tests)

    def test_d2fdx2(self):
        tests = {
            0.0 : 2.0,
            1.0 : 2.0,
            0.5 : 2.0,
            2.0 : 2.0
        }
        check_function(self.ho.d2fdx2, tests)

class testTanhInteraction(object):
    def setup(self):
        self.tanh = TanhInteraction(a=0.75, V0=0.1, R0=0.5)

    def test_f(self):
        tests = {
            0.0 : -0.0358357398350786,
            0.5 : 0.0,
            1.0 : 0.0358357398350786,
            2.0 : 0.0809301070201781
        }
        check_function(self.tanh.f, tests)
        check_function(self.tanh, tests)

    def test_dfdx(self):
        tests = {
            0.0 : 0.0653684981285442,
            0.5 : 0.0750000000000000,
            1.0 : 0.0653684981285442,
            2.0 : 0.0258773833327689
        }
        check_function(self.tanh.dfdx, tests)

    def test_d2fdx2(self):
        tests = {
            0.0 : 0.0351379273851650,
            0.5 : 0.0,
            1.0 : -0.0351379273851650,
            2.0 : -0.0314138910378474
        }
        check_function(self.tanh.d2fdx2, tests)

class testMorseInteraction(object):
    def setup(self):
        self.morse = MorseInteraction(D=30.0, beta=0.08, x0=0.5)

    def test_f(self):
        tests = {
            -1.0 : 0.487663414879601,
            0.0 : 0.0499655787054630,
            0.5 : 0.0,
            5.0 : 2.74198811453729
        }
        check_function(self.morse.f, tests)
        check_function(self.morse, tests)

    def test_dfdx(self):
        tests = {
            -1.0 : -0.690011033961740,
            0.0 : -0.203886208716337,
            0.5 : 0.0,
            5.0 : 1.01243553653309
        }
        check_function(self.morse.dfdx, tests)

    def testd2fdx2(self):
        tests = {
            -1.0 : 0.543360556440359,
            0.0 : 0.432293130684491,
            0.5 : 0.384000000000000,
            5.0 : 0.105918023365982
        }
        check_function(self.morse.d2fdx2, tests)

class testQuarticInteraction(object):
    def setup(self):
        self.quartic = QuarticInteraction(1.5, 1.25, 2.0, 1.0, 0.25, 0.5)

    def test_f(self):
        tests = {
            -1.0 : 6.625,
            0.0 : 0.1875,
            0.5 : 0.25,
            2.0 : 18.0625
        }
        check_function(self.quartic.f, tests)
        check_function(self.quartic, tests)

    def test_dfdx(self):
        tests = {
            -1.0 : -16.8125,
            0.0 : -0.8125,
            0.5 : 1.0,
            2.0 : 35.6875
        }
        check_function(self.quartic.dfdx, tests)

    def test_d2fdx2(self):
        tests = {
            -1.0 : 33.25,
            0.0 : 4.75,
            0.5 : 4.0,
            2.0 : 55.75
        }
        check_function(self.quartic.d2fdx2, tests)

class testGaussianInteraction(object):
    pass

class testLennardJonesInteraction(object):
    pass

class testWCAInteraction(object):
    pass

class testGeneralizedWCAInteraction(object):
    pass
