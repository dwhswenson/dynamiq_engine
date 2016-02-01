import dynamiq_engine as dynq
import numpy as np
from tools import *
import stubs
from dynamiq_engine.nonadiabatic_matrix import *

class testNonadiabaticMatrix(object):
    def setup(self):
        import dynamiq_engine.potentials as pes
        self.V1 = pes.OneDimensionalInteractionModel(
            pes.interactions.HarmonicOscillatorInteraction(k=2.0, x0=0.0)
        )
        self.V2 = pes.OneDimensionalInteractionModel(
            pes.interactions.HarmonicOscillatorInteraction(k=3.0, x0=1.0)
        )
        self.numbers_matrix = [[1.5, 2.0], [2.0, 0.0]]
        self.numbers_dict = {(0,0) : 1.5, (0,1) : 2.0, (1,0) : 2.0}
        self.mixed_matrix = [[self.V1, 2.0], [2.0, self.V2]]
        self.mixed_dict = {(0,0) : self.V1, (0,1) : 2.0, 
                           (1,0) : 2.0, (1,1) : self.V2}
        self.snap = dynq.Snapshot(
            coordinates=np.array([0.5]),
            momenta=np.array([0.0]),
            # arbitrarily chose V1: has the same (n_atoms, n_spatial) as any
            # other. A bit weird; ideally this will change when JHP changes
            # Snapshots a little.
            topology=dynq.Topology(masses=np.array([1.0]), potential=self.V1)
        )
        # V1(snap) = 0.5*2.0*(0.5-0.0)**2 = 0.25
        # V2(snap) = 0.5*3.0*(0.5-1.0)**2 = 0.375

    def test_build_from_number_matrix(self):
        na = NonadiabaticMatrix(self.numbers_matrix)
        assert_equal(na.dictionary, self.numbers_dict)

    def test_build_from_number_dictionary(self):
        na = NonadiabaticMatrix.from_dictionary(self.numbers_dict)
        assert_equal(na.matrix, self.numbers_matrix)
        na3 = NonadiabaticMatrix.from_dictionary(self.numbers_dict, 3)
        assert_equal(na3.n_electronic_states, 3)
        numbers_3x3 = [[1.5, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        assert_equal(na3.matrix, numbers_3x3)

    def test_build_from_mixed_matrix(self):
        na = NonadiabaticMatrix(self.mixed_matrix)
        assert_equal(na.dictionary, self.mixed_dict)

    def test_build_from_mixed_dictionary(self):
        na = NonadiabaticMatrix.from_dictionary(self.mixed_dict)
        assert_equal(na.matrix, self.mixed_matrix)

    def test_numeric_matrix_number_input(self):
        na = NonadiabaticMatrix(self.numbers_matrix)
        assert_array_almost_equal(na.numeric_matrix(self.snap),
                                  np.array(self.numbers_matrix))

    def test_numeric_matrix_mixed_input(self):
        na = NonadiabaticMatrix(self.mixed_matrix)
        expected = np.array([[0.25, 2.0], [2.0, 0.375]])
        assert_array_almost_equal(na.numeric_matrix(self.snap), expected)

    def test_getitem(self):
        mixed_matrix = NonadiabaticMatrix(self.mixed_matrix)
        mixed_dict = NonadiabaticMatrix.from_dictionary(self.mixed_dict)
        assert_equal(mixed_matrix[1,1], self.V2)
        assert_equal(mixed_dict[1,1], self.V2)

    def bad_entries_string(self):
        raise SkipTest
