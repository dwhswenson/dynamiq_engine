from nose.tools import (
    assert_equal, assert_not_equal, assert_almost_equal, assert_items_equal,
    raises
)
from nose.plugins.skip import Skip, SkipTest

from numpy.testing import assert_array_almost_equal
import numpy as np

def check_function(function, dictionary):
    """Test a single-variable function based on key-value pairs.

    Takes `function` and calls it with the keys of `dictionary`. Asserts
    that the result should be "almost_equal" to the value of `dictionary`.
    """
    for test_input in dictionary.keys():
        assert_almost_equal(function(test_input), dictionary[test_input])

def exact_ho(time, omega, m, p0, q0, x0=0.0):
    cos_wt = np.cos(omega*time)
    sin_wt = np.sin(omega*time)
    state_at_t = {
        'q' : np.array([(q0-x0)*cos_wt + p0/m/omega*sin_wt + x0]),
        'p' : np.array([p0*cos_wt - (q0-x0)*m*omega*sin_wt])
    }
    return state_at_t
