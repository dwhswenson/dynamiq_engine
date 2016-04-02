from nose.tools import (
    assert_equal, assert_not_equal, assert_almost_equal, assert_items_equal,
    raises, assert_true
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

def assert_snapshot_almost_equal(snap1, snap2):
    assert_array_almost_equal(snap1.coordinates, snap2.coordinates)
    assert_array_almost_equal(snap1.momenta, snap2.momenta)
    # TODO: proper automated testing for all attributes

# TODO: assert_array_differ; assert_snapshot_differ

def exact_ho(time, omega, m, p0, q0, x0=0.0):
    cos_wt = np.cos(omega*time)
    sin_wt = np.sin(omega*time)
    cos_2wt = np.cos(2*omega*time)
    sin_2wt = np.sin(2*omega*time)
    dq = q0-x0
    state_at_t = {
        'q' : np.array([dq*cos_wt + p0/m/omega*sin_wt + x0]),
        'p' : np.array([p0*cos_wt - dq*m*omega*sin_wt]),
        'L' : (0.5*(p0*p0/m - m*omega*omega*dq*dq)*cos_2wt 
               - omega*p0*dq*sin_2wt),
        'S' : (0.25*(p0*p0/m/omega - m*omega*dq*dq)*sin_2wt
               + 0.5*p0*dq*(cos_2wt-1.0))
    }
    return state_at_t
