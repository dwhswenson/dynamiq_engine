"""
Attributes
----------
Mqq : numpy.ndarray, shape=(ndim, ndim), dtype=numpy.float32
    derivative dq_t / dq_0
Mqp : numpy.ndarray, shape=(ndim, ndim), dtype=numpy.float32
    derivative dq_t / dp_0
Mpq : numpy.ndarray, shape=(ndim, ndim), dtype=numpy.float32
    derivative dp_t / dq_0
Mpp : numpy.ndarray, shape=(ndim, ndim), dtype=numpy.float32
    derivative dp_t / dp_0
"""

variables = ['Mqq', 'Mqp', 'Mpq', 'Mpp']
numpy = ['Mqq', 'Mqp', 'Mpq', 'Mpp']

# TODO: add netcdfplus_init

