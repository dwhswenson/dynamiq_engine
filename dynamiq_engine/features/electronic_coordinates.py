"""
Attributes
----------
electronic_coordinates : numpy.ndarray, shape=(n_electronic,), dtype=numpy.float32
    electronic (MMST) coordinates of the system
"""
variables = ['electronic_coordinates']
numpy = ['electronic_coordinates']


def netcdfplus_init(store):
    store.create_variable(
        'electronic_coordinates', 'numpy.float32', dimensions('n_electronic',),
        description="the coordinate for a given electronic degree of freedom",
        chunksizes=(1, 'n_electronic')
    )
