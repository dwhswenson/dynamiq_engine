"""
Attributes
----------
electronic_momenta : numpy.ndarray, shape=(n_electronic,), dtype=numpy.float32
    electronic (MMST) momenta of the system
"""
variables = ['electronic_momenta']
numpy = ['electronic_momenta']

def netcdfplus_init(store):
    store.create_variable(
        'electronic_momenta', 'numpy.float32', dimensions('n_electronic',),
        description="the momentum for a given electronic degree of freedom",
        chunksizes=(1, 'n_electronic')
    )
