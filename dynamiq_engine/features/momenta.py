"""
Attributes
----------
momenta : numpy.ndarray, shape=(ndim,), dtype=numpy.float32
    momenta of the system
"""


variables = ['momenta']
numpy = ['momenta']

def netcdfplus_init(store):
    store.create_variable(
        'momenta', 'numpy.float32', 
        dimensions=('ndim',),
        description="the momentum associated with the given degree of freedom",
        chunksizes=(1, 'ndim')
    )


@property
def velocities(snapshot):
    """
    Returns
    -------
    velocites : numpy.ndarray
        velocities
    """
    return snapshot.momenta * snapshot.topology.inverse_masses
