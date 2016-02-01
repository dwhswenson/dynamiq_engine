_variables = ['electronic_coordinates']

def _init(store):
    store.create_variable(
        'electronic_coordinates', 'numpy.float32', dimensions('n_electronic',),
        description="the coordinate for a given electronic degree of freedom",
        chunksizes=(1, 'n_electronic')
    )
