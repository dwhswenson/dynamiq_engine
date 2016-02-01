_variables = ['electronic_momenta']

def _init(store):
    store.create_variable(
        'electronic_momenta', 'numpy.float32', dimensions('n_electronic',),
        description="the momentum for a given electronic degree of freedom",
        chunksizes=(1, 'n_electronic')
    )
