import numpy as np

class NonadiabaticMatrix(object):
    def __init__(self, matrix):
        self.n_electronic_states = len(matrix)
        dct = {}
        for i in range(self.n_electronic_states):
            row = matrix[i]
            assert (len(row)==self.n_electronic_states), \
                    "Input matrix not square"
            for j in range(self.n_electronic_states):
                elem = row[j]
                if elem != 0.0:
                    dct[(i,j)] = elem

        self.matrix = matrix
        self.dictionary = dct
        self.check_entries(dct.values())
        self.set_runnable_entries()


    @staticmethod
    def check_entries(list_of_entries):
        pass # TODO: this should verify types

    def keys(self):
        """Keys, assuming we have an upper triangular matrix"""
        return [(i,j) for (i,j) in self.dictionary.keys() if i<=j]

    def set_runnable_entries(self):
        # TODO: better name for this function? sets the "mask" of the
        # _numeric_matrix
        n_elec = self.n_electronic_states
        self._numeric_matrix = np.zeros((n_elec, n_elec))
        self.runnable_entries = {}
        for key in self.dictionary.keys():
            val = self.dictionary[key]
            if not np.isscalar(val):
                self.runnable_entries[key] = val
            else:
                self._numeric_matrix[key] = val


    @classmethod
    def from_dictionary(cls, dct, n_electronic_states=None):
        from itertools import chain
        res = cls.__new__(cls)
        cls.check_entries(dct.values())
        if n_electronic_states is None:
            n_electronic_states = max(list(chain.from_iterable(dct.keys())))+1
        res.n_electronic_states = n_electronic_states
        res.dictionary = dct
        matrix = []
        for i in range(res.n_electronic_states):
            row_i = [0.0]*res.n_electronic_states
            for j in range(res.n_electronic_states):
                try:
                    row_i[j] = dct[(i,j)]
                except KeyError:
                    pass
            matrix.append(row_i)
                
        res.matrix = matrix
        res.set_runnable_entries()
        return res


    def __getitem__(self, label):
        return self.dictionary[label]


    def numeric_matrix(self, snap):
        matrix = self._numeric_matrix.copy()
        for key in self.runnable_entries.keys():
            matrix[key] = self.runnable_entries[key](snap)
        return matrix
        
