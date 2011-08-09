
class Matrix(object):
    pass

class AddedMatrices(Matrix):
    def __init__(self, matrices):
        self.matrices = matrices
        if len(matrices) == 0:
            raise ValueError()
        # TODO Check that they all have the same shape
        self.shape = matrices[0].shape

    def expression_string(self):
        return " + ".join(x.expression_string() for x in self.matrices)

    def __repr__(self):
        return "%dx%d matrix: %s" % (self.shape[0], self.shape[1],
                                     self.expression_string())

    def __add__(self, other):
        if isinstance(other, Matrix):
            return AddedMatrices([self, other])
        else:
            raise NotImplementedError()

class DenseMatrix(Matrix):
    def __init__(self, array, name):
        self.array = array
        self.name = name
        self.shape = array.shape

    def expression_string(self):
        return self.name

    def __repr__(self):
        return '%dx%d matrix: %s' % (self.array.shape[0],
                                     self.array.shape[1],
                                     self.name)

    def __add__(self, other):
        if isinstance(other, Matrix):
            return AddedMatrices([self, other])
        else:
            raise NotImplementedError()
