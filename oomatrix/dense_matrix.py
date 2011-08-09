
class Matrix(object):
    def __add__(self, other):
        if isinstance(other, Matrix):
            return AddedMatrices([self, other])
        else:
            raise NotImplementedError()

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return MultipliedMatrices([self, other])
        else:
            raise NotImplementedError()

    def __repr__(self):
        expr = self.expression_string()
        if expr[0] == '(':
            expr = expr[1:-1]
        return "%dx%d matrix: %s" % (self.shape[0], self.shape[1], expr)

class AddedMatrices(Matrix):
    def __init__(self, matrices):
        self.matrices = matrices
        if len(matrices) == 0:
            raise ValueError()
        # TODO Check that they all have the same shape
        self.shape = matrices[0].shape

    def __add__(self, other):
        if isinstance(other, AddedMatrices):
            return AddedMatrices(self.matrices + other.matrices)
        elif isinstance(other, Matrix):
            return AddedMatrices(self.matrices + [other])
        else:
            return Matrix.__add__(other)

    def expression_string(self):
        return "(%s)" % " + ".join(x.expression_string() for x in self.matrices)


class MultipliedMatrices(Matrix):
    def __init__(self, matrices):
        self.matrices = matrices
        if len(matrices) == 0:
            raise ValueError()
        # TODO Check that they all conform
        self.shape = [matrices[0].shape[0], matrices[-1].shape[1]]

    def __mul__(self, other):
        if isinstance(other, MultipliedMatrices):
            return MultipliedMatrices(self.matrices + other.matrices)
        elif isinstance(other, Matrix):
            return MultipliedMatrices(self.matrices + [other])
        else:
            return Matrix.__add__(other)

    def expression_string(self):
        return "(%s)" % " * ".join(x.expression_string() for x in self.matrices)



class DenseMatrix(Matrix):
    def __init__(self, array, name):
        self.array = array
        self.name = name
        self.shape = array.shape

    def expression_string(self):
        return "%s" % self.name

