import numpy as np
from ..kind import MatrixImpl
from ..computation import computation, conversion
from ..cost import FLOP, MEM, MEMOP

class ScaledIdentity(MatrixImpl):
    """Matrix with a constant value on the diagonal"""

    def __init__(self, value, n):
        self.value = value
        self.ncols = self.nrows = self.n = n
        self.dtype = np.float64

    def square_root(self, value):
        return np.sqrt(value)

    def diagonal(self):
        result = np.ones(self.n)
        if self.value != 1:
            result *= self.value
        return result

    factor = cholesky = square_root


@computation(ScaledIdentity + ScaledIdentity, ScaledIdentity)
def id_plus_id(a, b):
    return ScaledIdentity(a.value + b.value, a.n)

@computation(ScaledIdentity * ScaledIdentity, ScaledIdentity)
def id_times_id(a, b):
    return ScaledIdentity(a.value * b.value, a.n)

@computation(ScaledIdentity.h, ScaledIdentity)
def id_h(a):
    return ScaledIdentity(a.value.conjugate(), a.n)

@computation(ScaledIdentity.i, ScaledIdentity)
def id_i(a):
    return ScaledIdentity(1 / a.value, a.n)
