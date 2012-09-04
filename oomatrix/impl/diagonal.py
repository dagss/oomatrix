import numpy as np

from ..kind import MatrixImpl
from ..computation import computation, conversion, FLOP, MEMOP
from .dense import Strided

__all__ = ['Diagonal']

class Diagonal(MatrixImpl):

    
    name = 'diagonal'
    prose = ('the diagonal matrix', 'a diagonal matrix')

    def __init__(self, array):
        array = np.asarray(array)
        if array.ndim != 1:
            raise ValueError("Please pass a one-dimensional array")
        self.ncols = self.nrows = array.shape[0]
        self.array = array
        self.dtype = array.dtype

    def as_dtype(self, dtype):
        return Diagonal(self.array.astype(dtype))

    @conversion(Strided,
                cost=lambda node: node.ncols * node.nrows * MEMOP)
    def diagonal_to_dense(D):
        n = D.ncols
        i = np.arange(n)
        out = np.zeros((n, n), dtype=D.dtype)
        out[i, i] = D.array
        return Strided(out)

    def get_element(self, i, j):
        if i != j:
            return 0
        else:
            return self.array[i]

    def apply(self, vec, out, should_accumulate):
        result = (vec.T * self.array).T
        if should_accumulate:
            out += result
        else:
            out[...] = result

    #def square_root(self):
    #    return Diagonal(np.sqrt(self.array))

    def diagonal(self):
        return self.array

    #factor = cholesky = square_root

@computation(Diagonal.f, Diagonal, cost=lambda self: self.nrows * FLOP)
def square_root(self):
    return Diagonal(np.sqrt(self.array))

@computation(Diagonal.i, Diagonal, cost=lambda self: self.nrows * FLOP)
def inverse(self):
    return Diagonal(1 / self.array)

@computation(Diagonal.h, Diagonal, cost=0)
def conjugate_transpose(a):
    return Diagonal(a.array.conjugate())

@computation(Diagonal + Diagonal, Diagonal,
             cost=lambda a, b: a.ncols * FLOP)
def diagonal_plus_diagonal(a, b):
    return Diagonal(a.array + b.array)

@computation(Diagonal * Diagonal, Diagonal,
             cost=lambda a, b: a.ncols * FLOP)
def diagonal_times_diagonal(a, b):
    return Diagonal(a.array * b.array)

for T in [Strided]:
    @computation(Diagonal + T, T,
                 cost=lambda a, b: a.ncols * FLOP + a.ncols * a.nrows * MEMOP)
    def diagonal_plus_dense(diagonal, dense):
        array = dense.array.copy('F' if T is ColumnMajor else 'C')
        i = np.arange(array.shape[0])
        array[i, i] = diagonal.array
        return T(array)

# Optimized diagonal-times-dense
for T in [Strided]:
    @computation(Diagonal * T, T,
                 cost=lambda a, b: b.ncols * b.nrows * FLOP)
    def diagonal_times_dense(a, b):
        out = np.empty_like(b.array)
        np.multiply(a.array[:, None], b.array, out)
        return T(out)

# Optimized dense-times-diagonal
for T in [Strided]:
    @computation(T * Diagonal, T,
                 cost=lambda a, b: a.ncols * a.nrows * FLOP)
    def dense_times_diagonal(a, b):
        out = np.empty_like(a.array)
        np.multiply(a.array, b.array[None, :], out)
        return T(out)
