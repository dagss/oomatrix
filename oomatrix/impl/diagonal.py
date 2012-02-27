import numpy as np

from ..kind import MatrixImpl
from ..computation import computation, conversion
from .dense import SymmetricContiguous, ColumnMajor, RowMajor, Strided

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

    @conversion(SymmetricContiguous)
    def diagonal_to_dense(D):
        n = D.ncols
        i = np.arange(n)
        out = np.zeros((n, n), dtype=D.dtype)
        out[i, i] = D.array
        return SymmetricContiguous(out)

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

    def square_root(self):
        return Diagonal(np.sqrt(self.array))

    def diagonal(self):
        return self.array

    factor = cholesky = square_root

@computation(Diagonal.h, Diagonal)
def conjugate_transpose(a):
    return Diagonal(a.array.conjugate())

@computation(Diagonal + Diagonal, Diagonal)
def diagonal_plus_diagonal(a, b):
    return Diagonal(a.array + b.array)

@computation(Diagonal * Diagonal, Diagonal)
def diagonal_times_diagonal(a, b):
    return Diagonal(a.array * b.array)

for T in [ColumnMajor, RowMajor, Strided]:
    @computation(Diagonal + T, T)
    def diagonal_plus_dense(diagonal, dense):
        array = dense.array.copy('F' if T is ColumnMajor else 'C')
        i = np.arange(array.shape[0])
        array[i, i] = diagonal.array
        return T(array)
