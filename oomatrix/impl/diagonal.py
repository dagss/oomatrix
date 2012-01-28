import numpy as np

from ..kind import MatrixImpl
from ..computation import computation, conversion
from ..cost import FLOP, MEM, MEMOP
from .dense import SymmetricContiguous

__all__ = ['Diagonal']

class Diagonal(MatrixImpl):

    """


    Example
    -------

    >>> D = Matrix('D', [1, 2, 3], diagonal=True)
    >>> D
    3-by-3 diagonal matrix 'D' of int64
    [0 0 0]
    [0 1 0]
    [0 0 2]
    >>> (D + D).compute()
    3-by-3 diagonal matrix 'D' of int64
    [0 0 0]
    [0 2 0]
    [0 0 4]

    
    """

    
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

@computation(Diagonal + Diagonal, Diagonal)
def diagonal_plus_diagonal(A, B):
    return Diagonal(A.array + B.array)

