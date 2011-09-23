import numpy as np

from ..core import MatrixImpl, add_operation, conversion
from ..cost import FLOP, MEM, MEMOP
from .dense import SymmetricContiguousImpl

__all__ = ['DiagonalImpl']

class DiagonalImpl(MatrixImpl):
    name = 'diagonal'
    prose = ('the diagonal matrix', 'a diagonal matrix')

    def __init__(self, array):
        array = np.asarray(array)
        if array.ndim != 1:
            raise ValueError("Please pass a one-dimensional array")
        self.left_shape = self.right_shape = array.shape
        self._n = array.shape[0]
        self.array = array
        self.dtype = array.dtype

    def as_dtype(self, dtype):
        return DiagonalImpl(self.array.astype(dtype))

    @conversion(SymmetricContiguousImpl)
    def diagonal_to_dense(D):
        n = D._n
        i = np.arange(n)
        out = np.zeros((n, n), dtype=D.dtype)
        out[i, i] = D.array
        return SymmetricContiguousImpl(out)

    def get_element(self, i, j):
        if i != j:
            return 0
        else:
            return self.array[i]

@add_operation((DiagonalImpl, DiagonalImpl), DiagonalImpl)
def diagonal_plus_diagonal(A, B):
    return DiagonalImpl(A.array + B.array)

