import numpy as np

from ..core import MatrixImpl, AddAction, conversion
from ..cost import FLOP, MEM
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
    def to_dense(self):
        n = self._n
        i = np.arange(n)
        out = np.zeros((n, n), dtype=self.array.dtype)
        out[i, i] = self.array
        return SymmetricContiguousImpl(out)


class DiagonalPlusDiagonal(AddAction):
    in_types = [DiagonalImpl] * 2
    out_type = DiagonalImpl
    description = 'Sum diagonal elements of {0} and {1}.'

    @staticmethod
    def perform(A, B):
        return DiagonalImpl(A.array + B.array)

    @staticmethod
    def cost(A, B):
        return A.array.size * (FLOP + MEM)
