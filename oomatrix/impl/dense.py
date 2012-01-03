import numpy as np

from ..core import MatrixImpl, conversion, add_operation, multiply_operation

class NumPyWrapper(object):
    # Mix-in for dense matrix representations

    def __init__(self, array):
        self.array = array
        self.nrows = array.shape[0]
        self.ncols = array.shape[1]
        self.dtype = array.dtype

    def get_element(self, i, j):
        return self.array[i, j]

    def apply(self, vec, out, should_accumulate):
        if should_accumulate:
            out += np.dot(self.array, vec)
        else:
            out[...] = np.dot(self.array, vec)

class ColumnMajorImpl(MatrixImpl, NumPyWrapper):
    name = 'column-major'


class RowMajorImpl(MatrixImpl, NumPyWrapper):
    name = 'row-major'


class StridedImpl(MatrixImpl, NumPyWrapper):
    name = 'strided'


class SymmetricContiguousImpl(MatrixImpl, NumPyWrapper):
    """
    Matrices that are symmetric and contiguous, and stored in the full
    format, are contiguous in both column-major and row-major format.
    """
    name = 'symmetric contiguous'
    prose = ('the symmetrix contiguous', 'a symmetric contiguous')

    @conversion(ColumnMajorImpl)
    def to_column_major(self):
        return ColumnMajorImpl(self.array)

    @conversion(RowMajorImpl)
    def to_row_major(self):
        return RowMajorImpl(self.array)


for T in [ColumnMajorImpl, RowMajorImpl, StridedImpl, SymmetricContiguousImpl]:
    @add_operation((T, T), T)
    def add(A, B):
        return T(A.array + B.array)

    @multiply_operation((T, T), T)
    def multiply(A, B):
        return T(np.dot(A.array, B.array))


