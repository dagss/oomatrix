from ..core import MatrixImpl, conversion, add_operation

class NumPyWrapper(object):
    # Mix-in for dense matrix represenations

    def __init__(self, array):
        self.array = array
        self.left_shape = (array.shape[0],)
        self.right_shape = (array.shape[1],)
        self.dtype = array.dtype

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

        
