from ..core import MatrixRepresentation, AddAction, conversion, conversion_cost

class NumPyWrapper(object):
    # Mix-in for dense matrix represenations

    def __init__(self, array):
        self.array = array
        self.left_shape = (array.shape[0],)
        self.right_shape = (array.shape[1],)
        self.dtype = array.dtype

class ColumnMajorMatrixRepresentation(MatrixRepresentation, NumPyWrapper):
    name = 'column-major'


class RowMajorMatrixRepresentation(MatrixRepresentation, NumPyWrapper):
    name = 'row-major'


class StridedMatrixRepresentation(MatrixRepresentation, NumPyWrapper):
    name = 'strided'


class SymmetricContiguousMatrixRepresentation(MatrixRepresentation, NumPyWrapper):
    """
    Matrices that are symmetric and contiguous, and stored in the full
    format, are contiguous in both column-major and row-major format.
    """
    name = 'symmetric contiguous'
    prose = ('the symmetrix contiguous', 'a symmetric contiguous')

    @conversion(ColumnMajorMatrixRepresentation)
    def to_column_major(self):
        return ColumnMajorMatrixRepresentation(self.array)

    @conversion_cost(ColumnMajorMatrixRepresentation)
    def to_column_major_cost(self):
        return 0

    @conversion(RowMajorMatrixRepresentation)
    def to_row_major(self):
        return RowMajorMatrixRepresentation(self.array)

    @conversion_cost(RowMajorMatrixRepresentation)
    def to_row_major_cost(self):
        return 0

        
