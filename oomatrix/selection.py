import numpy as np

from .kind import MatrixImpl
from .computation import computation
from .impl.dense import RowMajor, ColumnMajor

class Selection(MatrixImpl):
    pass

class RangeSelection(Selection):
    def __init__(self, nrows, ncols, rows_range, cols_range):
        self.nrows = nrows
        self.ncols = ncols
        rows_start, rows_stop = self.rows_range = rows_range
        cols_start, cols_stop = self.cols_range = cols_range

for kind, order in [(RowMajor, 'C'), (ColumnMajor, 'F')]:
    @computation(RangeSelection * kind, kind, cost=0)
    def select_rows(self, dense_matrix):
        rows_start, rows_stop = self.rows_range
        cols_start, cols_stop = self.cols_range
        array = dense_matrix.array
        out = np.zeros((self.nrows,) + array.shape[1:], dtype=array.dtype,
                       order=order)
        out[rows_start:rows_stop, ...] = array[cols_start:cols_stop,...]
        return kind(out)

    @computation(kind * RangeSelection, kind, cost=0)
    def select_columns(dense_matrix, self):
        # TODO: If we supported .t, this could be automatic...
        rows_start, rows_stop = self.rows_range
        cols_start, cols_stop = self.cols_range
        array = dense_matrix.array
        out = np.zeros(array.shape[1:] + (self.ncols,), dtype=array.dtype,
                       order=order)
        out[..., cols_start:cols_stop] = array[..., rows_start:rows_stop]
        return kind(out)
