import numpy as np

from .kind import MatrixImpl
from .computation import computation
from .impl.dense import Strided

class Selection(MatrixImpl):
    pass

class RangeSelection(Selection):
    def __init__(self, nrows, ncols, rows_range, cols_range):
        self.nrows = nrows
        self.ncols = ncols
        rows_start, rows_stop = self.rows_range = rows_range
        cols_start, cols_stop = self.cols_range = cols_range
        if rows_stop - rows_start != cols_stop - cols_start:
            raise ValueError()
        if rows_stop > self.nrows or cols_stop > self.ncols:
            raise ValueError()

    def get_element(self, i, j):
        return (1 if (self.rows_range[0] <= i < self.rows_range[1] and
                      self.cols_range[0] <= j < self.cols_range[1] and
                      i == j)
                else 0)

    def transpose(self):
        return RangeSelection(self.ncols, self.nrows,
                              self.cols_range, self.rows_range)

@computation(RangeSelection.h, RangeSelection, cost=0)
def transpose(self):
    return self.transpose()

for kind, order in [(Strided, 'C')]: #, (ColumnMajor, 'F'), (Strided, 'C')]:
    @computation(RangeSelection * kind, kind, cost=0)
    def select_rows(self, dense_matrix):
        rows_start, rows_stop = self.rows_range
        cols_start, cols_stop = self.cols_range
        array = dense_matrix.array
        out = np.zeros((self.nrows,) + array.shape[1:], dtype=array.dtype,
                       order=order)
        out[rows_start:rows_stop, ...] = array[cols_start:cols_stop,...]
        assert out.shape[0] == self.nrows
        assert out.shape[1] == dense_matrix.ncols
        return kind(out)

    @computation(kind * RangeSelection, kind, cost=0)
    def select_columns(dense_matrix, self):
        # TODO: If we supported .t, this could be automatic...
        rows_start, rows_stop = self.rows_range
        cols_start, cols_stop = self.cols_range
        array = dense_matrix.array
        out = np.zeros(array.shape[:1] + (self.ncols,), dtype=array.dtype,
                       order=order)
        #print out[..., cols_start:cols_stop].shape, cols_start, cols_stop
        #print array[..., rows_start:rows_stop].shape, rows_start, rows_stop
        out[..., cols_start:cols_stop] = array[..., rows_start:rows_stop]
        #from pprint import pprint
        #pprint( locals())
        #print dense_matrix.array.shape
        #print out.shape
        #print self.nrows, self.ncols
        assert out.shape[0] == dense_matrix.nrows
        assert out.shape[1] == self.ncols
        #1/0 # transposes too!
        return kind(out)


class GatherMatrix(Selection):
    """
    self * dense gathers the given columns to the output.
    The transpose is a scatter matrix.
    """
    def __init__(self, indices, nrows, ncols):
        self.indices = np.asarray(indices, dtype=int)
        self.nrows = nrows
        self.ncols = ncols
        

@computation(GatherMatrix * Strided, Strided, cost=0) # TODO cost
def gather(self, dense):
    return Strided(dense.array[self.indices, :])

@computation(GatherMatrix.h * Strided, Strided, cost=0) # TODO cost
def scatter(self, dense):
    nrows = self.ncols # transpose
    out = np.zeros((nrows, dense.ncols))
    out[self.indices, :] = dense.array
    return Strided(out)
    
