from functools import total_ordering
import numpy as np

@total_ordering
class MatrixMetadata(object):
    universe = None # TODO
    
    def __init__(self, kind, rows_shape, cols_shape, dtype):
        self.kind = kind
        self.rows_shape = rows_shape
        self.cols_shape = cols_shape
        self.dtype = dtype
        self.nrows = np.prod(rows_shape)
        self.ncols = np.prod(cols_shape)

    def __repr__(self):
        return '<meta: %s %r-by-%r %s>' % (self.kind,
                                           self.rows_shape, self.cols_shape,
                                           self.dtype)

    def as_tuple(self):
        return (self.kind, self.rows_shape, self.cols_shape, self.dtype)
        
    def __eq__(self, other):
        if not isinstance(other, MatrixMetadata):
            return False
        return self.as_tuple() == other.as_tuple()

    def __lt__(self, other):
        return self.as_tuple() < other.as_tuple()

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.as_tuple())

def meta_add(a, b):
    assert a.rows_shape == b.rows_shape
    assert a.cols_shape == b.cols_shape
    # todo: dtype
    # todo: 
    return MatrixMetadata(a.rows_shape, a.cols_shape, a.dtype)

def meta_mul(a, b):
    assert a.cols_shape == b.rows_shape
    #TODO dtype
    return MatrixMetadata(a.rows_shape, b.cols_shape, a.dype)
