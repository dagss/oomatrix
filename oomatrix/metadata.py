import numpy as np


class MatrixMetadata(object):
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

    def __eq__(self, other):
        if not isinstance(other, MatrixMetadata):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other

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
