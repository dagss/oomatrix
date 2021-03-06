from functools import total_ordering
import numpy as np
import hashlib
import struct

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
        self._make_hash()

    def _make_hash(self):
        pack = struct.pack
        h = hashlib.sha512()
        h.update(pack('Q', id(self.kind)))
        h.update(str(self.rows_shape))
        h.update(str(self.cols_shape))
        h.update(str(self.dtype))
        self._shash = h.digest()

    def secure_hash(self):
        return self._shash

    def __repr__(self):
        return '<meta: %s %r-by-%r %s>' % (self.kind,
                                           self.rows_shape, self.cols_shape,
                                           self.dtype)

    def as_tuple(self):
        # TODO use something else than str(dtype)
        return (self.kind, self.rows_shape, self.cols_shape, str(self.dtype))

    def copy_with_kind(self, kind):
        return MatrixMetadata(kind, self.rows_shape, self.cols_shape,
                              self.dtype)

    def kindless(self):
        return MatrixMetadata(None, self.rows_shape, self.cols_shape, self.dtype)
        
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

    def transpose(self):
        return MatrixMetadata(self.kind, self.cols_shape, self.rows_shape, self.dtype)

def meta_add(children):
    first = children[0]
    for child in children[1:]:
        assert child.rows_shape == first.rows_shape
        assert child.cols_shape == first.cols_shape
    # todo: dtype
    # todo: 
    return MatrixMetadata(None, first.rows_shape, first.cols_shape, first.dtype)

def meta_multiply(children, target_kind=None):
    for i in range(1, len(children)):
        assert children[i - 1].cols_shape == children[i].rows_shape
    #TODO dtype
    return MatrixMetadata(target_kind,
                          children[0].rows_shape, children[-1].cols_shape,
                          children[0].dtype)

def meta_transpose(m):
    return MatrixMetadata(m.kind, m.cols_shape, m.rows_shape, m.dtype)
