import numpy as np

#from ..core import conversion, addition, multiplication
from ..kind import MatrixImpl
from ..computation import computation, conversion, FLOP, MEMOP

def array_conjugate(x):
    if x.dtype.kind == 'c':
        return x.T.conjugate()
    else:
        return x.T

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

class ColumnMajor(MatrixImpl, NumPyWrapper):
    name = 'column-major'

class RowMajor(MatrixImpl, NumPyWrapper):
    name = 'row-major'

class Strided(MatrixImpl, NumPyWrapper):
    name = 'strided'

class SymmetricContiguous(MatrixImpl, NumPyWrapper):
    """
    Matrices that are symmetric and contiguous, and stored in the full
    format, are contiguous in both column-major and row-major format.
    """
    name = 'symmetric contiguous'
    prose = ('the symmetrix contiguous', 'a symmetric contiguous')

    @conversion(ColumnMajor, cost=0)
    def to_column_major(self):
        return ColumnMajor(self.array)

    @conversion(RowMajor, cost=0)
    def to_row_major(self):
        return RowMajor(self.array)





#
# Implement all operations using NumPy with a C-contiguous target.
#
# In due course, have our own LAPACK wrapper to do the efficient
# thing in each case.
#

for T in [ColumnMajor, RowMajor, Strided, SymmetricContiguous]:
    
    @computation(T + T, RowMajor,
                 cost=lambda a, b: a.ncols * a.nrows * FLOP)
    def add(a, b):
        # Ensure result will be C-contiguous with any NumPy
        out = np.zeros(A.shape, order='C')
        np.add(a.array, b.array, out)
        return RowMajor(out)


    @computation(T * T, RowMajor, 'numpy.dot',
                 cost=lambda a, b: a.nrows * b.ncols * a.ncols * FLOP)
    def multiply(a, b):
        out = np.dot(a.array, b.array)
        if not out.flags.c_contiguous:
            raise NotImplementedError('numpy.dot returned non-C-contiguous array')
        return RowMajor(out)

    #
    # Then for the conjugate-transpose versions
    #
    @computation(T.h + T, RowMajor)
    def add(a, b):
        a_arr = a.array.T
        if issubclass(a_arr.dtype.type, np.complex):
            a_arr = a_arr.conjugate()
        # Ensure result will be C-contiguous with any NumPy
        out = np.zeros(a.array.shape, order='C')
        np.add(a_arr, b.array, out)
        return RowMajor(out)

    @computation(T.h * T, RowMajor, '(np.conjugate and) np.dot',
                 cost=lambda a, b: a.nrows * b.ncols * a.ncols * FLOP)
    def multiply(a, b):
        a_arr = a.array.T
        if issubclass(a_arr.dtype.type, np.complex):
            a_arr = a_arr.conjugate()
        out = np.dot(a_arr, b.array)
        if not out.flags.c_contiguous:
            raise NotImplementedError('numpy.dot returned non-C-contiguous array')
        return RowMajor(out)


#
# Transpose
#
@computation(ColumnMajor.h, ColumnMajor,
             cost=lambda node: node.ncols * node.nrows * MEMOP)
def ch_to_c(self):
    assert self.dtype != np.complex128 and self.dtype != np.complex64
    return ColumnMajor(self.array.T.copy('F'))

@computation(RowMajor.h, RowMajor,
             cost=lambda node: node.ncols * node.nrows * MEMOP)
def rh_to_r(self):
    assert self.dtype != np.complex128 and self.dtype != np.complex64
    return ColumnMajor(self.array.T.copy('C'))
