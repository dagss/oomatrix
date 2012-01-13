import numpy as np

from ..core import MatrixImpl, conversion, addition, multiplication

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





#
# Implement all operations using NumPy with a C-contiguous target.
#
# In due course, have our own LAPACK wrapper to do the efficient
# thing in each case.
#

for T in [ColumnMajorImpl, RowMajorImpl, StridedImpl, SymmetricContiguousImpl]:
    
    @addition((T, T), RowMajorImpl)
    def add(a, b):
        # Ensure result will be C-contiguous with any NumPy
        out = np.zeros(A.shape, order='C')
        np.add(a.array, b.array, out)
        return RowMajorImpl(out)

    @multiplication((T, T), RowMajorImpl)
    def multiply(a, b):
        out = np.dot(a.array, b.array)
        if not out.flags.c_contiguous:
            raise NotImplementedError('numpy.dot returned non-C-contiguous array')
        return RowMajorImpl(out)

    #
    # Then for the conjugate-transpose versions
    #
    @addition((T.H, T), RowMajorImpl)
    def add(a, b):
        a_arr = a.wrapped.array.T
        if issubclass(a_arr.dtype.type, np.complex):
            a_arr = a_arr.conjugate()
        # Ensure result will be C-contiguous with any NumPy
        out = np.zeros(A.shape, order='C')
        np.add(a_arr, b.array, out)
        return RowMajorImpl(out)

    @multiplication((T.H, T), RowMajorImpl)
    def multiply(a, b):
        a_arr = a.wrapped.array.T
        if issubclass(a_arr.dtype.type, np.complex):
            a_arr = a_arr.conjugate()
        out = np.dot(a_arr, b.array)
        if not out.flags.c_contiguous:
            raise NotImplementedError('numpy.dot returned non-C-contiguous array')
        return RowMajorImpl(out)




## for T in [ColumnMajorImpl, RowMajorImpl, StridedImpl, SymmetricContiguousImpl]:
    
##     class _(MultiplyOperation):
##         library = 'numpy'
        
##         @staticmethod
##         def perform(a, b):
##             pass

##         @staticmethod
##         def get_nnz(a, b):
##             return a.row_count * b.column_count

        
##             def multiply(a, b):
##         a
    
