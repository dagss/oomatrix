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

    def diagonal(self):
        return self.array.diagonal().copy()

class Strided(MatrixImpl, NumPyWrapper):
    name = 'strided'





#
# Implement all operations using NumPy with a C-contiguous target.
#
# In due course, have our own LAPACK wrapper to do the efficient
# thing in each case.
#

@computation(Strided.i, Strided,
             cost=lambda self: self.ncols**3 * FLOP)
def inverse(self):
    return Strided(np.linalg.inv(self.array))    
    
@computation(Strided + Strided, Strided,
             cost=lambda a, b: a.ncols * a.nrows * FLOP)
def add(a, b):
    # Ensure result will be C-contiguous with any NumPy
    out = np.zeros(a.array.shape, order='C')
    np.add(a.array, b.array, out)
    return Strided(out)


@computation(Strided * Strided, Strided, 'numpy.dot',
             cost=lambda a, b: a.nrows * b.ncols * a.ncols * FLOP)
def multiply(a, b):
    out = np.dot(a.array, b.array)
    return Strided(out)

@computation(Strided.h + Strided, Strided,
             cost=lambda a, b: a.nrows * a.ncols * FLOP)
def add(a, b):
    a_arr = a.array.T
    if issubclass(a_arr.dtype.type, np.complex):
        a_arr = a_arr.conjugate()
    # Ensure result will be C-contiguous with any NumPy
    return Strided(a_arr + b.array)

@computation(Strided.h * Strided, Strided, '(np.conjugate and) np.dot',
             cost=lambda a, b: a.nrows * b.ncols * a.ncols * FLOP)
def multiply(a, b):
    a_arr = a.array.T
    if issubclass(a_arr.dtype.type, np.complex):
        a_arr = a_arr.conjugate()
    out = np.dot(a_arr, b.array)
    return Strided(out)

#
# Transpose
#
@computation(Strided.h, Strided,
             cost=lambda node: node.ncols * node.nrows * MEMOP)
def ch_to_c(self):
    assert self.dtype != np.complex128 and self.dtype != np.complex64
    return Strided(self.array.T.copy('F'))
