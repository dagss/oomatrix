from nose.tools import ok_, eq_, assert_raises
from numpy.testing import assert_almost_equal
import numpy as np

from oomatrix import DenseMatrix

def ndrange(shape, dtype=np.double):
    return np.arange(np.prod(shape), dtype=dtype).reshape(shape)

array = ndrange((10, 10))

def test_errors():
    A = DenseMatrix(array, "A")
    def func():
        A + 2
    yield assert_raises, NotImplementedError, func

def test_repr():
    A = DenseMatrix(array, "A")
    B = DenseMatrix(array, "B")

    yield eq_, '10x10 matrix: A', repr(A)
    
    yield eq_, '10x10 matrix: A + B', repr(A + B)
    yield eq_, '10x10 matrix: A + A + A + B', repr(A + A + A + B)
    yield eq_, '10x10 matrix: A * A * A * B', repr(A * A * A * B)
    yield eq_, '10x10 matrix: (A * A) + (A * B)', repr(A * A + A * B)
    yield eq_, '10x10 matrix: ((A * A) + A) * B', repr((A * A + A) * B)

def test_matvec():
    A = DenseMatrix(array, "A")
    B = DenseMatrix(array, "B")
    u = ndrange((10, 1))
    v = ndrange((10, 2, 3, 4))

    yield ok_, np.all(np.dot(array, u) == A * u)
    yield ok_, np.all(np.dot(array, np.dot(array, u)) == A * A * u)
    yield ok_, np.all(np.dot((array + array), u) == (A + A) * u)
    #yield eq_, (5, 2, 3, 4), (DenseMatrix(ndrange((5, 10)), 'tmp') * v).shape
