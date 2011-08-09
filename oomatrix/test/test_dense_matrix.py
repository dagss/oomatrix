from nose.tools import ok_, eq_, assert_raises
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

    
    
