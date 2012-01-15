from .common import *
from .. import Matrix, compute, explain
from ..compiler import SimplisticCompiler
from ..core import (ConversionGraph, AdditionGraph, MatrixImpl, MultiplyPairGraph,
                    ImpossibleOperationError, credits)

def arrayeq_(x, y):
    assert np.all(x == y)

mock_conversion_graph = ConversionGraph()
mock_conversion = mock_conversion_graph.conversion_decorator
mock_addition_graph = AdditionGraph(mock_conversion_graph)
mock_multiply_graph = MultiplyPairGraph(mock_conversion_graph)
mock_multiplication = mock_multiply_graph.multiplication_decorator
mock_addition = mock_addition_graph.addition_decorator

class Mock1(MatrixImpl):
    name = 'Mock1'
    def __init__(self, value, nrows, ncols):
        self.value = value
        self.nrows = nrows
        self.ncols = ncols

    def __repr__(self):
        return '<%s:%s>' % (type(self).name, self.value)

def conjugate_repr(self):
    return '<%s:%s.h>' % (type(self).name, self.wrapped.value)
    
Mock1.H.__repr__ = conjugate_repr

class Mock2(Mock1):
    name = 'Mock2'
Mock2.H.__repr__ = conjugate_repr

class Mock3(Mock1):
    name = 'Mock3'
Mock3.H.__repr__ = conjugate_repr

@mock_multiplication((Mock1, Mock1), Mock1)
def applemul(a, b):
    assert a.ncols == b.nrows
    return Mock1('(%s %s)' % (a.value, b.value), a.nrows, b.ncols)

@mock_multiplication((Mock1, Mock1.H), Mock1)
def orangemul(a, b):
    assert a.ncols == b.nrows
    return Mock1('(%s %s.h)' % (a.value, b.wrapped.value), a.nrows, b.ncols)

@credits('libfairtrade', 'N. Roozen & F. van der Hoof (1988)')
@mock_multiplication((Mock1, Mock2), Mock2)
def bananamul(a, b):
    assert a.ncols == b.nrows
    return Mock2('(%s %s)' % (a.value, b.value), a.nrows, b.ncols)

@mock_multiplication((Mock1.H, Mock2), Mock2)
def figmul(a, b):
    assert a.ncols == b.nrows
    return Mock2('(%s %s.h)' % (a.wrapped.value, b.value), a.nrows, b.ncols)

A = Matrix(Mock1('A', 3, 3))
B = Matrix(Mock2('B', 3, 4))
u = Matrix(Mock2('u', 3, 1))
v = Matrix(Mock2('v', 4, 1))

#u = Vector(np.arange(3))

#
# SimplisticCompiler
#

## '<Mock2:(A u)>', '''\
##         A * u is computed in one step, "bananamul(A, u)", using [1]

##         [1] libfairtrade, N. Roozen & F. van der Hoof (1988)''',

def test_stupid_compiler_mock():
    compiler = SimplisticCompiler(multiply_graph=mock_multiply_graph)
    co = compiler.compute
#    ex = compiler.explain
    def test(expected, M, target_kind=None):
        eq_(expected, repr(co(M)._expr.matrix_impl))

    # A is 3-by-3 Mock1, B is 3-by-4 Mock2, u is 3-by-1 Mock2

    #
    # Multiplication
    #

    yield ok_, type(A * u) is Matrix
    yield test, '<Mock2:(A u)>', A * u

    yield test, '<Mock2:(A (A u))>', A * A * u
    yield test, '<Mock2:(A (A (A u)))>', A * A * A * u
    yield test, '<Mock2:(((A A) A) B)>', A * A * A * B

    #
    # Mul + conjugation
    #

    # Straightforward conjugate
    yield test, '<Mock1:(A A.h)>', A * A.h
    # Need to conjugate parent expression
    yield test, '<conjugate transpose Mock1:(A A).h>', A.h * A.h
    # A.h * A is not implemented
    yield assert_raises, ImpossibleOperationError, co, A.h * A
    # But A.h * B is (need to conjugate parent)
    yield test, '<conjugate transpose Mock2:(A B.h).h>', B.h * A
    yield test, '<conjugate transpose Mock1:(A x).h>', Matrix(Mock1('x', 3, 1)).h * A.h

    #
    # Post-multiply conversion
    #
    

def test_stupid_compiler_numpy():
    De_array = np.arange(9).reshape(3, 3).astype(np.int64) + 1j
    De = Matrix(De_array, 'De')
    Di_array = np.arange(3).astype(np.int64)
    Di = Matrix(Di_array, 'Di', diagonal=True)
    # The very basic computation
    compiler = SimplisticCompiler()
    co = compiler.compute
    a = ndrange((3, 1))
    yield ok_, type(De * a) is Matrix
    yield arrayeq_, co(De * a).as_array(), np.dot(De_array, a)
    yield arrayeq_, co(De.h * a).as_array(), np.dot(De_array.T.conjugate(), a)
