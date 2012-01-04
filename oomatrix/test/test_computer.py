from .common import *
from .. import Matrix, Vector, compute, explain
from ..computer import StupidComputer
from ..core import (ConversionGraph, AdditionGraph, MatrixImpl, MultiplyPairGraph,
                    ImpossibleOperationError)

def arrayeq_(x, y):
    assert np.all(x == y)

mock_conversion_graph = ConversionGraph()
mock_conversion = mock_conversion_graph.conversion
mock_addition_graph = AdditionGraph(mock_conversion_graph)
mock_multiply_graph = MultiplyPairGraph(mock_conversion_graph)
mock_multiply_operation = mock_multiply_graph.multiply_operation
mock_add_operation = mock_addition_graph.add_operation

class Mock1(MatrixImpl):
    name = 'Mock1'
    def __init__(self, value, nrows, ncols):
        self.value = value
        self.nrows = nrows
        self.ncols = ncols

    def __repr__(self):
        return '<%s:%s>' % (type(self).name, self.value)

def _(self):
    return '<%s:%s.h>' % (type(self).name, self.wrapped.value)
    
Mock1.h.__repr__ = _

class Mock2(Mock1):
    name = 'Mock2'

# Can only multiply (1 1) and (1 2)
@mock_multiply_operation((Mock1, Mock1), Mock1)
def mul(a, b):
    assert a.ncols == b.nrows
    return Mock1('(%s %s)' % (a.value, b.value), a.nrows, b.ncols)

@mock_multiply_operation((Mock1, Mock2), Mock2)
def mul(a, b):
    assert a.ncols == b.nrows
    return Mock2('(%s %s)' % (a.value, b.value), a.nrows, b.ncols)

@mock_multiply_operation((Mock1.h, Mock2), Mock2)
def mul(a, b):
    assert a.ncols == b.nrows
    return Mock2('(%s.h %s)' % (a.value, b.value), a.nrows, b.ncols)

@mock_multiply_operation((Mock1, Mock2.h), Mock2)
def mul(a, b):
    assert a.ncols == b.nrows
    return Mock2('(%s %s.h)' % (a.value, b.value), a.nrows, b.ncols)

A = Matrix(Mock1('A', 3, 3))
B = Matrix(Mock2('B', 3, 4))
u = Matrix(Mock2('u', 3, 1))
v = Matrix(Mock2('v', 4, 1))

#u = Vector(np.arange(3))

#
# StupidComputer
#


def test_stupid_computer_mock():
    computer = StupidComputer(multiply_graph=mock_multiply_graph)
    co = computer.compute
    def do(M):
        return repr(co(M)._expr.matrix_impl)

    # A is 3-by-3 Mock1, B is 3-by-4 Mock2, u is 3-by-1 Mock2
    yield ok_, type(A * u) is Matrix
    yield eq_, '<Mock2:(A u)>', do(A * u)
    yield eq_, '<Mock2:(A (A u))>', do(A * A * u)
    yield eq_, '<Mock2:(A (A (A u)))>', do(A * A * A * u)
    yield eq_, '<Mock2:(((A A) A) B)>', do(A * A * A * B)

    yield eq_, '<conjugate transpose Mock1:(A A).h>', do(A.h * A.h)

    # Straightforward conjugate
    yield eq_, '<Mock1:(Ah A)>', do(A.h * A)
    # A * A.h is not implemented directly, so requires reformulation
    yield eq_, '<Mock1:(A Ah)>', do(A * A.h)
    yield eq_, '<Mock2:(Ah (A( Ahu)))>', do(A.h * A * A.h * u)
    yield assert_raises, ImpossibleOperationError, do, u.h * A.h
    yield eq_, '<Mock1:(xhAh)>', do(Matrix(Mock1('x', 3, 1)).h * A.h)


def test_stupid_computer_numpy():
    De_array = np.arange(9).reshape(3, 3).astype(np.int64) + 1j
    De = Matrix(De_array, 'De')
    Di_array = np.arange(3).astype(np.int64)
    Di = Matrix(Di_array, 'Di', diagonal=True)
    # The very basic computation
    computer = StupidComputer()
    co = computer.compute
    a = ndrange((3, 1))
    yield ok_, type(De * a) is Matrix
    yield arrayeq_, co(De * a).as_array(), np.dot(De_array, a)
    yield arrayeq_, co(De.h * a).as_array(), np.dot(De_array.T.conjugate(), a)
