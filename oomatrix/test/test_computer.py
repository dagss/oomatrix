from .common import *
from .. import Matrix, Vector, compute, explain
from ..computer import StupidComputer
from ..core import ConversionGraph, AdditionGraph, MatrixImpl, MultiplyPairGraph

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
        return '<Mock1:%s>' % self.value

class Mock2(Mock1):
    name = 'Mock2'
    def __repr__(self):
        return '<Mock2:%s>' % self.value

# Can only multiply (1 1) and (1 2)
@mock_multiply_operation((Mock1, Mock1), Mock1)
def mul(a, b):
    assert a.ncols == b.nrows
    return Mock1('(%s%s)' % (a.value, b.value), a.nrows, b.ncols)

@mock_multiply_operation((Mock1, Mock2), Mock2)
def mul(a, b):
    assert a.ncols == b.nrows
    return Mock2('(%s%s)' % (a.value, b.value), a.nrows, b.ncols)

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

    yield ok_, type(A * u) is Matrix
    yield eq_, '<Mock2:(Au)>', do(A * u)
    yield eq_, '<Mock2:(A(Au))>', do(A * A * u)
    yield eq_, '<Mock2:(A(A(Au)))>', do(A * A * A * u)
    yield eq_, '<Mock2:(((AA)A)B)>', do(A * A * A * B)


def test_stupid_computer_numpy():
    De_array = np.arange(9).reshape(3, 3).astype(np.int64)
    De = Matrix(De_array, 'De')
    Di_array = np.arange(3).astype(np.int64)
    Di = Matrix(Di_array, 'Di', diagonal=True)
    # The very basic computation
    computer = StupidComputer()
    co = computer.compute
    a = ndrange((3, 1))
    yield ok_, type(De * a) is Matrix
    yield arrayeq_, co(De * a).as_array(), np.dot(De_array, a)
    

#    yield ok_, np.all(co(De * (Di + Di) * a) == np.dot(De_array, Di_array * a + Di_array * a))
