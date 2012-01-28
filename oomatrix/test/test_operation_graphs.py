from .common import *

from ..kind import MatrixImpl
from ..computation import *
from ..impl.diagonal import *
from ..impl.dense import *
from ..operation_graphs import *

from .. import Matrix, actions

mock_kinds = []
for X in 'ABCD':
    class MockImpl(MatrixImpl):
        "A 5x5 matrix of a single value in all elements"
        nrows = ncols = 5
        def __init__(self, value):
            self.value = value
        def get_element(self, i, j):
            return self.value
        def __eq__(self, other):
            return type(other) is type(self) and other.value == self.value
        def __repr__(self):
            return '%s(%d)' % (self.name, self.value)
        name = X
    mock_kinds.append(MockImpl)

    @computation(MockImpl + MockImpl, MockImpl)
    def add(a, b):
        return type(a)(a.value + b.value)

    if X != 'D':
        # Type D does not have within-type multiplication; only
        # way to get to D result is after-operation conversion
        @computation(MockImpl * MockImpl, MockImpl)
        def multiply(a, b):
            return type(a)(a.value * b.value)

A, B, C, D = mock_kinds

@computation(A * B, C)
def multiply_A_B_to_C(a, b):
    return C(a.value * b.value)

def make_conv(kind_a, kind_b):
    @conversion(kind_a, kind_b)
    def _converter(input):
        return kind_b(input.value)


# Conversions:
#
# A --> B --> C --> D
# ^     |
# +-----+
#

make_conv(A, B)
make_conv(B, C)
make_conv(C, D)
make_conv(B, A)

a = A(1)
b = B(2)
c = C(3)
d = D(4)

def assert_add(expected, lst, target_kinds=None):
    lst = [ComputableLeaf(x) for x in lst]
    action = addition_graph.find_cheapest_action(lst, target_kinds)
    got = action.perform()
    eq_(expected, got)

def assert_mul(expected, lst, target_kinds=None):
    lst = [ComputableLeaf(x) for x in lst]
    action = multiplication_graph.find_cheapest_action(lst, target_kinds)
    got = action.compute()
    eq_(expected, got)

def test_addition_get_vertices():
    V = list(addition_graph.get_vertices(3, [A, B, C, D]))
    V0 = [[A], [B], [C], [D],
          [A, B], [A, C], [A, D], [B, C], [B, D], [C, D],
          [A, B, C], [A, B, D], [A, C, D], [B, C, D]]
    V0 = [frozenset(x) for x in V0]
    assert len(V0) == len(V)
    for v in V:
        assert type(v) is frozenset
        assert v in V0

def test_add_perform_two():

    #plot_add_graph(addition_graph)
    #plot_mul_graph(multiplication_graph)

    yield assert_add, A(2), [a, a]
    yield assert_add, C(4), [a, c]

    yield assert_add, A(3), [a, b], [A]
    yield assert_add, B(3), [a, b], [B]

    # Can result in both a and b, so raises exception
    yield assert_raises, ValueError, assert_add, None, [a, b]


def test_add_perform_many():
    yield assert_add, D(8), [a, c, d], [D]

def test_mul_two():
    
    yield assert_mul, B(4), [b, b]
    yield assert_mul, C(6), [b, c]
    yield assert_mul, C(6), [c, b]
    yield assert_mul, A(2), [a, b], [A]
    yield assert_mul, C(2), [a, b], [C]
    # (A, B) -> C has fast path and is cheaper:
    yield assert_mul, C(2), [a, b], [A, C]
    # But (B, A) -> C does not
    yield assert_mul, A(2), [b, a], [A, C]

    # Post-multiply conversion
    yield assert_mul, D(2), [a, b], [D]
    
