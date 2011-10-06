import numpy as np
from nose import SkipTest
from nose.tools import ok_, eq_, assert_raises

from ..core import ConversionGraph, AdditionGraph, MatrixImpl
from ..impl.diagonal import *
from ..impl.dense import *

from .. import Matrix

mock_conversion_graph = ConversionGraph()
mock_conversion = mock_conversion_graph.conversion
mock_addition_graph = AdditionGraph(mock_conversion_graph)
mock_add_operation = mock_addition_graph.add_operation

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

    @mock_add_operation((MockImpl, MockImpl), MockImpl)
    def _adder(a, b):
        return type(a)(a.value + b.value)

A, B, C, D = mock_kinds

def make_conv(kind_a, kind_b):
    @mock_conversion(kind_a, kind_b)
    def _converter(input):
        return kind_b(input.value)

make_conv(A, B)
make_conv(B, C)
make_conv(C, D)
make_conv(B, A)

a = A(1)
b = B(2)
c = C(3)
d = D(4)

def plot(max_node_size=4, block=False):
    # Plot the addition graph, for use during debugging
    from ..plot_graphs import plot_graph
    def format_vertex(v):
        names = [kind.name for kind in v]
        names.sort()
        return dict(label=' + '.join(names), color='red' if len(v) == 1 else 'black')
    def format_edge(cost, payload):
        return '%s %.2f' % (payload[0], cost)
    plot_graph(mock_addition_graph,
               max_node_size=max_node_size,
               format_vertex=format_vertex, format_edge=format_edge,
               block=block)

def test_addition_get_vertices():
    V = list(mock_addition_graph.get_vertices(3, [A, B, C, D]))
    V0 = [[A], [B], [C], [D],
          [A, B], [A, C], [A, D], [B, C], [B, D], [C, D],
          [A, B, C], [A, B, D], [A, C, D], [B, C, D]]
    V0 = [frozenset(x) for x in V0]
    assert len(V0) == len(V)
    for v in V:
        assert type(v) is frozenset
        assert v in V0

def test_add_perform_two():
    yield eq_, A(2), mock_addition_graph.perform([a, a])
    yield eq_, C(4), mock_addition_graph.perform([a, c])

    yield eq_, A(3), mock_addition_graph.perform([a, b], [A])
    yield eq_, B(3), mock_addition_graph.perform([a, b], [B])

    # Can result in both a and b, so raises exception
    yield assert_raises, ValueError, mock_addition_graph.perform, [a, b]


def test_add_perform_many():
    yield eq_, D(8), mock_addition_graph.perform([a, c, d], [D])
