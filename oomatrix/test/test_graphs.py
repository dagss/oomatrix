import numpy as np
from nose import SkipTest
from nose.tools import ok_, eq_, assert_raises

from ..core import ConversionGraph, AdditionGraph, MatrixImpl
from ..impl.diagonal import *
from ..impl.dense import *

from .. import Matrix

mock_conversion_graph = ConversionGraph()
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
        name = X
    mock_kinds.append(MockImpl)

    @mock_add_operation((MockImpl, MockImpl), MockImpl)
    def f(a, b):
        return type(a)(a.value + b.value)

A, B, C, D = mock_kinds

# TODO: Mock full AdditionGraph

def test_addition_get_vertices():
    V = list(mock_addition_graph.get_vertices(3, [A, B, C, D]))
    V0 = [[A], [B], [C], [D],
          [A, B], [A, C], [A, D], [B, C], [B, D], [C, D],
          [A, B, C], [A, B, D], [A, C, D], [B, C, D]]
    V0 = [set(x) for x in V0]
    assert len(V0) == len(V)
    for v in V:
        assert v in V0

def test_add_perform():
    a = A(2)
    b = B(1)
    
    r = mock_addition_graph.perform([a, a])
    yield ok_, type(r) is A
    yield eq_, 4, r.value

    #r = mock_addition_graph.perform([a, b])
    #print Matrix('R', r)

    # TODO...
    
