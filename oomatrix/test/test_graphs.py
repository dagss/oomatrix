import numpy as np
from nose import SkipTest
from nose.tools import ok_, eq_, assert_raises

from ..core import addition_conversion_graph, MatrixImpl, add_operation, conversion
from ..impl.diagonal import *
from ..impl.dense import *

from .. import Matrix

mock_kinds = []

for X in 'ABCD':
    class MockKind(MatrixImpl):
        name = X
    mock_kinds.append(MockKind)

    @add_operation((MockKind, MockKind), MockKind)
    def f(a, b):
        return 

A, B, C, D = mock_kinds

# TODO: Mock full AdditionGraph

def test_get_vertices():
    V = list(addition_conversion_graph.get_vertices(3, [A, B, C, D]))
    V0 = [[A], [B], [C], [D],
          [A, B], [A, C], [A, D], [B, C], [B, D], [C, D],
          [A, B, C], [A, B, D], [A, C, D], [B, C, D]]
    V0 = [set(x) for x in V0]
    assert len(V0) == len(V)
    for v in V:
        assert v in V0

def test_add_perform():
    raise SkipTest()
    a = A()
    b = B()
    
    r = addition_conversion_graph.perform([A, A])
    print Matrix('R', r)

    r = addition_conversion_graph.perform([Di.get_impl(), De.get_impl()])
    print Matrix('R', r)

    # TODO...
    
