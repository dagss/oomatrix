import numpy as np
from nose.tools import ok_, eq_, assert_raises

from ..core import addition_conversion_graph, MatrixImplType
from ..shortest_path import find_shortest_path
from ..impl.diagonal import *
from ..impl.dense import *

from .. import Matrix





De = Matrix('De', np.arange(9).reshape(3, 3).astype(np.int64))
Di = Matrix('Di', np.arange(3).astype(np.int64), diagonal=True)


def test_get_vertices():
    C, R, S, D = ColumnMajorImpl, RowMajorImpl, SymmetricContiguousImpl, DiagonalImpl
    V = list(addition_conversion_graph.get_vertices(3, [C, R, S, D]))
    V0 = [[C], [R], [S], [D],
          [C, R], [C, S], [C, D], [R, S], [R, D], [S, D],
          [C, R, S], [C, R, D], [C, S, D], [R, S, D]]
    V0 = [set(x) for x in V0]
    assert len(V0) == len(V)
    for v in V:
        assert v in V0

def test_add_perform():
    r = addition_conversion_graph.perform([Di.get_impl(), Di.get_impl()])
    print Matrix('R', r)

    r = addition_conversion_graph.perform([Di.get_impl(), De.get_impl()])
    print Matrix('R', r)

    # TODO...
    
