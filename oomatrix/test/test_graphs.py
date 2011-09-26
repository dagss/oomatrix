import numpy as np
from nose.tools import ok_, eq_, assert_raises

from ..core import addition_conversion_graph
from ..shortest_path import find_shortest_path
from ..impl.diagonal import *

from .. import Matrix

De = Matrix('De', np.arange(9).reshape(3, 3).astype(np.int64))
Di = Matrix('Di', np.arange(3).astype(np.int64), diagonal=True)


def test_shortest():
    print addition_conversion_graph.add_graph
    path = find_shortest_path(addition_conversion_graph.get_edges,
                              (DiagonalImpl, DiagonalImpl),
                              set([DiagonalImpl]))
#    r = addition_conversion_graph.perform((Di._impl, Di._impl))
#    print Matrix('R', r)

#    r = addition_conversion_graph.perform((Di._impl, De._impl))
#    print Matrix('R', r)

    # TODO...
    
