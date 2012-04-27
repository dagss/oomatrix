from .common import *
from .. import Matrix, compute, explain, symbolic

from ..kind import MatrixImpl, MatrixKind
from ..computation import (computation, conversion, ImpossibleOperationError,
                           FLOP, UGLY)
from ..compiler import ExhaustiveCompiler
from .. import compiler

from .mock_universe import MockKind, MockMatricesUniverse


def test_outer():
    lst = list(compiler.outer([1,2,3], [4,5,6], [7,8]))
    assert lst == [(1, 4, 7), (1, 4, 8), (1, 5, 7), (1, 5, 8),
                   (1, 6, 7), (1, 6, 8), (2, 4, 7), (2, 4, 8),
                   (2, 5, 7), (2, 5, 8), (2, 6, 7), (2, 6, 8),
                   (3, 4, 7), (3, 4, 8), (3, 5, 7), (3, 5, 8),
                   (3, 6, 7), (3, 6, 8)]
    

def test_basic():
    pass
