from .common import *
from .. import Matrix, compute, explain, symbolic

from ..kind import MatrixImpl, MatrixKind
from ..computation import (computation, conversion, ImpossibleOperationError,
                           FLOP, UGLY)
from ..compiler import ExhaustiveCompiler

from .mock_universe import MockKind, MockMatricesUniverse



def test_basic():
    pass
