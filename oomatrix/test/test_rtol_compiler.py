
from .common import *
from .. import Matrix, compute, explain, symbolic

from ..kind import MatrixImpl, MatrixKind
from ..computation import (computation, conversion, ImpossibleOperationError,
                           FLOP, UGLY)
from ..compiler import ShortestPathCompiler
from .. import compiler, formatter, metadata, transforms, task

from .mock_universe import (MockKind, MockMatricesUniverse, check_compilation,
                            create_mock_matrices)


def assert_compile(expected_task_graph, matrix):
    c = compiler.RightToLeftCompiler()
    check_compilation(c, expected_task_graph, matrix)


def test_basic():
    ctx, (A, a) = create_mock_matrices('A')
    # Can only compile expressions on the form (expr * vector)
    ctx.define(A * A, A)
    assert_compile('T0 = multiply_A_A(a, a)', a * a)
    with assert_raises(compiler.ImpossibleOperationError):
        assert_compile('', a + a)

def test_multiply():
    ctx, (A, a), (B, b) = create_mock_matrices('A B')
    ctx.define(A * B, B)
    assert_compile('T2 = multiply_A_B(a, b); T1 = multiply_A_B(a, T2); '
                   'T0 = multiply_A_B(a, T1)', a * a * a * b)
    with assert_raises(compiler.ImpossibleOperationError):
        assert_compile('', a * a * a)
    
def test_distributive():
    ctx, (A, a), (B, b) = create_mock_matrices('A B')
    ctx.define(A * B, B)
    assert_compile('T1 = multiply_A_B(a, b); T2 = multiply_A_B(a, b); '
                   'T0 = add_B_B(T1, T2)', (a + a) * b)
    assert_compile('T2 = multiply_A_B(a, b); T1 = multiply_A_B(a, T2); '
                   'T3 = multiply_A_B(a, T2); T0 = add_B_B(T1, T3)',
                   (a + a) * a * b)

def test_transpose():
    ctx, (A, a), (B, b) = create_mock_matrices('A B')
    ctx.define(A.h * B, B)
    assert_compile('T0 = multiply_Ah_B(a, b)', a.h * b)

def test_transpose_convert():
    ctx, (A, a), (B, b) = create_mock_matrices('A B')
    ctx.define(A * B, B)
    ctx.define(A.h, A)
    assert_compile('T1 = Ah(a); T0 = multiply_A_B(T1, b)', a.h * b)

def test_multi_vectors():
    ctx, (A, a), (B, b) = create_mock_matrices('A B')
    ctx.define(A * B, B)
    assert_compile('T2 = multiply_A_B(a, b); '
                   'T3 = multiply_A_B(a, b); '
                   'T1 = add_B_B(T2, T3); '
                   'T0 = multiply_A_B(a, T1)', a * (a * b + a * b))
