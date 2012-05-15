import re

from .common import *
from .. import Matrix, compute, explain, symbolic

from ..kind import MatrixImpl, MatrixKind
from ..computation import (computation, conversion, ImpossibleOperationError,
                           FLOP, UGLY)
from ..compiler import ExhaustiveCompiler
from .. import compiler, formatter

from .mock_universe import MockKind, MockMatricesUniverse, check_compilation

def test_outer():
    lst = list(compiler.outer([1,2,3], [4,5,6], [7,8]))
    assert lst == [(1, 4, 7), (1, 4, 8), (1, 5, 7), (1, 5, 8),
                   (1, 6, 7), (1, 6, 8), (2, 4, 7), (2, 4, 8),
                   (2, 5, 7), (2, 5, 8), (2, 6, 7), (2, 6, 8),
                   (3, 4, 7), (3, 4, 8), (3, 5, 7), (3, 5, 8),
                   (3, 6, 7), (3, 6, 8)]


def assert_compile(expected_task_graph, expected_transposed, matrix):
    compiler = ExhaustiveCompiler()
    check_compilation(compiler, expected_task_graph, expected_transposed, matrix)

def test_basic():
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A') 
    B, b, bu, buh = ctx.new_matrix('B')
    ctx.define(A + B, A)
    assert_compile('T0 = a + b', False, a + b)
    assert_compile('T1 = b + b; T0 = a + T1', False, a + b + b)

def test_distributive():
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A') 
    B, b, bu, buh = ctx.new_matrix('B')
    ctx.define(A * B, A)
    ctx.define(B * B, A)
    assert_compile('''
    T2 = b + b;
    T1 = a * T2;
    T3 = b * T2;
    T0 = T1 + T3
    ''', False, (a + b) * (b + b))
    

def test_transpose():
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A') 
    B, b, bu, buh = ctx.new_matrix('B')
    ctx.define(B.h, A)
    ctx.define(A * A, A)
    assert_compile('T1 = b.h; T0 = T1 * a', False, b.h * a)

def test_factor():
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A') 
    B, b, bu, buh = ctx.new_matrix('B')
    ctx.define(B * A, A)
    ctx.define(B.h * A, A)
    ctx.define(B.f, B)
    ctx.define(B.i, B)
    assert_compile('T0 = b.f', False, b.f)
    assert_compile('T1 = b + b; T0 = T1.f', False, (b + b).f)
    assert_compile('T1 = b.f; T0 = T1 * a', False, b.f * a)
    assert_compile('T2 = b.f; T1 = T2.i; T0 = T1 * a', False, b.f.i * a)
    assert_compile('T2 = b.i; T1 = T2.f; T0 = T1 * a', False, b.i.f * a)

def test_inverse():
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A') 
    B, b, bu, buh = ctx.new_matrix('B')
    ctx.define(B * A, A)
    ctx.define(B.i, B)
    ctx.define(B.h, B)
    assert_compile('T0 = b.i', False, b.i)
    assert_compile('T1 = b.i; T0 = T1 * a', False, b.i * a)
    assert_compile('T2 = b.i; T1 = T2.h; T0 = T1 * a', False, b.i.h * a)
    assert_compile('T2 = b.i; T1 = T2.h; T0 = T1 * a', False, b.h.i * a)

    
    
