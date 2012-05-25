import re

from .common import *
from .. import Matrix, compute, explain, symbolic

from ..kind import MatrixImpl, MatrixKind
from ..computation import (computation, conversion, ImpossibleOperationError,
                           FLOP, UGLY)
from ..compiler import ShortestPathCompiler
from .. import compiler, formatter, metadata, transforms, task

from .mock_universe import MockKind, MockMatricesUniverse, check_compilation

def test_outer():
    lst = list(compiler.outer([1,2,3], [4,5,6], [7,8]))
    assert lst == [(1, 4, 7), (1, 4, 8), (1, 5, 7), (1, 5, 8),
                   (1, 6, 7), (1, 6, 8), (2, 4, 7), (2, 4, 8),
                   (2, 5, 7), (2, 5, 8), (2, 6, 7), (2, 6, 8),
                   (3, 4, 7), (3, 4, 8), (3, 5, 7), (3, 5, 8),
                   (3, 6, 7), (3, 6, 8)]

def test_splits():
    # odd number of elements
    lst = list(compiler.set_of_pairwise_nonempty_splits([1, 2, 3]))
    assert lst == [([1,], [2, 3]),
                   ([2,], [1, 3]),
                   ([3,], [1, 2])]
    # even number of elements
    lst = list(compiler.set_of_pairwise_nonempty_splits([1, 2, 3, 4]))
    assert lst == [([1,], [2, 3, 4]),
                   ([2,], [1, 3, 4]),
                   ([3,], [1, 2, 4]),
                   ([4,], [1, 2, 3]),
                   ([1, 2], [3, 4]),
                   ([1, 3], [2, 4]),
                   ([1, 4], [2, 3]),
                   ([2, 3], [1, 4]),
                   ([2, 4], [1, 3]),
                   ([3, 4], [1, 2])]
    # no elements
    assert [] == list(compiler.set_of_pairwise_nonempty_splits([]))


def test_sorted_mixed_list():
    meta_a = metadata.MatrixMetadata(1, None, None, None)
    meta_b = metadata.MatrixMetadata(2, None, None, None)
    task_a = compiler.TaskLeaf(task.Task(None, 0, [], meta_a, None))
    matrix_b = transforms.MatrixMetadataLeaf(0, meta_b)
    assert matrix_b > task_a
    meta_b.kind = -2
    assert matrix_b < task_a

def assert_compile(expected_task_graph, matrix):
    compiler = ShortestPathCompiler()
    check_compilation(compiler, expected_task_graph, matrix)

def test_add():
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A') 
    B, b, bu, buh = ctx.new_matrix('B')
    ctx.define(A + B, A)
    assert_compile('T0 = a + b', a + b)
    assert_compile('T1 = b + b; T0 = a + T1', a + b + b)

def test_multiply():
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A') 
    B, b, bu, buh = ctx.new_matrix('B')
    ctx.define(A * B, A)
    assert_compile('T0 = a * b', a * b)
    with assert_raises(compiler.ImpossibleOperationError):
        assert_compile('T1 = b * b; T0 = a + T1', a * b * b)
    ctx.define(B * B, B)
    assert_compile('T1 = b * b; T0 = a * T1', a * b * b)
    

def test_caching():
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A') 
    B, b, bu, buh = ctx.new_matrix('B')
    ctx.define(A + B, A)

    compiler_obj = compiler.ShortestPathCompiler()
    task0, _ = compiler_obj.compile((a + b)._expr)
    task1, _ = compiler_obj.compile((b + a)._expr) # note the changed order

    assert task0 is task1
    assert len(compiler_obj.cache) == 1

def test_distributive():
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A') 
    B, b, bu, buh = ctx.new_matrix('B')
    C, c, cu, cuh = ctx.new_matrix('C')
    ctx.define(A * B, A)
    ctx.define(A * C, A)
    assert_compile('''
    T2 = a + a;
    T1 = T2 * b;
    T3 = T2 * c;
    T0 = T1 + T3
    ''', (a + a) * (b + c)) # b + c is impossible
    

def test_transpose():
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A') 
    B, b, bu, buh = ctx.new_matrix('B')
    ctx.define(B.h, A)
    ctx.define(A * A, A)
    assert_compile('T1 = b.h; T0 = T1 * a', b.h * a)

def test_factor():
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A') 
    B, b, bu, buh = ctx.new_matrix('B')
    ctx.define(B * A, A)
    ctx.define(B.h * A, A)
    ctx.define(B.f, B)
    ctx.define(B.i, B)
    assert_compile('T0 = b.f', b.f)
    assert_compile('T1 = b + b; T0 = T1.f', (b + b).f)
    assert_compile('T1 = b.f; T0 = T1 * a', b.f * a)
    assert_compile('T2 = b.f; T1 = T2.i; T0 = T1 * a', b.f.i * a)
    assert_compile('T2 = b.i; T1 = T2.f; T0 = T1 * a', b.i.f * a)

def test_inverse():
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A') 
    B, b, bu, buh = ctx.new_matrix('B')
    ctx.define(B * A, A)
    ctx.define(B.i, B)
    ctx.define(B.h, B)
    assert_compile('T0 = b.i', b.i)
    assert_compile('T1 = b.i; T0 = T1 * a', b.i * a)
    assert_compile('T2 = b.i; T1 = T2.h; T0 = T1 * a', b.i.h * a)
    assert_compile('T2 = b.i; T1 = T2.h; T0 = T1 * a', b.h.i * a)

    
    
