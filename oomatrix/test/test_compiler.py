import re

from .common import *
from .. import Matrix, compute, explain, symbolic

from ..kind import MatrixImpl, MatrixKind
from ..computation import (computation, conversion, ImpossibleOperationError,
                           FLOP, UGLY)
from ..compiler import ShortestPathCompiler
from .. import compiler, formatter, metadata, transforms, task

from .mock_universe import (MockKind, MockMatricesUniverse, check_compilation,
                            create_mock_matrices)

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
    task_a = symbolic.TaskLeaf(task.Task(None, 0, [], meta_a, None), ())
    matrix_b = symbolic.MatrixMetadataLeaf(meta_b)
    matrix_b.set_leaf_index(0)
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
    #assert_compile('T0 = add_A_B(a, b)', a + b)
    assert_compile('T1 = add_B_B(b, b); T0 = add_A_B(a, T1)', a + b + b)

def test_add_conversion():
    ctx, (A, a), (B, b) = create_mock_matrices('A B')
    ctx.define(B, A)
    assert_compile('T1 = B(b); T0 = add_A_A(a, T1)', a + b)

def test_multiply():
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A') 
    B, b, bu, buh = ctx.new_matrix('B')
    ctx.define(A * B, A)
    assert_compile('T0 = multiply_A_B(a, b)', a * b)
    with assert_raises(compiler.ImpossibleOperationError):
        assert_compile('', b * b)
    ctx.define(B * B, B)
    assert_compile('T1 = multiply_B_B(b, b); T0 = multiply_A_B(a, T1)', a * b * b)
    C, c, cu, cuh = ctx.new_matrix('B')
    ctx.define(A * C, A)
    assert_compile('T1 = multiply_A_B(a, b); T0 = multiply_A_B(T1, b)',
                   a * b * c)

def test_multiply_conjugation():
    # check that the conjugate is attempted
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A') 
    B, b, bu, buh = ctx.new_matrix('B')
    ctx.define(A.h * B, A)
    ctx.define(B.h, B)
    assert_compile('transposed: T1 = Bh(b); T0 = multiply_Ah_B(a, T1)', b * a)
    

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
    T2 = add_A_A(a, a);
    T1 = multiply_A_B(T2, b);
    T3 = multiply_A_C(T2, c);
    T0 = add_A_A(T1, T3)
    ''', (a + a) * (b + c)) # b + c is impossible

def test_multi_distributive():
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A')
    B, b, bu, buh = ctx.new_matrix('B')
    C, c, cu, cuh = ctx.new_matrix('C')
    ctx.define(B * C, C)
    ctx.define(A * C, C)
    # Force distributive law (there's no A+B)
    assert_compile('''
       T3 = multiply_A_C(a, c);
       T4 = multiply_B_C(b, c);
       T2 = add_C_C(T3, T4);
       T1 = multiply_A_C(a, T2);
       T5 = multiply_B_C(b, T2);
       T0 = add_C_C(T1, T5)''', (a + b) * (a + b) * c)

def test_does_not_reuse_tasks():
    # Currently, matrices are identified by position, not
    #
    # This test is a starting point in case we want to change this behaviour...
    ctx, (A, a), (B, b), (C, c) = create_mock_matrices('A B C')
    ctx.define(B * C, C)
    ctx.define(A * C, C)
    # Force distributive law (there's no A*A)
    assert_compile('''
    T1 = multiply_A_C(a, c);
    T3 = multiply_A_C(a, c);
    T2 = multiply_A_C(a, T3);
    T0 = add_C_C(T1, T2)
    ''', (a + a * a) * c)

def test_transpose():
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A') 
    B, b, bu, buh = ctx.new_matrix('B')
    ctx.define(B.h, A)
    ctx.define(A * A, A)
    assert_compile('T1 = Bh(b); T0 = multiply_A_A(T1, a)', b.h * a)

def test_factor():
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A') 
    B, b, bu, buh = ctx.new_matrix('B')
    ctx.define(B * A, A)
    ctx.define(B.h * A, A)
    ctx.define(B.f, B)
    ctx.define(B.i, B)
    assert_compile('T0 = Bf(b)', b.f)
    assert_compile('T1 = add_B_B(b, b); T0 = Bf(T1)', (b + b).f)
    assert_compile('T1 = Bf(b); T0 = multiply_B_A(T1, a)', b.f * a)
    assert_compile('T2 = Bf(b); T1 = Bi(T2); T0 = multiply_B_A(T1, a)', b.f.i * a)
    assert_compile('T2 = Bi(b); T1 = Bf(T2); T0 = multiply_B_A(T1, a)', b.i.f * a)

def test_inverse():
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A') 
    B, b, bu, buh = ctx.new_matrix('B')
    ctx.define(B * A, A)
    ctx.define(B.i, B)
    ctx.define(B.h, B)
    assert_compile('T0 = Bi(b)', b.i)
    assert_compile('T1 = Bi(b); T0 = multiply_B_A(T1, a)', b.i * a)
    assert_compile('T2 = Bi(b); T1 = Bh(T2); T0 = multiply_B_A(T1, a)', b.i.h * a)
    assert_compile('T2 = Bi(b); T1 = Bh(T2); T0 = multiply_B_A(T1, a)', b.h.i * a)

def test_impossible():        
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A') 
    B, b, bu, buh = ctx.new_matrix('B')
    with assert_raises(ImpossibleOperationError):
        assert_compile('', a + b)
    with assert_raises(ImpossibleOperationError):
        assert_compile('', a * b)

def test_loop():
    # You can do an infinite number of conversions between A and B so
    # the graph-of-trees is infinitely large; check that A + C is
    # still ruled out as impossible
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A') 
    B, b, bu, buh = ctx.new_matrix('B')
    C, c, cu, cuh = ctx.new_matrix('C')
    ctx.define(A, B)
    ctx.define(B, A)
    with assert_raises(ImpossibleOperationError):
        assert_compile('', a + c)


    
