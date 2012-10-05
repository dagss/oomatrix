from __future__ import division
import re

from .common import *
from .. import Matrix, compute, explain, symbolic

from ..kind import MatrixImpl, MatrixKind
from ..computation import (Computation, computation, conversion, ImpossibleOperationError,
                           FLOP, UGLY, MEMOP)
from .. import compiler, formatter, metadata, transforms, task, cost_value

from .mock_universe import (MockKind, MockMatricesUniverse, check_compilation,
                            create_mock_matrices, cnode_to_str, mock_meta)

import time

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
    c = compiler.GreedyCompiler()
    check_compilation(c, expected_task_graph, matrix)


def test_conversion_cache():
    ctx, (A, a), (B, b), (C, c), (D, d) = create_mock_matrices('A B C D')
    A2B = ctx.define(A, B, cost=2, name='A->B')
    B2C = ctx.define(B, C, cost=1, name='B->C')
    C2D = ctx.define(C, D, cost=2, name='C->D')
    D2B = ctx.define(D, B, cost=1, name='D->B')
    cache = compiler.ConversionCache(mock_cost_map)
    for X in [A, B, C, D]:
        cache.get_conversions_from(mock_meta(X))

    d = dict((meta.kind, dict((meta.kind, lst) for meta, lst in sub_d.iteritems()))
             for meta, sub_d in cache._conversions.iteritems())
    assert d ==  {A: {A: (0, []),
                      B: (2, [A2B]),
                      C: (3, [A2B, B2C]),
                      D: (5, [A2B, B2C, C2D])},
                  B: {B: (0, []),
                      C: (1, [B2C]),
                      D: (3, [B2C, C2D])},
                  C: {B: (3, [C2D, D2B]),
                      C: (0, []),
                      D: (2, [C2D])},
                  D: {B: (1, [D2B]),
                      C: (2, [D2B, B2C]),
                      D: (0, [])}}

def test_addition_cache():
    ctx, (A, a), (B, b), (C, c) = create_mock_matrices('A B C')
    BplusB = ctx.adders[B]
    CplusC = ctx.adders[C]
    AplusB = ctx.define(A + B, A, cost=3)
    AplusC = ctx.define(A + C, A, cost=4)
    A2B = ctx.define(A, B, name='A2B', cost=1)
    B2C = ctx.define(B, C, name='B2C', cost=1)
    add_cache = compiler.AdditionCache(compiler.ConversionCache(mock_cost_map))
    result = add_cache.get_computations([mock_meta(A), mock_meta(B)])
    #pprint(result)
    eq_(result, {A: (3, (0, 1), AplusB, [], []),
                 B: (2, (0, 1), BplusB, [A2B], []),
                 C: (4, (0, 1), CplusC, [A2B, B2C], [B2C])})

def test_fill_in_conversions():
    ctx, (A, a), (B, b), (C, c), (D, d) = create_mock_matrices('A B C D')
    ctx.define(A, B, cost=2, name='A->B')
    ctx.define(B, C, cost=1, name='B->C')
    ctx.define(C, D, cost=2, name='C->D')
    options = [mock_task(A, 200), mock_task(C, 1), mock_task(D, 100), mock_task(A, 2)]
    full_options = compiler.fill_in_conversions(options, mock_cost_map)
    expected = [(C, 1), (A, 2), (D, 3), (B, 4)]
    got = [(opt.metadata.kind, opt.get_total_cost().weigh(mock_cost_map)) for opt in full_options]
    assert expected == got
    

def test_find_cheapest_addition():
    raise SkipTest()
    # This used to be the test case for compiler.AdditionFinder, so check
    # git history for that...
    obj = compiler.GreedyAdditionFinder(compiler.AdditionCache(
        compiler.ConversionCache(mock_cost_map)))
    ctx, (A, a), (B, b) = create_mock_matrices('A B')
    ctx.define(A, B, name='A2B', cost=100)

    matrix_descriptions = [mock_compiled_node(A), mock_compiled_node(B), mock_compiled_node(B)]
    options = obj.find_cheapest_addition(matrix_descriptions)


    def format_options(options):
        return [(x.get_total_cost().weigh(mock_cost_map), task_to_str(x)) for x in options]

    B_solution = (10.0,
                  'T1 = mock_B[cost=4.0](); T3 = mock_B[cost=3.0](); T4 = mock_B[cost=1.0](); '
                  'T2 = add_B_B(T3, T4); T0 = add_B_B(T1, T2)')

    assert [B_solution] == options
    print
    ctx.define(A + B, A, cost=1)
    options = format_options(obj.find_cheapest_addition(matrix_descriptions))
    assert options == [
        (9.0, 'T1 = mock_A[cost=3.0](); T3 = mock_B[cost=3.0](); T4 = mock_B[cost=1.0](); '
         'T2 = add_B_B(T3, T4); T0 = add_A_B(T1, T2)'),
        (10.0, 'T2 = mock_B[cost=3.0](); T3 = mock_B[cost=1.0](); T1 = add_B_B(T2, T3); '
         'T4 = mock_B[cost=4.0](); T0 = add_B_B(T1, T4)')]

def benchmark_cheapest_addition():
    k = 4
    n = 5
    case = 'dense' # 'sparse', 'dense'
    
    obj = compiler.CheapestAdditionFinder(compiler.AdditionCache(
        compiler.ConversionCache(mock_cost_map)))
    mocks = create_mock_matrices(['Kind%d' % i for i in range(k)])
    ctx = mocks[0]
    kinds = [x[0] for x in mocks[1:]]

    if case == 'dense':
        for i, lkind in enumerate(kinds):
            for rkind in kinds[i:]:
                ctx.define(lkind, rkind, cost=1)
                ctx.define(lkind + rkind, lkind, cost=1)

    matrix_descriptions = [mock_task(kinds[i % k], 100) for i in range(n)]
    t0 = time.clock()
    obj.find_cheapest_addition(matrix_descriptions)
    t = time.clock()
    print
    print 'n=%d k=%d %s: time taken %s, nodes visited %s' % (n, k, case, t - t0, obj.nodes_visited)

    


#
# Full expression compilation tests
#

def test_add():
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A') 
    B, b, bu, buh = ctx.new_matrix('B')
    ctx.define(A + B, A)
    assert_compile('T0 = add_A_B(a, b)', a + b)
    assert_compile(['T1 = add_B_B(b, b); T0 = add_A_B(a, T1)',
                    'T1 = add_A_B(a, b); T0 = add_A_B(T1, b)'], a + b + b)
    assert_compile(['T2 = add_A_A(a, a); T1 = add_A_A(a, T2); T0 = add_A_A(a, T1)'], a + a + a + a)


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
    ctx.define(B * B, B, cost=10)
    assert_compile('T1 = multiply_A_B(a, b); T0 = multiply_A_B(T1, b)', a * b * b)
    ctx.define(B * B, B, cost=0.1)
    assert_compile('T1 = multiply_B_B(b, b); T0 = multiply_A_B(a, T1)', a * b * b)
    C, c, cu, cuh = ctx.new_matrix('B')
    ctx.define(A * C, A)
    assert_compile('T1 = multiply_A_B(a, b); T0 = multiply_A_B(T1, b)',
                   a * b * c)

def test_multiply_convert_transpose():
    ctx, (A, a) = create_mock_matrices('A')
    ctx.define(A * A, A)
    ctx.define(A.h, A)
    assert_compile('T1 = Ah(a); T0 = multiply_A_A(T1, a)', a.h * a)

def test_multiply_transpose():
    ctx, (A, a) = create_mock_matrices('A')
    ctx.define(A.h * A, A)
    ctx.define(A * A.h, A)
    # TODO: test that metadata is transposed
    assert_compile('T0 = multiply_Ah_A(a, a)', a.h * a)
    assert_compile('T0 = multiply_A_Ah(a, a)', a * a.h)

def test_multiply_transpose_expression():
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

def test_distributive_right():
    ctx, (A, a), (B, b), (C, c), (D, d) = create_mock_matrices('A B C D')
    ctx.define(A * B, A)
    ctx.define(A * C, A)
    ctx.define(B * D, B)
    ctx.define(C * D, C)

    # Right-distribute
    assert_compile('''
    T1 = multiply_A_B(a, b);
    T2 = multiply_A_C(a, c);
    T0 = add_A_A(T1, T2)
    ''', a * (b + c)) # b + c is impossible

    assert_compile('''
    T1 = multiply_A_B(a, b);
    T4 = multiply_B_D(b, d);
    T3 = multiply_A_B(a, T4);
    T6 = multiply_C_D(c, d);
    T5 = multiply_A_C(a, T6);
    T2 = add_A_A(T3, T5);
    T0 = add_A_A(T1, T2)
    ''', a * (b * d + b + c * d))

    assert_compile('''
    T2 = multiply_A_B(a, b);
    T1 = multiply_A_B(T2, b);
    T3 = multiply_A_C(T2, c);
    T0 = add_A_A(T1, T3)
    ''', a * b * (b + c))

def test_distributive_left():
    ctx, (A, a), (B, b), (C, c), (D, d) = create_mock_matrices('A B C D')
    ctx.define(A * C, A)
    ctx.define(B * C, A)
    ctx.define(C * D, C)
    assert_compile('''
    T2 = multiply_C_D(c, d);
    T1 = multiply_A_C(a, T2);
    T3 = multiply_B_C(b, T2);
    T0 = add_A_A(T1, T3)
    ''', (a + b) * c * d)

def test_distributive_nested():
    ctx, (A, a), (B, b) = create_mock_matrices('A B')
    ctx.define(A * B, B)
    a0, a1, a2, a3, a4, a5, a6, a7 = [Matrix(A(i, 3, 3), name='a%d' % i) for i in range(8)]
    
    # This one produced a problem with shuffles
    assert_compile('T1 = multiply_A_B(a0, b); T4 = multiply_A_B(a4, b); '
                   'T3 = multiply_A_B(a3, T4); T6 = multiply_A_B(a2, T4); '
                   'T5 = multiply_A_B(a1, T6); T2 = add_B_B(T3, T5); '
                   'T0 = add_B_B(T1, T2)', (a0 + (a1 * a2 + a3) * a4) * b)




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
    T2 = multiply_A_C(a, c);
    T1 = multiply_A_C(a, T2);
    T3 = multiply_A_C(a, c);
    T0 = add_C_C(T1, T3)
    ''', (a + a * a) * c)




def test_transpose():
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A') 
    B, b, bu, buh = ctx.new_matrix('B')
    ctx.define(B.h, A)
    ctx.define(A * A, A)
    assert_compile('T1 = Bh(b); T0 = multiply_A_A(T1, a)', b.h * a)

def test_factor_simple():
    ctx, (A, a) = create_mock_matrices('A')
    ctx.define(A.f, A)
    assert_compile('T0 = Af(a)', a.f)

def test_factor_full():
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


def test_find_cheapest_direct_computation():
    raise SkipTest()
    cost_map = cost_value.default_cost_map
    ctx, (Dense, de), (Diagonal, di) = create_mock_matrices(
        'Dense Diagonal', [100 * FLOP, 11 * FLOP])
    ctx.define(Diagonal, Dense, 100 * MEMOP)
    def expr_of(mat):
        meta_tree, args = transforms.metadata_transform(mat._expr)
        return meta_tree
    #print compiler.find_cheapest_direct_computation(expr_of(de + di), cost_map)

def test_correct_distributive_cost():
    ctx, (A, a), (B, b), (C, c), (D, d), (E, e) = create_mock_matrices('A B C D E')
    # look at (a + b) * c * d = (a * c * d) + (b * c * d)
    # Now, make sure that the reuse of the result of (c * d) (kind E) is
    # taken into account in cost calculations:
    
    # ((a * c) * d): Impossible, so always takes a * (c * d) (cost=2)
    # b * (c * d): Costs 2 + 2 = 4; but 2 shared, so in reality 2
    # (b * c) * d: Costs 3; no cost sharing
    
    ctx.define(C * D, E, cost=2)
    ctx.define(A * E, A, cost=0)
    ctx.define(B * E, A, cost=2)
    ctx.define(B * C, A, cost=3)
    ctx.define(A * D, A, cost=0) # should not take!

    assert_compile('T2 = multiply_C_D(c, d); '
                   'T1 = multiply_B_E(b, T2); '
                   'T3 = multiply_A_E(a, T2); '
                   'T0 = add_A_A(T1, T3)', (a + b) * c * d)

def test_nonoptimal_distribution():
    # This is a case where the compiler returns suboptimal results!
    # (Or, currently, doesn't allow the operation)
    # So this is a test case one can use if one wants to work on that
    ctx, (A, a), (B, b), (C, c), (D, d), (E, e) = create_mock_matrices('A B C D E')
    ctx.define(A * C, A)
    ctx.define(A * D, A)
    ctx.define(C * D, E)
    ctx.define(B * E, A)

    # A solution is
    # T2 = multiply_A_C(a, c); T1 = multiply_A_D(T2, d); T4 = multiply_C_D(c, d);
    # T3 = multiply_B_E(b, T4); T0 = add_A_A(T1, T3)
    # but this is not found
    with assert_raises(ImpossibleOperationError):
        assert_compile('', (a + b) * c * d)
    
    


def benchmark_distributive():
    # prefer diagonal multiplication to addition, to create a deterministic
    # result
    ctx, (Dense, de), (Diagonal, di) = create_mock_matrices(
        'Dense Diagonal', [100 * FLOP, 11 * FLOP])
    ctx.define(Diagonal, Dense, cost=100 * MEMOP)
    ctx.define(Diagonal * Diagonal, Diagonal, cost=10 * FLOP)

    def mat(name):
        return Matrix(Diagonal(i, 10, 10), name=name)

    nprod_outer = 4
    nadd = 4
#    nprod_inner = 4

    matexpr = 1
    for i in range(nprod_outer):
        terms = 0
        for j in range(nadd):
            terms += mat('m%d_%d' % (i, j))
        matexpr = matexpr * terms

    print matexpr
    print
    c = compiler.GreedyCompiler()
    t0 = time.clock()
    tree, args = c.compile(matexpr._expr)
    t = time.clock()
    print 'Time taken %s, stats %s' % (t - t0, c.stats)
    print 'Cost:', tree.as_task().get_total_cost()
    print 'Solution:\n   ', cnode_to_str(tree, args, sep='\n    ')

