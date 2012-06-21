from .common import *
from .. import Matrix, compute, explain, symbolic

from ..kind import MatrixImpl, MatrixKind
from ..computation import (computation, conversion, ImpossibleOperationError,
                           FLOP, UGLY)

from ..compiler import ShortestPathCompiler

from .mock_universe import MockKind, MockMatricesUniverse, create_mock_matrices

def arrayeq_(x, y):
    assert np.all(x == y)

def assert_impossible(M):
    compiler = ShortestPathCompiler()
    with assert_raises(ImpossibleOperationError):
        compiler.compile(M._expr)

def co_(expected, M, target_kind=None):
    compiler = ShortestPathCompiler()
    expr = M.compute(compiler=compiler)._expr
    if isinstance(expr, symbolic.ConjugateTransposeNode):
        r = '[%r].h' % expr.child.matrix_impl
    else:
        r = repr(expr.matrix_impl)
    if isinstance(expected, list):
        if not r in expected:
            raise AssertionError("Could not find %r in %r" % (r, expected))
    else:
        eq_(expected, r)

def test_basic1():
    ctx = MockMatricesUniverse()

    # D: only exists as a conversion target and source. No ops
    # E: only way to E is through conversion from D. No ops.
    # S: symmetric matrix; no ops with others
    A, a, au, auh = ctx.new_matrix('A')
    B, b, bu, buh = ctx.new_matrix('B')
    C, c, cu, cuh = ctx.new_matrix('C')
    D, d, du, duh = ctx.new_matrix('D')
    E, e, eu, euh = ctx.new_matrix('E')
    S, s, su, suh = ctx.new_matrix('S')

    ctx.define(S.h, S, 'sym(%s)', cost=0)
    ctx.define(S * S, S, '%s * %s')
    
    # Disallowed multiplication
    assert_impossible(a * b)

    # Straight pair multiplication
    ctx.define(A * B, B, '%s * %s')
    co_('B:(a * b)', a * b)
    # Multiple operands
    co_('B:(a * (a * (a * (a * b))))', a * a * a * a * b)
    ctx.define(A * A, A, '%s * %s')

    # Multiplication with conversion
    assert_impossible(a * c)
    ctx.define_conv(A, B)
    ctx.define(B * C, C, '%s * %s')
    co_('C:(B(a) * c)', a * c)
    # ...and make sure there's no infinite loops of conversions
    ctx.define_conv(B, C)
    ctx.define_conv(C, A)
    co_(['C:(B(a) * c)', 'A:(a * A(c))'], a * c)
    ctx.define_conv(B, A)
    co_(['C:(B(a) * c)', 'A:(a * A(c))'], a * c)
    co_('A:(a * a)', a * a)
    # Multiplication with forced target kind
    co_(['C:(B(a) * c)', 'A:(a * A(c))'], a * c)
    #co_('A:(a * A(c))', (a * c).as_kind(A)) # TODO
    # Forced post-conversion
    ctx.define_conv(C, D)
    ctx.define_conv(D, E)
    #co_('D:D((B(a) * c))', (a * c).as_kind(D)) # TODO
    #co_('E:E(D((B(a) * c)))', (a * c).as_kind(E)) # TODO
    # Transposed operands in multiplication
    assert_impossible(a * a.h)
    return

    ctx.define(A * A.h, A, '%s * %s.h')
    co_('A:(a * a.h)', a * a.h)

    co_('S:(s * (sym(s)))', s * s.h) # TODO: prefer 'S:(s * (sym(s)))'
    co_('[S:(s * s)].h', s.h * s.h)
    

    # transpose only thorugh symmetry conversion
    #co_('S:(s + (sym(s)))', s + s.h) addition through conversion todo
    
    # force use of distributive law...
    #ctx.define(A * C, C, '%s * %s')
    #ctx.define(C * C, C, '%s * %s')
    #co_('B:((a + a) * b)', (a + c) * c)

def test_basic2():
    ctx = MockMatricesUniverse()

    # D: only exists as a conversion target and source. No ops
    # E: only way to E is through conversion from D. No ops.
    # S: symmetric matrix; no ops with others
    A, a, au, auh = ctx.new_matrix('A')
    B, b, bu, buh = ctx.new_matrix('B')
    C, c, cu, cuh = ctx.new_matrix('C')
    D, d, du, duh = ctx.new_matrix('D')
    E, e, eu, euh = ctx.new_matrix('E')
    S, s, su, suh = ctx.new_matrix('S')
    
    ctx.define(S.h, S, 'sym(%s)', cost=0)
    ctx.define(S * S, S, '%s * %s')

    ctx.define(A * B, B, '%s * %s')
    ctx.define(A * A, A, '%s * %s')
    ctx.define_conv(A, B)
    ctx.define(B * C, C, '%s * %s')
    ctx.define_conv(B, C)
    ctx.define_conv(C, A)
    ctx.define_conv(B, A)
    ctx.define_conv(C, D)
    ctx.define_conv(D, E)
    #co_('D:D((B(a) * c))', (a * c).as_kind(D)) # TODO
    #co_('E:E(D((B(a) * c)))', (a * c).as_kind(E)) # TODO
    # Transposed operands in multiplication
    assert_impossible(a * a.h)

def test_nested():
    ctx = MockMatricesUniverse()

    # D: only exists as a conversion target and source. No ops
    # E: only way to E is through conversion from D. No ops.
    # S: symmetric matrix; no ops with others
    A, a, au, auh = ctx.new_matrix('A')
    B, b, bu, buh = ctx.new_matrix('B')
    C, c, cu, cuh = ctx.new_matrix('C')
    D, d, du, duh = ctx.new_matrix('D')
    E, e, eu, euh = ctx.new_matrix('E')

    ctx.define(A * B, B, '%s * %s')
    ctx.define(A * A, A, '%s * %s')
    ctx.define_conv(A, B)
    ctx.define_conv(B, C)
    ctx.define_conv(B, A)
    ctx.define_conv(C, A)
    ctx.define_conv(C, D)
    ctx.define_conv(D, E)

    # Nested expressions
    co_('A:((a * a) + (a * a))', a * a + a * a)
    #co_('B:((a + a) * b)', (a + a) * b)
    

def test_add():
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A')
    B, b, bu, buh = ctx.new_matrix('B')
    C, c, cu, cuh = ctx.new_matrix('C')

    # Addition
    co_('A:(a + a)', a + a)
    assert_impossible(a + b)
    ctx.define(A + B, A, '%s + %s')
    co_('A:(a + b)', a + b)
    co_('A:(a + b)', b + a) # note how arguments are sorted
    co_('A:(a + (a + (b + b)))', b + a + b + a)

    # Addition through conversion
    ctx.define_conv(C, A)
    co_('A:(a + A(c))', a + c)

    # Transposed operands in addition
    assert_impossible(a.h + a)
    ctx.define(A.h + A, A, '%s.h + %s')
    co_('A:(a.h + a)', a.h + a)
    co_('A:(a + ((a.h + a) + (a.h + a)))', a + a.h + a + a.h + a)
    co_('A:(a.h + (a.h + a))', a.h + a.h + a)
    ctx.define(B + B.h, B, '%s + %s.h')
    co_('B:(b + b.h)', b.h + b)

def test_more_mul():
    ctx = MockMatricesUniverse()
    # check that ((ab)c) is found when only option
    A, a, au, auh = ctx.new_matrix('A')
    B, b, bu, buh = ctx.new_matrix('B')
    C, c, cu, cuh = ctx.new_matrix('C')
    ctx.define(A * B, A, '%s * %s')
    ctx.define(A * C, A, '%s * %s')
    co_('A:((a * b) * c)', a * b * c)
    # check that the conjugate is attempted
    ctx.define(A.h * B, A, '%s.h * %s')
    ctx.define(B.h, B, '%s.h')
    co_('[A:(a.h * (b.h))].h', b * a)

def test_distributive():
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A')
    B, b, bu, buh = ctx.new_matrix('B')
    # (a + b) * a, with no sum between A and B, so must use distributive
    ctx.define(A * A, A, "%s * %s")
    ctx.define(B * A, A, "%s * %s")
    co_('A:((a * a) + (b * a))', (a + b) * a)
    co_('A:((a * a) + (b * a))', (a.h + b.h).h * a)

def test_mixed():
    # check that we can safely mix mul and add
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A')
    B, b, bu, buh = ctx.new_matrix('B')
    ctx.define(A * A, A, "%s * %s")
    ctx.define(A.h * A, A, "%s.h * %s")
    co_('A:((a * a) + (a * a))', (a * a) + (a * a))
    co_('A:(a + (a.h * (a.h * (a * a))))', a + (a * a).h * (a * a))

def test_commander_failure():
    # This used to fail due to loosing nodes when applying distributive
    # law
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A')
    B, b, bu, buh = ctx.new_matrix('B')
    C, c, cu, cuh = ctx.new_matrix('C')
    ctx.define(A * A, A, "#|%s * %s")
    #ctx.define(A.h * A, A, "%s.h * %s")
    ctx.define(B * C, C, '#|%s * %s')
    ctx.define(A * C, C, '#|%s * %s')
    ctx.define(A.h * C, C, '#|%s.h * %s')
    ctx.define(B.h * C, C, '#|%s.h * %s')
    # Force distributive law (there's no A+B)
    ctx.reset()
    # the duplicate b*c is because caching currently happens *by
    # argument position*, not by contents
    co_('C:((0|b * c) + ((3|a.h * ((1|a * c) + (2|b * c))) + (4|b.h * ((1|a * c) + (2|b * c)))))',
        (b + ((a + b).h * (a + b))) * c)


def test_stupid_compiler_numpy():
    return
    De_array = np.arange(9).reshape(3, 3).astype(np.int64) + 1j
    De = Matrix(De_array, 'De')
    Di_array = np.arange(3).astype(np.complex128)
    Di = Matrix(Di_array, 'Di', diagonal=True)
    # The very basic computation
    compiler = SimplisticCompiler()
    co = compiler.compute
    a = ndrange((3, 1), dtype=np.complex128)
    yield ok_, type(De * a) is Matrix
    yield arrayeq_, co(De * a).as_array(), np.dot(De_array, a)
    yield arrayeq_, co(De.h * a).as_array(), np.dot(De_array.T.conjugate(), a)
