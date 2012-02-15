from .common import *
from .. import Matrix, compute, explain, symbolic

from ..kind import MatrixImpl, MatrixKind, ConjugateTransposePatternNode
from ..computation import computation, conversion, ImpossibleOperationError

from ..compiler import *

def arrayeq_(x, y):
    assert np.all(x == y)

def test_set_of_pairwise_nonempty_splits():
    eq_([(('a',), ('b',)),
         (('b',), ('a',))],
        list(set_of_pairwise_nonempty_splits('ab')))
    
    eq_([(('a',), ('b', 'c', 'd')),
         (('b',), ('a', 'c', 'd')),
         (('c',), ('a', 'b', 'd')),
         (('d',), ('a', 'b', 'c')),
         (('a', 'b'), ('c', 'd')),
         (('a', 'c'), ('b', 'd')),
         (('a', 'd'), ('b', 'c')),
         (('b', 'c'), ('a', 'd')),
         (('b', 'd'), ('a', 'c')),
         (('c', 'd'), ('a', 'b')),
         (('a', 'b', 'c'), ('d',)),
         (('a', 'b', 'd'), ('c',)),
         (('a', 'c', 'd'), ('b',)),
         (('b', 'c', 'd'), ('a',))
         ],
        list(set_of_pairwise_nonempty_splits('abcd')))

class MockKind(MatrixImpl):
    def __init__(self, value, nrows, ncols):
        self.value = value
        self.nrows = nrows
        self.ncols = ncols
        self.dtype = np.double
        
    def __repr__(self):
        return '%s:%s' % (type(self).name, self.value)

class MockMatricesUniverse:
    def __init__(self):
        pass

    def define_mul(self, a, b, result_kind):
        def get_kind_and_transpose(x):
            if isinstance(x, ConjugateTransposePatternNode):
                return x.child, '.h'
            else:
                return x, ''
        
        a_kind, ah = get_kind_and_transpose(a)
        b_kind, bh = get_kind_and_transpose(b)
        assert isinstance(result_kind, MatrixKind)

        @computation(a_kind * b_kind, result_kind)
        def mul(a, b):
            x = result_kind('(%s%s * %s%s)' %
                              (a.value, ah, b.value, bh),
                              a.nrows, b.ncols)
            return x

    def define(self, match, result_kind, reprtemplate):
        reprtemplate = '(%s)' % reprtemplate
        @computation(match, result_kind)
        def comp(*args):
            return result_kind(reprtemplate % tuple(arg.value for arg in args),
                               args[0].nrows, args[-1].ncols)

    def define_conv(self, from_kind, to_kind):
        @conversion(from_kind, to_kind)
        def conv(a):
            return to_kind('%s(%s)' % (to_kind.name, a.value),
                           a.nrows, a.ncols)

    def new_matrix(self, name_,
                   right=(), right_h=(), add=(),
                   result='self'):
        class NewKind(MockKind):
            name = name_

        if result == 'self':
            result_kind = NewKind
        else:
            result_kind = result.get_type()

        # Always have within-kind addition
        @computation(NewKind + NewKind, NewKind)
        def add(a, b):
            return NewKind('(%s + %s)' % (a.value, b.value),
                           a.nrows, b.ncols)
        
        return (NewKind,
                Matrix(NewKind(name_.lower(), 3, 3)),
                Matrix(NewKind(name_.lower() + 'u', 3, 1)),
                Matrix(NewKind(name_.lower() + 'uh', 1, 3)))
        


def assert_impossible(M):
    compiler = ExhaustiveCompiler()
    with assert_raises(ImpossibleOperationError):
        compiler.compile(M._expr)

def co_(expected, M, target_kind=None):
    compiler = ExhaustiveCompiler()
    expr = M.compute(compiler=compiler)._expr
    if isinstance(expr, symbolic.ConjugateTransposeNode):
        r = '[%r].h' % expr.child.matrix_impl
    else:
        r = repr(expr.matrix_impl)
    eq_(expected, r)

def test_exhaustive_compiler():
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
    
    ctx.define(S.h, S, 'sym(%s)')
    ctx.define(S * S, S, '%s * %s')

    # Disallowed multiplication
    assert_impossible(a * b)

    # Straight pair multiplication
    ctx.define_mul(A, B, B)
    co_('B:(a * b)', a * b)

    # Multiple operands
    ctx.define_mul(A, A, A)
    co_('B:(a * (a * (a * (a * b))))', a * a * a * a * b)

    # Multiplication with conversion
    assert_impossible(a * c)
    ctx.define_conv(A, B)
    ctx.define_mul(B, C, C)
    co_('C:(B(a) * c)', a * c)
    # ...and make sure there's no infinite loops of conversions
    ctx.define_conv(B, C)
    ctx.define_conv(C, A)
    co_('C:(B(a) * c)', a * c)
    ctx.define_conv(B, A)
    co_('C:(B(a) * c)', a * c)
    co_('A:(a * a)', a * a)
    # Multiplication with forced target kind
    co_('C:(B(a) * c)', a * c)
    co_('A:(a * A(c))', (a * c).as_kind(A))
    # Forced post-conversion
    ctx.define_conv(C, D)
    ctx.define_conv(D, E)
    co_('D:D((B(a) * c))', (a * c).as_kind(D))
    co_('E:E(D((B(a) * c)))', (a * c).as_kind(E))

    # Transposed operands in multiplication
    assert_impossible(a * a.h)
    ctx.define(A * A.h, A, '%s * %s.h')
    co_('A:(a * a.h)', a * a.h)

    co_('[S:(s * (sym(s)))].h', s * s.h) # TODO: prefer 'S:(s * (sym(s)))'
    co_('[S:(s * s)].h', s.h * s.h)
    

    # Addition
    co_('A:(a + a)', a + a)
    assert_impossible(a + b)
    ctx.define(A + B, A, '%s + %s')
    co_('A:(a + b)', a + b)
    co_('A:(a + b)', b + a) # note how arguments are sorted
    co_('A:((a + (a + b)) + b)', b + a + b + a)

    # Addition through conversion TODO
    #ctx.define_conv(C, A)
    #co_('A:(a + c)', a + c)

    # Transposed operands in addition
    assert_impossible(a.h + a)
    ctx.define(A.h + A, A, '%s.h + %s')
    co_('A:(a.h + a)', a.h + a)
    co_('A:(a + (a.h + (a + (a.h + a))))', a + a.h + a + a.h + a)
    co_('A:(a.h + (a.h + a))', a.h + a.h + a)
    ctx.define(B + B.h, B, '%s + %s.h')
    co_('B:(b + b.h)', b.h + b)
    # transpose only thorugh symmetry conversion
    #co_('S:(s + (sym(s)))', s + s.h) addition through conversion todo
    
    # Nested expressions
    co_('A:((a * a) + (a * a))', a * a + a * a)
    co_('B:((a + a) * b)', (a + a) * b)
    # force use of distributive law...
    #ctx.define(A * C, C, '%s * %s')
    #ctx.define(C * C, C, '%s * %s')
    #co_('B:((a + a) * b)', (a + c) * c)

def test_exhaustive_compiler_more_mul():
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
