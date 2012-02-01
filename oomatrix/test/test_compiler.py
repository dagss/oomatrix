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

    def define_add(self, a_kind, b_kind, result_kind):
        @computation(a_kind + b_kind, result_kind)
        def add(a, b):
            return result_kind('(%s + %s)' % (a.value, b.value),
                               a.nrows, a.ncols)

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
        




def test_exhaustive_compiler():
    ctx = MockMatricesUniverse()
    compiler = ExhaustiveCompiler()
    def test(expected, M, target_kind=None):
        eq_(expected, repr(compiler.compute(M)._expr.matrix_impl))
    def assert_impossible(M):
        with assert_raises(ImpossibleOperationError):
            compiler.compute(M)

    A, a, au, auh = ctx.new_matrix('A')
    B, b, bu, buh = ctx.new_matrix('B')
    C, c, cu, cuh = ctx.new_matrix('C')

    # Disallowed multiplication
    assert_impossible(a * b)

    # Straight pair multiplication
    ctx.define_mul(A, B, B)
    test('B:(a * b)', a * b)

    # Multiple operands
    ctx.define_mul(A, A, A)
    test('B:(a * (a * (a * (a * b))))', a * a * a * a * b)

    # Multiplication with conversion
    assert_impossible(a * c)
    ctx.define_conv(A, B)
    ctx.define_mul(B, C, C)
    test('C:(B(a) * c)', a * c)
    # ...and make sure there's no infinite loops of conversions
    ctx.define_conv(B, C)
    ctx.define_conv(C, A)
    test('C:(B(a) * c)', a * c)
    ctx.define_conv(B, A)
    test('C:(B(a) * c)', a * c)
    test('A:(a * a)', a * a)

    # Addition
    test('A:(a + a)', a + a)
    assert_impossible(a + b)
    ctx.define_add(A, B, A)
    test('A:(a + b)', a + b)
    test('A:(a + b)', b + a) # note how arguments are sorted
    test('A:((a + (a + b)) + b)', b + a + b + a)








def test_stupid_compiler_mock():
    return
    ctx = MockMatricesUniverse()
    compiler = SimplisticCompiler()
    co = compiler.compute
#    ex = compiler.explain
    def test(expected, M, target_kind=None):
        eq_(expected, repr(co(M)._expr.matrix_impl))

    A, a, au, auh = ctx.new_matrix('A')
    B, b, bu, buh = ctx.new_matrix('B')

    #
    # Straight multiplication
    #

    # A * B - B
    ctx.define_mul(A, B, B)
    test('B:(a * b)', a * b)

    with assert_raises(ImpossibleOperationError):
        test('', b * a)

    # order of multiplication (mat-mat vs. mat-vec)
    ctx.define_mul(A, A, A)
    test('A:(((a * a) * a) * a)', a * a * a * a)
    test('A:(a * (a * (a * au)))', a * a * a * au)
    test('A:(((auh * a) * a) * a)', auh * a * a * a)
    return
    #
    # Transpose multiplication
    #
    # Straightforward conjugate -- doesn't work until mul action
    # is registered
    with assert_raises(ImpossibleOperationError):
        test('', a * a.h)
    return

    ctx.define_mul(A, A.h, A)
    yield test, 'A:(A * A.h)', A * A.h

    yield (assert_raises, ImpossibleOperationError,
           test, '', A.h * A)
    ctx.define_mul(A.h, A, A)
    yield test, 'A:(A.h * A)', A.h * A


    #
    # Post-multiply conversion
    #
    S, s, sh = ctx.new_matrix('S') # only used as conversion 'sink'
    ctx.define_conv(A, S)
    yield test, 'S:S(A * A)', (A * A).as_kind(S.get_type())

    #
    # Addition
    #

    # Normal add
    yield test, 'A:(A + A)', A + A

    # Do not use distributive rule for matrices
    yield test, 'A:((A + A) * A)', (A + A) * A
    yield test, 'A:(A * (A + A))', A * (A + A)

    # But, use it for vectors
    yield test, 'A:((A * a) + (A * a))', (A + A) * a
    yield test, 'A:((ah * A) + (ah * A))', ah * (A + A)
    yield (test, 'A:(((((ah * A) + (ah * A)) * A) + '
                      '(((ah * A) + (ah * A)) * A)) * A)',
           ah * (A + A) * (A + A) * A)

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
