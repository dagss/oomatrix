from .common import *
from .. import Matrix, compute, explain, symbolic
from ..compiler import SimplisticCompiler
from ..core import (ConversionGraph, AdditionGraph, MatrixImpl, MultiplyPairGraph,
                    ImpossibleOperationError, credits)

def arrayeq_(x, y):
    assert np.all(x == y)

class MockKind(MatrixImpl):
    def __init__(self, value, nrows, ncols):
        self.value = value
        self.nrows = nrows
        self.ncols = ncols
        self.dtype = np.double
        
    def __repr__(self):
        return '<%s:%s>' % (type(self).name, self.value)

class MockMatricesUniverse:
    def __init__(self):
        self.conversion_graph = ConversionGraph()
        self.conversion = (
            self.conversion_graph.conversion_decorator)
        self.add_graph = AdditionGraph(
            self.conversion_graph)
        self.multiply_graph = MultiplyPairGraph(
            self.conversion_graph)
        self.multiplication = (
            self.multiply_graph.multiplication_decorator)
        self.addition = self.add_graph.addition_decorator

    def define_mul(self, a, b, result):
        def get_kind_and_transpose(x):
            if isinstance(x._expr, symbolic.LeafNode):
                return x.get_type(), ''
            elif isinstance(x._expr, symbolic.ConjugateTransposeNode):
                return type(x._expr.child.matrix_impl).H, '.h'
        
        a_kind, ah = get_kind_and_transpose(a)
        b_kind, bh = get_kind_and_transpose(b)
        result_kind, rh = get_kind_and_transpose(result)

        @self.multiplication((a_kind, b_kind), result_kind)
        def mul(a, b):
            x = result_kind('(%s%s * %s%s)' %
                              (a.value, ah, b.value, bh),
                              a.nrows, b.ncols)
            return x

    def define_conv(self, a, result):
        def get_kind_and_transpose(x):
            if isinstance(x._expr, symbolic.LeafNode):
                return x.get_type(), ''
            elif isinstance(x._expr, symbolic.ConjugateTransposeNode):
                return type(x._expr.child.matrix_impl).H, '.h'

        a_kind, ah = get_kind_and_transpose(a)
        result_kind, rh = get_kind_and_transpose(result)
        @self.conversion(a_kind, result_kind)
        def conv(a):
            return result_kind('%s%s%s' %
                               (result_kind.name, a.value, ah),
                               a.nrows, a.ncols)


    def new_matrix(self, name_,
                   right=(), right_h=(), add=(),
                   result='self'):
        class NewKind(MockKind):
            name = name_
            is_transpose_kind = False
            def conjugate_transpose(self):
                return NewKind_H(
                    self.value, self.ncols, self.nrows)

        class NewKind_H(MockKind):
            name = name_ + '.H'
            def conjugate_transpose(self):
                return NewKind(self.value, self.ncols, self.nrows)
            def __repr__(self):
                return '<%s:%s>' % (type(self).name,
                                    self.value)
                

        NewKind.conjugate_transpose_class = NewKind_H
        NewKind_H.conjugate_transpose_class = NewKind

        if result == 'self':
            result_kind = NewKind
        else:
            result_kind = result.get_type()

        # Always have within-kind addition
        @self.addition((NewKind, NewKind), NewKind)
        def add(a, b):
            return NewKind('(%s + %s)' % (a.value, b.value),
                           a.nrows, b.ncols)
        
        return (Matrix(NewKind(name_, 3, 3)),
                Matrix(NewKind(name_.lower(), 3, 1)),
                Matrix(NewKind(name_.lower() + 'h', 1, 3)))
        

def test_stupid_compiler_mock():
    ctx = MockMatricesUniverse()
    compiler = SimplisticCompiler(
        multiply_graph=ctx.multiply_graph,
        add_graph=ctx.add_graph)
    co = compiler.compute
#    ex = compiler.explain
    def test(expected, M, target_kind=None):
        eq_(expected, repr(co(M)._expr.matrix_impl))

    A, a, ah = ctx.new_matrix('A')
    B, b, bh = ctx.new_matrix('B')

    #
    # Straight multiplication
    #

    # A * B -> B
    ctx.define_mul(A, B, B)
    yield test, '<B:(A * B)>', A * B
    
    yield (assert_raises, ImpossibleOperationError,
           test, '<B:(A * B)>', B * A)
    # B * A -> B.h
    ctx.define_mul(B, A, B.h)
    yield test, '<B.H:(B * A)>', B * A

    # order of multiplication (mat-mat vs. mat-vec)
    ctx.define_mul(A, A, A)
    yield test, '<A:(((A * A) * A) * A)>', A * A * A * A
    yield test, '<A:(A * (A * (A * a)))>', A * A * A * a
    yield test, '<A:(((ah * A) * A) * A)>', ah * A * A * A

    
    #
    # Transpose multiplication
    #

    # Cause transpose of entire product
    yield test, '<A.H:(A * A)>', A.h * A.h
    yield test, '<B:(B * A)>', A.h * B.h # -> (B * A).h which has
                                         # kind B.H.H

    # Straightforward conjugate -- doesn't work until mul action
    # is registered
    yield (assert_raises, ImpossibleOperationError,
           test, '', A * A.h)
    ctx.define_mul(A, A.h, A)
    yield test, '<A:(A * A.h)>', A * A.h

    yield (assert_raises, ImpossibleOperationError,
           test, '', A.h * A)
    ctx.define_mul(A.h, A, A)
    yield test, '<A:(A.h * A)>', A.h * A


    #
    # Post-multiply conversion
    #
    S, s, sh = ctx.new_matrix('S') # only used as conversion 'sink'
    ctx.define_conv(A, S)
    yield test, '<S:S(A * A)>', (A * A).as_kind(S.get_type())

    #
    # Addition
    #

    # Normal add
    yield test, '<A:(A + A)>', A + A

    # Do not use distributive rule for matrices
    yield test, '<A:((A + A) * A)>', (A + A) * A
    yield test, '<A:(A * (A + A))>', A * (A + A)

    # But, use it for vectors
    yield test, '<A:((A * a) + (A * a))>', (A + A) * a
    yield test, '<A:((ah * A) + (ah * A))>', ah * (A + A)
    yield (test, '<A:(((((ah * A) + (ah * A)) * A) + '
                      '(((ah * A) + (ah * A)) * A)) * A)>',
           ah * (A + A) * (A + A) * A)

def test_stupid_compiler_numpy():
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
