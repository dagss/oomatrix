import re
from nose.tools import eq_, ok_, assert_raises
from ..kind import MatrixImpl
from ..formatter import BasicExpressionFormatter
from ..symbolic import *

class MockImpl(MatrixImpl):
    nrows = ncols = 3
    dtype = None

def format(expr):
    return BasicExpressionFormatter({}).format(expr)

def assert_expr(expected_repr, expr):
    got = format(expr)
    expected_repr = re.sub('\s', '', expected_repr)
    got = re.sub('\s', '', got)
    eq_(expected_repr, got)

impl = MockImpl()
a = LeafNode('a', impl)
b = LeafNode('b', impl)
c = LeafNode('c', impl)
d = LeafNode('d', impl)
e = LeafNode('e', impl)

def add(*args):
    return AddNode(args)

def mul(*args):
    return MultiplyNode(args)

def H(arg):
    return ConjugateTransposeNode(arg)

def I(arg):
    return InverseNode(arg)

def test_basic():
    yield assert_expr, 'a * (b + c + d)', mul(a, add(b, c, d))

def test_tree_constraints():
    yield ok_, H(H(a)) is a
    yield ok_, I(I(a)) is a
    yield assert_expr, 'a.i.h', H(I(a))
    yield assert_expr, 'a.i.h', I(H(a))
    yield ok_, I(H(H(I(a)))) is a
    yield ok_, H(I(H(I(a)))) is a
    yield ok_, H(I(I(H(a)))) is a
    yield ok_, I(H(I(H(a)))) is a


def test_distributive():
    inp = mul(add(a, b, c), d)
    out = apply_right_distributive_rule(inp)
    yield assert_expr, 'a * d + b * d + c * d', out
    out = apply_left_distributive_rule(inp)
    yield assert_expr, '(a + b + c) * d', out

    # e * (a + b + c) * e * (c + c) * d * e
    inp = mul(e, add(a, b, c), e, add(c, c), d, e)
    out = apply_right_distributive_rule(inp)
    yield (assert_expr, '''
        e * (a * e * (c * d * e + c * d * e) +
             b * e * (c * d * e + c * d * e) +
             c * e * (c * d * e + c * d * e))''', out)

    out = apply_left_distributive_rule(inp)
    yield (assert_expr, '''
        ((e * a + e * b + e * c) * e * c +
         (e * a + e * b + e * c) * e * c) * d * e''', out)

def test_get_key():
    class A(MatrixImpl):
        _sort_id = 1
        nrows = ncols = 3
        dtype = None

    class B(MatrixImpl):
        _sort_id = 2
        nrows = ncols = 3
        dtype = None

    a = LeafNode('a', A())
    b = LeafNode('b', B())
    key = mul(add(I(H(b)), b, a), a).get_key()
    # note that the + is sorted
    eq_(('*',
         ('+', A, B, ('h', ('i', B))),
         A), key)

    # reverse sort order and construct new tree, now the order should be B, A
    A._sort_id = 3
    key = mul(add(I(H(b)), b, a), a).get_key()
    eq_(('*',
         ('+', B, A, ('h', ('i', B))),
         A), key)
    
