import re
from nose.tools import eq_, ok_, assert_raises
from ..symbolic import *
from ..formatter import BasicExpressionFormatter

class MockImpl:
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
    return MulNode(args)

def test_basic():
    yield assert_expr, 'a * (b + c + d)', mul(a, add(b, c, d))

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

