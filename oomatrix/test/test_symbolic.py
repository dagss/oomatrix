from .common import *
import re
from ..kind import MatrixImpl
from ..formatter import BasicExpressionFormatter
from ..symbolic import *
from ..computation import computation
from .. import symbolic

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
    return symbolic.add(args)

def mul(*args):
    return symbolic.multiply(args)

def H(arg):
    return conjugate_transpose(arg)

def I(arg):
    return inverse(arg)

def test_basic():
    yield assert_expr, 'a * (b + c + d)', mul(a, add(b, c, d))

def test_factories_obeys_constraints():
    assert H(H(a)) is a
    assert I(I(a)) is a
    assert_expr('a.i.h', H(I(a)))
    assert_expr('a.i.h', I(H(a)))
    assert I(H(H(I(a)))) is a
    assert H(I(H(I(a)))) is a
    assert H(I(I(H(a)))) is a
    assert I(H(I(H(a)))) is a
    assert mul(H(mul(b, a)), a) == mul(H(a), H(b), a)
    assert 3 == len(mul(H(mul(b, a)), a).children)
    assert add(H(add(b, a)), a) == add(H(b), H(a), a)
    assert 3 == len(add(H(add(b, a)), a).children)
    assert 2 == len(add(H(mul(b, a)), a).children)

def test_as_tuple():
    class A(MockImpl):
        _sort_id = 1
    class B(MockImpl):
        _sort_id = 2

    a = LeafNode('a', A())
    b = LeafNode('b', B())
    tup = mul(add(I(H(b)), b, a), a).as_tuple()
    # note that the + is sorted
    eq_(('*',
         ('+', a, b, ('h', ('i', b))),
         a), tup)

    # reverse sort order and construct new tree, now the order should be B, A
    A._sort_id = 3
    tup = mul(add(I(H(b)), b, a), a).as_tuple()
    eq_(('*',
         ('+', b, a, ('h', ('i', b))),
         a), tup)

def test_call_func():
    class A(MockImpl):
        _sort_id = 1
    class B(MockImpl):
        _sort_id = 2
    class C(MockImpl):
        _sort_id = 3
    ao = A()
    bo = B()
    co = C()
    a = LeafNode('a', ao)
    b = LeafNode('b', bo)
    c = LeafNode('c', co)

    def assert_argslist(expected, expr, match):
        args = expr.as_computable_list(match)
        for arg in args:
            assert isinstance(arg, LeafNode)
        eq_(expected, [x.matrix_impl for x in args])

    yield assert_argslist, [ao], I(a), A.i
    yield assert_raises, PatternMismatchError, a.as_computable_list, A.i
    yield assert_argslist, [ao], H(I(a)), A.i.h

    yield assert_argslist, [co, ao, bo], add(a, b, c), C + A + B
    yield assert_argslist, [ao, bo, co], add(a, b, c), A + B + C
    yield assert_argslist, [bo, co, ao], add(a, b, c), B + C + A

    yield assert_argslist, [co, ao, bo], mul(c, a, b), C * A * B

    yield (assert_argslist, [co, ao, bo, bo, co], 
           mul(c, add(mul(H(I(a)), b), mul(H(b), c))), 
           C * (A.i.h * B + B.h * C))

    # make C appear before A and try again, should be exact same behaviour
    C._sort_id = 0
    yield assert_argslist, [co, ao, bo], add(a, b, c), C + A + B
    
def test_hash_and_eq():
    from ..symbolic import conjugate_transpose, inverse


    class A(MockImpl):
        _sort_id = 1
    class B(MockImpl):
        _sort_id = 2
    ao = A()
    bo = B()
    a = LeafNode('a', ao)
    b = LeafNode('b', bo)

    # Leafnode eq
    ne_(a, b)
    ok_(not a == b)
    # Plus eq
    assert_eq_and_hash(add(a, a), add(a, a))
    assert_eq_and_hash(add(a, b), add(b, a)) # commutativity of add
    ne_(add(a, a), add(a, b))
    # More complicated eq
    assert_eq_and_hash(add(a, a, conjugate_transpose(mul(a, b))),
                       add(a, a, conjugate_transpose(mul(a, b))))
    ne_(add(a, a, conjugate_transpose(mul(a, b))),
        add(a, a, conjugate_transpose(mul(b, b))))
    ne_(add(a, a, conjugate_transpose(mul(a, b))),
        add(a, a, conjugate_transpose(mul(a, inverse(b)))))

def test_metadata_tree():
    raise SkipTest()
    class A(MockImpl):
        _sort_id = 1
    ao = A()
    class B(MockImpl):
        _sort_id = 2
    bo = B()
    a = LeafNode('a', ao)
    b = LeafNode('b', bo)

    m1 = add(a, b).metadata_tree()
    m2 = add(a, b).metadata_tree()
    m3 = add(b, a).metadata_tree() # reverse order in add()

    d1 = mul(a, b).metadata_tree()
    d2 = mul(b, a).metadata_tree()

    yield assert_eq_and_hash, m1, m2
    yield assert_eq_and_hash, m2, m3
    yield ok_, d1 != d2
    
def test_comparison():
    class A(MockImpl):
        _sort_id = 1
    class B(MockImpl):
        _sort_id = 2
    ao = A()
    bo = B()
    a = LeafNode('a', ao)
    b = LeafNode('b', bo)

    yield ok_, add(a, b) == add(b, a)
    yield ok_, add(mul(a, a), a) == add(mul(a, a), a)
    yield ok_, add(mul(a, a), a) != add(mul(conjugate_transpose(a), a), a)
    yield ok_, add(a, a) > mul(a, a)
    yield ok_, not add(a, a) < mul(a, a)
