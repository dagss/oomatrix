from .common import *

from .. import Matrix, compute, explain
from nose import SkipTest

De_array = np.arange(9).reshape(3, 3).astype(np.int64)
De = Matrix(De_array, 'De')
Di_array = np.arange(3).astype(np.int64)
Di = Matrix(Di_array, 'Di', diagonal=True)

def assert_repr(fact, object_to_repr):
    fact = dedent(fact)
    r = repr(object_to_repr)
    if not fact == r:
        print 'EXPECTED:'
        print fact
        print 'GOT:'
        print r
        print '---'
    assert fact == r

def test_repr():
    yield assert_repr, """\
    3-by-3 row-major matrix 'De' of int64
    [0 1 2]
    [3 4 5]
    [6 7 8]""", De

    yield assert_repr, """\
    3-by-3 diagonal matrix 'Di' of int64
    [0 0 0]
    [0 1 0]
    [0 0 2]""", Di

    yield assert_repr, """\
    3-by-3 matrix of int64 given by:
    
        De + De
    
    where

        De: 3-by-3 row-major matrix of int64""", De + De

    yield assert_repr, """\
    3-by-3 matrix of int64 given by:
    
        De + Di
    
    where

        De: 3-by-3 row-major matrix of int64
        Di: 3-by-3 diagonal matrix of int64""", De + Di

    yield assert_repr, """\
    3-by-3 matrix of int64 given by:

        De.h

    where

        De: 3-by-3 row-major matrix of int64""", De.h

    yield assert_repr, """\
    3-by-3 matrix of int64 given by:

        De.h * Di

    where

        De: 3-by-3 row-major matrix of int64
        Di: 3-by-3 diagonal matrix of int64""", De.h * Di

    yield assert_repr, '''\
    4-by-1 row-major matrix 'Foo' of float64
    [1.0]
    [1.0]
    [1.0]
    [1.0]''', Matrix(np.ones((4, 1)), 'Foo')

    yield assert_repr, '''\
    4-by-1 row-major matrix of float64
    [1.0]
    [1.0]
    [1.0]
    [1.0]''', Matrix(np.ones((4, 1)))

    yield (assert_repr, '''\
    4-by-1 matrix of float64 given by:

        $0 + $1

    where

        $0: 4-by-1 row-major matrix of float64
        $1: 4-by-1 row-major matrix of float64''',
           Matrix(np.ones((4, 1))) + Matrix(np.ones((4, 1))))


def test_symbolic():
    raise SkipTest()
    namemap0 = dict(De=De, Di=Di)

    def test(exprstr0, expr, namemap0=namemap0):
        exprstr, namemap = expr.format_expression()
        eq_(exprstr0, exprstr)
        eq_(namemap0, namemap)
        
    yield test, 'De + Di', De + Di
    yield test, 'Di + De', Di + De
    yield test, '(De + Di).h', (De + Di).h
    yield test, 'De + Di.h', De + Di.h
    yield test, 'De + Di.h * Di', De + Di.h * Di
    yield test, '(De + Di.h) * Di', (De + Di.h) * Di
    yield test, 'De', De.h.h.h.h, dict(De=De)
    yield test, 'De', De.i.i.i.i, dict(De=De)
    yield test, 'De.i * Di', De.i * Di
    yield test, 'De.i.h * Di', De.h.i * Di
    yield test, 'De.i.h * Di', De.i.h * Di
    yield test, 'De.h * Di', De.i.h.i * Di
    yield test, '(De + Di).i', (De + Di).h.i.h

def test_matvec():
    raise SkipTest()
    a = np.arange(3)
    yield ok_, type(De * a) is Vector
    yield ok_, type(compute(De * a)) is np.ndarray
    yield ok_, np.all(compute(De * a) == np.dot(De_array, a))
    yield ok_, np.all(compute(De * (Di + Di) * a) == np.dot(De_array, Di_array * a + Di_array * a))

    #explain(De * (Di + Di) * a)

    
def test_decompositions():
    F = Di.factor()
    #print compute(F)
    #print compute(compute(F) * compute(F).h)
    
