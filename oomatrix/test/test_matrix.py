import numpy as np
from nose.tools import ok_, eq_, assert_raises
from textwrap import dedent

from .. import Matrix, Vector, compute, describe

De_array = np.arange(9).reshape(3, 3).astype(np.int64)
De = Matrix('De', De_array)
Di_array = np.arange(3).astype(np.int64)
Di = Matrix('Di', Di_array, diagonal=True)

def assert_repr(fact, test):
    fact = dedent(fact)
    if not fact == test:
        print 'EXPECTED:'
        print fact
        print 'GOT:'
        print test
        print '---'
    eq_(fact, test)

def test_basic():
    yield assert_repr, """\
    3-by-3 row-major matrix 'De' of int64
    [0 1 2]
    [3 4 5]
    [6 7 8]""", repr(De)

    yield assert_repr, """\
    3-by-3 diagonal matrix 'Di' of int64
    [0 0 0]
    [0 1 0]
    [0 0 2]""", repr(Di)

    yield assert_repr, """\
    3-by-3 matrix of int64 given by:
    
        De + De
    
    where

        De: 3-by-3 row-major matrix of int64""", repr(De + De)

    yield assert_repr, """\
    3-by-3 matrix of int64 given by:
    
        De + Di
    
    where

        De: 3-by-3 row-major matrix of int64
        Di: 3-by-3 diagonal matrix of int64""", repr(De + Di)

def test_symbolic():
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
    yield test, 'De.h.i * Di', De.h.i * Di
    yield test, 'De.h.i * Di', De.i.h * Di
    yield test, 'De.h * Di', De.i.h.i * Di
    yield test, '(De + Di).i', (De + Di).h.i.h

def test_matvec():
    a = np.arange(3)
    yield ok_, type(De * a) is Vector
    yield ok_, type(compute(De * a)) is np.ndarray
    yield ok_, np.all(compute(De * a) == np.dot(De_array, a))

    describe(De * (Di + Di) * a)
    yield ok_, np.all(compute(De * (Di + Di) * a) == np.dot(De_array, Di_array * a + Di_array * a))

    
