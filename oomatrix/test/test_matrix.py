import numpy as np
from nose.tools import ok_, eq_, assert_raises
from textwrap import dedent

from .. import Matrix

De = Matrix('De', np.arange(9).reshape(3, 3).astype(np.int64))
Di = Matrix('Di', np.arange(3).astype(np.int64), diagonal=True)

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
    yield test, '(De + Di).H', (De + Di).H
    yield test, 'De + Di.H', De + Di.H
    yield test, 'De + Di.H * Di', De + Di.H * Di
    yield test, '(De + Di.H) * Di', (De + Di.H) * Di
    yield test, 'De', De.H.H.H.H, dict(De=De)
    yield test, 'De', De.I.I.I.I, dict(De=De)
    yield test, 'De.I * Di', De.I * Di
    yield test, 'De.H.I * Di', De.H.I * Di
    yield test, 'De.H.I * Di', De.I.H * Di
    yield test, 'De.H * Di', De.I.H.I * Di
    yield test, '(De + Di).I', (De + Di).H.I.H
