import numpy as np
from nose.tools import ok_, eq_, assert_raises
from textwrap import dedent

from .. import Matrix

De = Matrix('De', np.arange(9).reshape(3, 3).astype(np.int64))
Di = Matrix('Di', np.arange(3).astype(np.int64), diagonal=True)

def assert_repr(fact, test):
    fact = dedent(fact)
    if not fact == test:
        print 'GOT: ---'
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
    
    print De + De
    print De + Di
