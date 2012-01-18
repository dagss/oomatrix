from .common import *
from ..kind import *



class Dense(MatrixImpl):
    _sort_id = 1
        
class Diagonal(MatrixImpl):
    _sort_id = 2

def assert_key(expected, pattern):
    eq_(expected, pattern.get_key())

def test_basic():
    assert isinstance(Dense, MatrixKind)
    assert Dense < Diagonal
    assert Dense != Diagonal
    assert Dense == Dense

def test_tree_building():
    assert isinstance(Dense + Dense, AddPatternNode)

    p = (Diagonal.i.h + Dense.h) * Diagonal * Dense
    eq_(('*',
         ('+', ('h', ('i', Diagonal)),
               ('h', Dense)),
         Diagonal,
         Dense),
        p.get_key())


def test_tree_normalization():
    # Only A.i and A.i.h allowed for .i and .h
    yield assert_raises, IllegalPatternError, getattr, Dense.h, 'h'
    yield assert_raises, IllegalPatternError, getattr, Dense.h, 'i'
    yield assert_not_raises, getattr, Dense, 'i'
    yield assert_not_raises, getattr, Dense.i, 'h'
    yield (assert_raises, IllegalPatternError,
           getattr, (Dense + Dense), 'h')
    yield (assert_raises, IllegalPatternError,
           getattr, (Dense + Dense), 'i')
    # no nested adds
    yield ok_, all(child is Dense for child in
                   (Dense + Dense + Dense + Dense).children)
    # no nested muls
    yield ok_, all(child is Dense for child in
                   (Dense * Dense * Dense * Dense).children)
