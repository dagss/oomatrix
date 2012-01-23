from .common import *
from ..kind import *



class Dense(MatrixImpl):
    _sort_id = 1
        
class Diagonal(MatrixImpl):
    _sort_id = 2

class CSC(MatrixImpl):
    _sort_id = 3

def assert_key(expected, pattern):
    eq_(expected, pattern.get_key())

def test_basic():
    assert isinstance(Dense, MatrixKind)

def test_ordering():
    # kinds
    yield ok_, Dense < Diagonal
    yield ok_, Dense != Diagonal
    yield ok_, Dense == Dense

    # expressions always larger than kinds
    yield ok_, Dense + Dense > Diagonal
    yield ok_, Diagonal < Dense + Dense

    # since we use @total_ordering, we're going to trust that !=, <=, >, >=
    # works as advertised without tests...

    # one operator
    yield ok_, Dense + Dense < Diagonal + Diagonal
    yield ok_, not Dense + Dense == Diagonal + Diagonal
    yield ok_, Diagonal * Diagonal == Diagonal * Diagonal
    yield ok_, Diagonal * Diagonal < Diagonal * Diagonal * Diagonal
    yield ok_, Diagonal * Diagonal < Diagonal * Diagonal * Diagonal
    yield ok_, Diagonal * Diagonal > Dense * Diagonal * Diagonal
    yield ok_, Diagonal * Diagonal > Dense * Diagonal

    # mixing operators
    yield ok_, Diagonal * Diagonal > Dense * Dense
    # but:
    yield ok_, Diagonal * Diagonal < Dense + Dense
    yield ok_, Diagonal * Diagonal < Dense.h * Dense
    yield ok_, Diagonal.h * Diagonal < Dense.i * Dense
    yield ok_, Diagonal.i * Diagonal > Dense.h * Dense
    yield ok_, Dense.h > Diagonal + Diagonal

    # sorting of three types
    yield eq_, [Dense, Diagonal, CSC], sorted([CSC, Dense, Diagonal])
    
def test_add_permutation():
    yield eq_, [0, 1], (Dense + Dense).child_permutation
    yield eq_, [1, 0], (Diagonal + Dense).child_permutation
    yield eq_, [0, 1], (Dense + Diagonal).child_permutation
    yield eq_, [1, 2, 0], (Diagonal + Dense + Dense).child_permutation
    yield eq_, [Dense, Diagonal, CSC], (CSC + Dense + Diagonal).sorted_children

def test_tree_building():
    assert isinstance(Dense + Dense, AddPatternNode)

    p = (Diagonal + Dense) * Diagonal * Dense
    eq_(('*',
         ('+', Dense, Diagonal), # note reversed order!
         Diagonal,
         Dense),
        p.get_key())

    p = (Diagonal.i.h + Dense.h) * Diagonal * Dense
    eq_(('*',
         ('+', ('h', Dense),
               ('h', ('i', Diagonal))),
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

def test_kind_universe():
    class A(MatrixImpl): pass
    class C(MatrixImpl): pass
    class B(MatrixImpl): pass

    yield eq_, set([A]), A.universe.get_kinds()
    A + B # get A and B to know one another
    yield eq_, set([A, B]), A.universe.get_kinds()
    yield eq_, set([A, B]), B.universe.get_kinds()
    yield eq_, set([C]), C.universe.get_kinds()
    # get C and the rest to know each other in a convoluted way
    ((C.i.h + C) + C) * B
    yield eq_, set([A, B, C]), A.universe.get_kinds()
    yield eq_, set([A, B, C]), C.universe.get_kinds()
    yield eq_, set([A, B, C]), B.universe.get_kinds()
    
def test_kind_universe_links_behaviour():
    # closer inspection of the length of links
    def depth(node):
        if node._linked is None:
            return 0
        else:
            return depth(node._linked) + 1

    cur = first = MatrixKindUniverse()
    for i in range(5):
        newroot = MatrixKindUniverse()
        newroot.join_with(cur)
        cur = newroot

    # first knows nothing of all the stuff that's been going on since...
    assert 5 == depth(first)
    # but looking up the root collapses the linked list
    assert first._get_root() is cur
    assert 1 == depth(first)

