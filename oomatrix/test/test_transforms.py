from .common import *

from .. import symbolic, kind, transforms

def add(*args):
    return symbolic.add(args)

def mul(*args):
    return symbolic.multiply(args)

def H(arg):
    return symbolic.conjugate_transpose(arg)

def I(arg):
    return symbolic.inverse(arg)

class MockImpl(kind.MatrixImpl):
    nrows = ncols = 3
    dtype = None
    
class A(MockImpl):
    _sort_id = 1
    
class B(MockImpl):
    _sort_id = 2
    
class C(MockImpl):
    _sort_id = 3
    
ao = A()
bo = B()
co1 = C()
co2 = C()

a = symbolic.LeafNode('a', ao)
b = symbolic.LeafNode('b', bo)
c1 = symbolic.LeafNode('c1', co1)
c2 = symbolic.LeafNode('c2', co2)
c2.ncols = 10

def test_metadata_transform_sorted_by_kind():
    tree, args = transforms.metadata_transform(add(c1, b, a))
    assert [a, b, c1] == args
    assert [0, 1, 2] == [x.leaf_index for x in tree.children]

def test_metadata_transform_sorted_by_shape():
    # sorted by kind
    tree, args = transforms.metadata_transform(add(c2, c1))
    assert [c1, c2] == args
    assert [0, 1] == [x.leaf_index for x in tree.children]

def test_kind_key_transform():
    def process(tree):
        tree, args = transforms.metadata_transform(tree)
        return transforms.kind_key_transform(tree)
    
    class A(MockImpl):
        _sort_id = 1

    class B(MockImpl):
        _sort_id = 2

    a = symbolic.LeafNode('a', A())
    b = symbolic.LeafNode('b', B())
    key, universe = process(mul(add(I(H(b)), b, a), a))
    # note that the + is sorted
    eq_(('*',
         ('+', A, B, ('h', ('i', B))),
         A), key)

    # reverse sort order and construct new tree, now the order should be B, A
    old_sort = A._sort_id
    try:
        A._sort_id = 3
        key, universe = process(mul(add(I(H(b)), b, a), a))
        eq_(('*',
             ('+', B, A, ('h', ('i', B))),
             A), key)
    finally:
        A._sort_id = old_sort

