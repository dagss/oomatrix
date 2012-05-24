from functools import total_ordering
from ..heap import Heap

@total_ordering
class Value(object):
    def __init__(self, value):
        self.value = value
        
    def __lt__(self, other):
        assert False

    def __eq__(self, other):
        assert False

def test_basic():
    heap = Heap()
    heap.push(10, Value('a'))
    heap.push(5, Value('b'))
    heap.push(5, Value('c'))

    cost, item = heap.pop()
    assert (cost, item.value) == (5, 'b')
    cost, item = heap.pop()
    assert (cost, item.value) == (5, 'c')
    cost, item = heap.pop()
    assert (cost, item.value) == (10, 'a')
