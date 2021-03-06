from .common import *
from ..utils import *

def test_argsort():
    x = [1, 2, 6, 3, 45, 7, 56, 23, 64, 3, -34]
    y = [x[i] for i in argsort(x)]
    eq_(sorted(x), y)

def test_invert_permutation():
    yield eq_, [0, 1], invert_permutation([0, 1])
    yield eq_, [1, 2, 0], invert_permutation([2, 0, 1])

def test_sort_by():
    yield eq_, [], sort_by([], [])
    yield eq_, ['a', 'b'], sort_by(['a', 'b'], [4, 10])
    yield eq_, ['a', 'b'], sort_by(['b', 'a'], [10, 4])
    yield eq_, ['a', 'b'], sort_by(['a', 'b'], [10, 4], reverse=True)


