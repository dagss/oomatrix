from nose.tools import ok_, eq_, assert_raises
from numpy.testing import assert_almost_equal
import numpy as np

from shortest_path import find_shortest_path

def test_find_shortest_path():
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'
    f = 'f'
    g = 'g'
    h = 'h'
    graph = {a : [(b, 1), (c, 2)], b : [(a, 1), (d, 1)], c : [(a, 2), (d, 2)],
             d : [(c, 2), (b, 1)]}
    path = find_shortest_path(graph, a, [d])
    yield eq_, path, [a, b, d]

    graph = {a : [(b, 2), (c, 1)], b : [(a, 2), (d, 2)], c : [(a, 1), (d, 1)],
             d : [(c, 1), (b, 2)]}
    path = find_shortest_path(graph, a, [d])
    yield eq_, path, [a, c, d]

    paths = [[a, c, d], [a, b, d]]
    graph = {a : [(b, 1), (c, 1)], b : [(a, 1), (d, 1)], c : [(a, 1), (d, 1)],
             d : [(c, 1), (b, 1)]}
    path = find_shortest_path(graph, a, [d])
    yield ok_, path in paths

    graph = {a : [(b, 1), (c, 10)], b : [(a, 1), (c, 1), (d, 10)], 
            c : [(a, 10), (b, 1), (d, 1)], d : [(c, 1), (b, 10)]}
    path = find_shortest_path(graph, a, [d])
    yield eq_, path, [a, b, c, d]

    graph = {a : [(b, 1), (c, 2)], b : [(d, 1)], c : [(d, 2)],
            d : []}
    path = find_shortest_path(graph, a, [d])
    yield eq_, path, [a, b, d]

    graph = {a : []}
    path = find_shortest_path(graph, a, [a])
    yield eq_, path, [a]

    graph = {a : [(b, 1), (c, 2)], b : [(a, 1), (d, 1)], c : [(a, 2), (d, 2)],
             d : [(c, 2), (b, 1)]}
    path = find_shortest_path(graph, a, [c])
    yield eq_, path, [a, c]

    graph = {a : [(b, 1), (c, 10)], b : [(a, 1), (c, 1), (d, 10)], 
            c : [(a, 10), (b, 1), (d, 1)], d : [(c, 1), (b, 10)]}
    path = find_shortest_path(graph, a, [c])
    yield eq_, path, [a, b, c]

    def func():
        graph = {a : [(b, 1), (c, 1)], b : [], c : []}
        path = find_shortest_path(graph, a, [b, c])
    yield assert_raises, ValueError, func

    graph = {a : [(b, 1), (c, 2)], b : [], c : []}
    path = find_shortest_path(graph, a, [b, c])
    yield eq_, path, [a, b]

    def func():
        #No way to get to c - can only get *from* c
        graph = {a : [(b, 1)], b : [], c : [(a, 1)]}
        path = find_shortest_path(graph, a, [c])
    yield assert_raises, ValueError, func

    graph = {a : [(b, 1), (c, 2), (d, 1)], b : [(a, 1), (e, 1)], 
            c : [(a, 1), (e, 1)], d : [(a, 1), (e, 1)], e : [(b, 1), (c, 1), 
                (d, 1)]}
    path = find_shortest_path(graph, a, [b, c])
    yield eq_, path, [a, b]
