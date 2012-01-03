from nose.tools import ok_, eq_, assert_raises
from numpy.testing import assert_almost_equal
import numpy as np

from ..shortest_path import find_shortest_path

def test_find_shortest_path():
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'
    f = 'f'
    g = 'g'
    h = 'h'

    graph = {a : [(b, 1, 'ab'), (c, 2, 'ac')], b : [(a, 1, 'ba'), 
            (d, 1, 'bd')], c : [(a, 2, 'ca'), (d, 2, 'cd')], d : [(c, 2, 'dc'), 
            (b, 1, 'db')]}
    path = find_shortest_path(graph, a, [d])
    yield eq_, path, ['ab', 'bd']

    graph = {a : [(b, 2, 'ab'), (c, 1, 'ac')], b : [(a, 2, 'ba'), (d, 2, 'bd')],
            c : [(a, 1, 'ca'), (d, 1, 'cd')], d : [(c, 1, 'dc'), (b, 2, 'db')]}
    path = find_shortest_path(graph, a, [d])
    yield eq_, path, ['ac', 'cd']

    paths = [['ac', 'cd'], ['ab', 'bd']]
    graph = {a : [(b, 1, 'ab'), (c, 1, 'ac')], b : [(a, 1, 'ba'), (d, 1, 'bd')],
            c : [(a, 1, 'ca'), (d, 1, 'cd')], d : [(c, 1, 'dc'), (b, 1, 'db')]}
    path = find_shortest_path(graph, a, [d])
    yield ok_, path in paths

    graph = {a : [(b, 1, 'ab'), (c, 10, 'ac')], b : [(a, 1, 'ba'), (c, 1, 'bc'),
        (d, 10, 'bd')], c : [(a, 10, 'ca'), (b, 1, 'cb'), (d, 1, 'cd')], 
        d : [(c, 1, 'dc'), (b, 10, 'db')]}
    path = find_shortest_path(graph, a, [d])
    yield eq_, path, ['ab', 'bc', 'cd']

    graph = {a : [(b, 1, 'ab'), (c, 2, 'ac')], b : [(d, 1, 'bd')], 
            c : [(d, 2, 'cd')], d : []}
    path = find_shortest_path(graph, a, [d])
    yield eq_, path, ['ab', 'bd']

    graph = {a : []}
    path = find_shortest_path(graph, a, [a])
    yield eq_, path, [None]

    graph = {a : [(b, 1, 'ab'), (c, 2, 'ac')], b : [(a, 1, 'ba'), (d, 1, 'bd')],
            c : [(a, 2, 'ca'), (d, 2, 'cd')], d : [(c, 2, 'dc'), (b, 1, 'db')]}
    path = find_shortest_path(graph, a, [c])
    yield eq_, path, ['ac']

    graph = {a : [(b, 1, 'ab'), (c, 10, 'ac')], b : [(a, 1, 'ba'), (c, 1, 'bc'),
        (d, 10, 'bd')], c : [(a, 10, 'ca'), (b, 1, 'cb'), (d, 1, 'cd')], 
        d : [(c, 1, 'dc'), (b, 10, 'db')]}
    path = find_shortest_path(graph, a, [c])
    yield eq_, path, ['ab', 'bc']

    def func():
        graph = {a : [(b, 1, 'ab'), (c, 1, 'ac')], b : [], c : []}
        path = find_shortest_path(graph, a, [b, c])
    yield assert_raises, ValueError, func

    graph = {a : [(b, 1, 'ab'), (c, 2, 'ac')], b : [], c : []}
    path = find_shortest_path(graph, a, [b, c])
    yield eq_, path, ['ab']

    def func():
        #No way to get to c - can only get *from* c
        graph = {a : [(b, 1, 'ab')], b : [], c : [(a, 1, 'ca')]}
        path = find_shortest_path(graph, a, [c])
    yield assert_raises, ValueError, func

    graph = {a : [(b, 1, 'ab'), (c, 2, 'ac'), (d, 1, 'ad')], b : [(a, 1, 'ba'),
            (e, 1, 'be')], c : [(a, 1, 'ca'), (e, 1, 'ce')], d : [(a, 1, 'da'),
            (e, 1, 'de')], e : [(b, 1, 'eb'), (c, 1, 'ec'), (d, 1, 'ed')]}
    path = find_shortest_path(graph, a, [b, c])
    yield eq_, path, ['ab']

    #Should be able to follow a straight line
    graph = {a : [(b, 1, 'ab')], b : [(c, 1, 'bc')], c : []}
    path = find_shortest_path(graph, a, [c])
    yield eq_, path, ['ab', 'bc']

