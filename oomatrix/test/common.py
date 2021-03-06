import numpy as np
from nose.tools import ok_, eq_, assert_raises
from nose import SkipTest
from textwrap import dedent
import contextlib
import os, sys, re
from pprint import pprint

def assert_eq_and_hash(a, b):
    eq_(a, b)
    eq_(hash(a), hash(b))

def assert_not_raises(func, *args):
    "Turn any exception into AssertionError"
    try:
        func(*args)
    except:
        raise AssertionError()

def alleq_(a, b):
    assert np.all(a == b)

def ndrange(shape, dtype=np.double):
    return np.arange(np.prod(shape)).reshape(shape).astype(dtype)

def ne_(a, b):
    ok_(a != b)

def plot_add_graph(graph, max_node_size=4, block=False):
    # Plot the addition graph, for use during debugging
    from ..graph.plot_graphs import plot_graph
    def format_vertex(v):
        names = [kind.name for kind in v]
        names.sort()
        return dict(label=' + '.join(names), color='red' if len(v) == 1 else 'black')
    def format_edge(cost, payload):
        return '%s %.2f' % (payload[0], cost)
    plot_graph(graph,
               max_node_size=max_node_size,
               format_vertex=format_vertex, format_edge=format_edge,
               block=block)

def plot_mul_graph(graph, block=False):
    # Plot the addition graph, for use during debugging
    from ..graph.plot_graphs import plot_graph
    def format_vertex(v):
        names = [kind.name for kind in v]
        names.sort()
        return dict(label=' '.join(names), color='red' if len(v) == 1 else 'black')
    def format_edge(cost, payload):
        return '%.2f' % cost
    plot_graph(graph,
               format_vertex=format_vertex, format_edge=format_edge,
               block=block)
