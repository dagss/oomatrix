from .common import *
from .mock_universe import mock_meta, create_mock_matrices

from ..compiler import CompiledNode

from .. import scheduler, matrix

class MockComputation(object):
    def __init__(self, name):
        self.name = name

ctx, (A, a) = create_mock_matrices('A')

def mock_compiled_node(computation_name, children=(), shuffle=None, flat_shuffle=None):
    return CompiledNode(MockComputation(computation_name), 1.0, children, mock_meta(A),
                        shuffle=shuffle, flat_shuffle=flat_shuffle)

def mock_leaf():
    return CompiledNode.create_leaf(mock_meta(A))

def test_basic():
    a2 = matrix.Matrix(A(4, 3, 3), name='a')

    leaf = mock_leaf()
    add_cnode = mock_compiled_node("adder", [leaf, leaf])
    root = mock_compiled_node("multiplier", [add_cnode, add_cnode])
    s = scheduler.BasicScheduler()
    assert repr(s.schedule(root, [a, a, a, a])) == dedent('''\
        <oomatrix.Program:[
          T0 = adder(a, a) # cost=1.0
          $result = multiplier(T0, T0) # cost=1.0
        ]>''')
    assert repr(s.schedule(root, [a, a, a, a2])) == dedent('''\
        <oomatrix.Program:[
          T0 = adder(a, a) # cost=1.0
          T1 = adder(a, a_1) # cost=1.0
          $result = multiplier(T0, T1) # cost=1.0
        ]>''')
    
    
