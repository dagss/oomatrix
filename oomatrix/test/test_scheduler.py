from .common import *
from .mock_universe import mock_meta, create_mock_matrices

from ..cost_value import FLOP
from ..function import Function

from .. import scheduler, matrix, computation

class MockComputation(computation.Computation):
    def __init__(self, name, arg_count):
        self.name = name
        self.arg_count = arg_count

    def get_cost(self, metas):
        return 1 * FLOP

ctx, (A, a) = create_mock_matrices('A')

def mock_function(computation_name, child_count):
    A_meta = mock_meta(A)
    return Function.create_from_computation(
        MockComputation(computation_name, child_count), [A_meta] * child_count, A_meta)

def test_basic():
    a2 = matrix.Matrix(A(4, 3, 3), name='a')

    f_add = mock_function("adder", 2)
    f_multiply = mock_function("multiplier", 2)

    func = Function((f_multiply, (f_add, 0, 1), (f_add, 2, 3)))
    
    s = scheduler.BasicScheduler()
    prog = s.schedule(func, [a, a, a, a])
    assert repr(prog) == dedent('''\
        <oomatrix.Program:[
          T0 = adder(a, a)
          $result = multiplier(T0, T0)
        ]>''')
    assert repr(s.schedule(func, [a, a, a, a2])) == dedent('''\
        <oomatrix.Program:[
          T0 = adder(a, a)
          T1 = adder(a, a_1)
          $result = multiplier(T0, T1)
        ]>''')

def test_unnamed_arg():
    anonymous = matrix.Matrix(A(4, 3, 3), name=None)
    f_add = mock_function("adder", 2)
    s = scheduler.BasicScheduler()
    got = s.schedule(f_add, [anonymous, anonymous])
    assert repr(got) == dedent('''\
        <oomatrix.Program:[
          $result = adder(input_0, input_0)
        ]>''')    
