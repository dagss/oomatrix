from .common import *

from ..kind import *
from ..computation import *
from ..cost_value import zero_cost

def test_computation_registration():
    class A(MatrixImpl): pass
    class B(MatrixImpl): pass
    @computation(A, B, cost=0)
    def convert_A_to_B(x):
        return 3
    assert A.universe._get_root() is B.universe._get_root()
    assert A.universe.get_computations(A) == {B : [convert_A_to_B]}
    
def test_add():
    # addition needs to compute a permutation of the arguments so that
    # the computation can be invoked in sorted order
    class A(MatrixImpl):
        _sort_id = 1
    class B(MatrixImpl):
        _sort_id = 2
    class C(MatrixImpl):
        _sort_id = 3
    @computation(B + C + A, A, cost=0)
    def add(b, c, a):
        # PS: Use a non-commutative computation in order to track
        # which order arguments were passed in
        return 100 * a + 10 * b + 1 * c
    assert {A: [add]} == A.universe.get_computations((A + B + C).get_key())
    # add.__call__ forwards arguments in same order
    assert 123 == add(2, 3, 1)
    # add.compute takes flattened, ordered expression
    assert 123 == add.compute([1, 2, 3])

def test_conversion_method_decorator():
    class B(MatrixImpl): pass
    class A(MatrixImpl):
        @conversion(B, cost=0)
        def to_B(self):
            return 'B'
    conv = A.universe.get_computations(A)[B][0]
    assert 'B' == conv.compute([A()])

def test_decorate_function():
    class A(MatrixImpl):
        _sort_id = 1
    class B(MatrixImpl):
        _sort_id = 2
    @computation(A + B, A, cost=0)
    def A_plus_B(a, b):
        return a + b
    obj = A.universe.get_computations((A + B).get_key())[A][0]
    assert 3 == obj.compute([1, 2])

