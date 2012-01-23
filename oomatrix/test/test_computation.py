from .common import *

from ..kind import *
from ..computation import *


def test_single_arg_computation():
    class A(MatrixImpl): pass
    class B(MatrixImpl): pass
    @computation(A, B)
    def convert_A_to_B(x):
        return 3
    assert A.universe._get_root() is B.universe._get_root()
    assert A.universe.get_computations(A) == {B : [convert_A_to_B]}
    

def test_conversion_method_decorator():
    class B(MatrixImpl): pass
    class A(MatrixImpl):
        @conversion(B)
        def to_B(self):
            return 'B'
    conv = A.universe.get_computations(A)[B][0]
    assert 'B' == conv.compute(A())


def test_decorate_class():
    class A(MatrixImpl): pass
    class B(MatrixImpl): pass
    @computation(A + B, A)
    class A_plus_B:
        @staticmethod
        def compute(a, b):
            return a + b
        @staticmethod
        def cost(a, b):
            pass
    obj = A.universe.get_computations((A + B).get_key())[A][0]
    assert obj is A_plus_B
    assert 3 == obj.compute(1, 2)


def test_decorate_function():
    class A(MatrixImpl): pass
    class B(MatrixImpl): pass
    @computation(A + B, A)
    def A_plus_B(a, b):
        return a + b
    obj = A.universe.get_computations((A + B).get_key())[A][0]
    assert 3 == obj.compute(1, 2)

