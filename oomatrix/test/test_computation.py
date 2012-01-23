from .common import *

from ..kind import *
from ..computation import *



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

