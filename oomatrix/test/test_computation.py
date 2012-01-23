from .common import *

from ..kind import *
from ..computation import *



def test_basic():
    class A(MatrixImpl): pass
    class B(MatrixImpl): pass

    @computation(A + B, A)
    class A_plus_B:
        @staticmethod
        def compute(a, b):
            pass

        @staticmethod
        def cost(a, b):
            pass

    cls = A.universe.get_computations((A + B).get_key())[A][0]
    assert cls is A_plus_B
    
    cls.compute(1,2)
