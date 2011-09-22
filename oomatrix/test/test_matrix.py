import numpy as np
from nose.tools import ok_, eq_, assert_raises

from .. import Matrix


De = Matrix('De', np.arange(9).reshape(3, 3).astype(np.int64))
Di = Matrix('Di', np.arange(3).astype(np.int64), diagonal=True)



def test_basic():
    yield eq_, "3-by-3 row-major matrix 'De' of int64", repr(De)
    yield eq_, "3-by-3 diagonal matrix 'Di' of int64", repr(Di)

