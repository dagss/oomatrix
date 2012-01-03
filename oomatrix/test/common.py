import numpy as np
from nose.tools import ok_, eq_, assert_raises
from textwrap import dedent

def ndrange(shape):
    return np.arange(np.prod(shape)).reshape(shape).astype(np.double)
