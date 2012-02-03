from ...test.common import *
from ... import Matrix, identity_matrix

n = 4
one = identity_matrix(n)

def test_basic():
    alleq_(2 * np.ones(4), (one + one).diagonal())
    alleq_(np.ones(4), (one * one.h).diagonal())
