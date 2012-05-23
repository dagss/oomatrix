from .common import *

from ..metadata import MatrixMetadata

KIND_A = 1
KIND_B = 2

def test_ordering():
    m1 = MatrixMetadata(KIND_A, (2, 3), (3, 4), np.double)
    m2 = MatrixMetadata(KIND_A, (2, 3), (3, 4), np.double)
    n = MatrixMetadata(KIND_B, (2, 3), (3, 4), np.double)
    o = MatrixMetadata(KIND_B, (2, 3), (3, 5), np.double)
    assert_eq_and_hash(m1, m2)
    assert m1 < n
    assert not n < m1
    assert m1 != n
    assert not m1 != m2
    assert n < o
    assert n != o


    
