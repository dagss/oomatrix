from ..function import Function
from . import mock_universe as mock

def test_hashing():
    ctx, (A, a) = mock.create_mock_matrices('A')
    f = mock.mock_computation(A)
    g = mock.mock_computation(A)
    meta_A = mock.mock_meta(A)
    
    h1 = Function(34, meta_A, (f, 0, 1, (g, 0), (g, (g, 2))))
    h2 = Function(34, meta_A, (f, 0, 1, (g, 0), (g, (g, 2))))
    j = Function(34, meta_A, (g, 0, 1, (g, 0), (g, (g, 2))))
    assert h1 == h2
    assert h1 != j
    assert hash(h1) == hash(h2)
    assert h1.secure_hash() == h2.secure_hash()
    assert h1.secure_hash() != j.secure_hash()
    assert h1.arg_count == h2.arg_count == j.arg_count == 3
