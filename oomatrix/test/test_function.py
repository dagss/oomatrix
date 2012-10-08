from ..function import Function
from . import mock_universe as mock

def test_hashing():
    ctx, (A, a) = mock.create_mock_matrices('A')
    meta_A = mock.mock_meta(A)
    f = Function.create_from_computation(mock.mock_computation(A), [meta_A, meta_A], meta_A)
    g = Function.create_from_computation(mock.mock_computation(A), [meta_A, meta_A], meta_A)
    
    h1 = Function(f, 0, 1, (g, 0), (g, (g, 2)))
    h2 = Function(f, 0, 1, (g, 0), (g, (g, 2)))
    j = Function(g, 0, 1, (g, 0), (g, (g, 2)))
    assert h1 == h2
    assert h1 != j
    assert hash(h1) == hash(h2)
    assert h1.secure_hash() == h2.secure_hash()
    assert h1.secure_hash() != j.secure_hash()
    assert h1.arg_count == h2.arg_count == j.arg_count == 3

    #x = Function(h1, 0, 1, (h2, 2))
    #y = Function(x, (h1, 2), (x, 3), (h2, 0), (j, 1))
    #print
    #print y

def test_remove_identity():
    ctx, (A, a) = mock.create_mock_matrices('A')
    meta_A = mock.mock_meta(A)
    f = Function.create_from_computation(mock.mock_computation(A), [meta_A, meta_A], meta_A)
    i = Function.create_identity(meta_A)

    a = Function((f, 0, 1))
    b = Function((f, (i, 0), (i, (i, (i, 1)))))
    assert a == b

